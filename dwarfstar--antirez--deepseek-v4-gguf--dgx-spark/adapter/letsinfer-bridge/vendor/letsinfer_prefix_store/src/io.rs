// SPDX-License-Identifier: AGPL-3.0-only

//! Page-aligned, queue-depth record reads for the Let's Infer prefix store.
//!
//! Production Linux reads use `O_DIRECT`, four disjoint `pread` bands, and
//! page-aligned host memory. Small records are filled directly; large records
//! are CRC-checked through one bounded scratch buffer per band before an
//! untouched lazy mapping is returned. Startup indexing reads only the header
//! and exact token IDs.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::mem::MaybeUninit;
use std::ops::Range;
use std::os::fd::AsRawFd;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use anyhow::{bail, Context, Result};

use crate::format::{RecordHeader, ALIGNMENT, HEADER_BYTES};

const DIRECT_IO_BLOCK: usize = 4 * 1024 * 1024;
const DIRECT_READ_WORKERS: usize = 4;
pub(crate) const FILE_BACKED_RECORD_MIN_BYTES: usize = 64 * 1024 * 1024;

pub(crate) struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: ownership follows the buffer. Mutable access requires `&mut self`;
// resident-tier readers receive only an immutable `Arc`.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    fn new_uninit(len: usize) -> Result<Self> {
        if len == 0 || !len.is_multiple_of(ALIGNMENT) {
            bail!("aligned buffer length must be a non-zero page multiple");
        }
        let mut raw = std::ptr::null_mut();
        // SAFETY: `posix_memalign` writes one allocation pointer on success.
        let rc = unsafe { libc::posix_memalign(&mut raw, ALIGNMENT, len) };
        if rc != 0 {
            bail!("posix_memalign({len}) failed with errno {rc}");
        }
        let ptr = NonNull::new(raw.cast::<u8>())
            .ok_or_else(|| anyhow::anyhow!("posix_memalign returned null"))?;
        Ok(Self { ptr, len })
    }

    pub(crate) fn new_zeroed(len: usize) -> Result<Self> {
        let mut buffer = Self::new_uninit(len)?;
        buffer.as_mut_slice().fill(0);
        Ok(buffer)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: every immutable exposure follows complete initialization.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` provides exclusive access to the allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn as_uninit_mut_slice(&mut self) -> &mut [MaybeUninit<u8>] {
        // SAFETY: the exclusive borrow prevents observation until workers join.
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr().cast::<MaybeUninit<u8>>(), self.len)
        }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `posix_memalign` and is owned here.
        unsafe { libc::free(self.ptr.as_ptr().cast()) };
    }
}

/// Writable record staging backed by the store's NVMe filesystem instead of
/// anonymous RAM. Large engine snapshots can therefore be reclaimed by the
/// kernel while they are being persisted and do not consume a second copy of
/// the model's unified-memory safety margin.
#[cfg(target_os = "linux")]
pub(crate) struct MappedCapture {
    ptr: NonNull<u8>,
    len: usize,
    file: File,
    path: PathBuf,
    mapped: bool,
    remove_on_drop: bool,
}

#[cfg(target_os = "linux")]
unsafe impl Send for MappedCapture {}
#[cfg(target_os = "linux")]
unsafe impl Sync for MappedCapture {}

#[cfg(target_os = "linux")]
impl MappedCapture {
    pub(crate) fn new(path: PathBuf, len: usize) -> Result<Self> {
        if len == 0 || !len.is_multiple_of(ALIGNMENT) {
            bail!("mapped capture length must be a non-zero page multiple");
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .with_context(|| format!("create prefix capture {}", path.display()))?;
        file.set_len(u64::try_from(len)?)?;
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            let error = std::io::Error::last_os_error();
            let _ = fs::remove_file(&path);
            return Err(error).context("mmap prefix capture");
        }
        let ptr =
            NonNull::new(raw.cast::<u8>()).ok_or_else(|| anyhow::anyhow!("mmap returned null"))?;
        let _ = unsafe { libc::madvise(raw, len, libc::MADV_SEQUENTIAL) };
        Ok(Self {
            ptr,
            len,
            file,
            path,
            mapped: true,
            remove_on_drop: true,
        })
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn commit(mut self, root: &Path, destination: &Path) -> Result<()> {
        self.file.sync_all()?;
        fs::rename(&self.path, destination)?;
        OpenOptions::new().read(true).open(root)?.sync_all()?;
        self.remove_on_drop = false;
        let _ = unsafe { libc::madvise(self.ptr.as_ptr().cast(), self.len, libc::MADV_DONTNEED) };
        self.unmap();
        let _ =
            unsafe { libc::posix_fadvise(self.file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        Ok(())
    }

    fn unmap(&mut self) {
        if self.mapped && unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.len) } == 0 {
            self.mapped = false;
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for MappedCapture {
    fn drop(&mut self) {
        self.unmap();
        if self.remove_on_drop {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Immutable record view backed by the durable file. Large records are
/// validated with bounded direct I/O before this untouched lazy mapping opens.
#[cfg(target_os = "linux")]
pub(crate) struct MappedRecord {
    ptr: NonNull<u8>,
    len: usize,
    file: File,
}

#[cfg(target_os = "linux")]
unsafe impl Send for MappedRecord {}
#[cfg(target_os = "linux")]
unsafe impl Sync for MappedRecord {}

#[cfg(target_os = "linux")]
impl MappedRecord {
    fn open(path: &Path, len: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .with_context(|| format!("open mapped prefix record {}", path.display()))?;
        if usize::try_from(file.metadata()?.len())? != len {
            bail!("prefix-record file size changed before mapping");
        }
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if raw == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error()).context("mmap prefix record");
        }
        let ptr =
            NonNull::new(raw.cast::<u8>()).ok_or_else(|| anyhow::anyhow!("mmap returned null"))?;
        let _ = unsafe { libc::madvise(raw, len, libc::MADV_SEQUENTIAL) };
        Ok(Self { ptr, len, file })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

#[cfg(target_os = "linux")]
impl Drop for MappedRecord {
    fn drop(&mut self) {
        let _ = unsafe { libc::madvise(self.ptr.as_ptr().cast(), self.len, libc::MADV_DONTNEED) };
        let _ =
            unsafe { libc::posix_fadvise(self.file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        let _ = unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.len) };
    }
}

pub(crate) enum RecordBuffer {
    Memory(AlignedBuffer),
    #[cfg(target_os = "linux")]
    Mapped(MappedRecord),
}

impl RecordBuffer {
    pub(crate) fn from_memory(buffer: AlignedBuffer) -> Self {
        Self::Memory(buffer)
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Memory(buffer) => buffer.len(),
            #[cfg(target_os = "linux")]
            Self::Mapped(buffer) => buffer.len,
        }
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Memory(buffer) => buffer.as_slice(),
            #[cfg(target_os = "linux")]
            Self::Mapped(buffer) => buffer.as_slice(),
        }
    }

    #[cfg(all(test, target_os = "linux"))]
    fn is_file_backed(&self) -> bool {
        matches!(self, Self::Mapped(_))
    }
}

/// Read and validate one complete immutable record.
pub(crate) fn read_record_checked(
    path: &Path,
    expected: &RecordHeader,
    direct: bool,
) -> Result<RecordBuffer> {
    let file_bytes = usize::try_from(fs::metadata(path)?.len())?;
    if file_bytes != expected.file_bytes || !file_bytes.is_multiple_of(ALIGNMENT) {
        bail!(
            "prefix-record file size mismatch: got {file_bytes}, expected {}",
            expected.file_bytes
        );
    }

    let region_ranges = (0..expected.regions.len())
        .map(|index| {
            expected
                .region_range(index)
                .ok_or_else(|| anyhow::anyhow!("invalid region range {index}"))
        })
        .collect::<Result<Vec<_>>>()?;

    #[cfg(target_os = "linux")]
    if file_bytes >= FILE_BACKED_RECORD_MIN_BYTES {
        let mut options = OpenOptions::new();
        options.read(true);
        if direct {
            options.custom_flags(libc::O_DIRECT);
        }
        let file = options
            .open(path)
            .with_context(|| format!("open prefix record {}", path.display()))?;
        let checksums = checksum_file_regions(&file, path, file_bytes, &region_ranges, direct)?;
        if checksums != expected.region_crc32 {
            bail!("prefix-record region CRC mismatch");
        }
        let (loaded_header, _) = read_header_and_tokens(path)?;
        if &loaded_header != expected {
            bail!("prefix-record header changed after indexing");
        }
        let _ = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        drop(file);
        return MappedRecord::open(path, file_bytes).map(RecordBuffer::Mapped);
    }

    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    if direct {
        options.custom_flags(libc::O_DIRECT);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = direct;

    let file = options
        .open(path)
        .with_context(|| format!("open prefix record {}", path.display()))?;
    let bands = read_bands(file_bytes, DIRECT_READ_WORKERS, DIRECT_IO_BLOCK);
    let band_size = bands[0].len();
    let mut buffer = AlignedBuffer::new_uninit(file_bytes)?;

    let checksums = std::thread::scope(|scope| -> Result<Vec<u32>> {
        let mut readers = Vec::new();
        for (band, destination) in bands
            .iter()
            .cloned()
            .zip(buffer.as_uninit_mut_slice().chunks_mut(band_size))
        {
            let file = &file;
            let ranges = &region_ranges;
            readers.push(
                scope.spawn(move || read_band(file, path, destination, band, ranges, direct)),
            );
        }

        // Join in file-band order so `Hasher::combine` preserves byte order.
        let mut combined: Vec<Option<crc32fast::Hasher>> =
            (0..region_ranges.len()).map(|_| None).collect();
        for reader in readers {
            let parts = reader
                .join()
                .map_err(|_| anyhow::anyhow!("prefix-record read worker panicked"))??;
            for (target, part) in combined.iter_mut().zip(parts) {
                let Some(part) = part else { continue };
                if let Some(target) = target.as_mut() {
                    target.combine(&part);
                } else {
                    *target = Some(part);
                }
            }
        }
        combined
            .into_iter()
            .map(|value| {
                value
                    .map(crc32fast::Hasher::finalize)
                    .ok_or_else(|| anyhow::anyhow!("record region was not read"))
            })
            .collect()
    })?;

    if checksums != expected.region_crc32 {
        bail!("prefix-record region CRC mismatch");
    }

    let loaded_header = RecordHeader::decode(&buffer.as_slice()[..HEADER_BYTES])?;
    if &loaded_header != expected {
        bail!("prefix-record header changed after indexing");
    }
    let token_range = expected.tokens_range();
    if crc32fast::hash(&buffer.as_slice()[token_range]) != expected.tokens_crc32 {
        bail!("prefix-record token CRC mismatch");
    }
    Ok(RecordBuffer::Memory(buffer))
}

fn read_bands(total: usize, workers: usize, io_block: usize) -> Vec<Range<usize>> {
    debug_assert!(total > 0);
    debug_assert!(workers > 0);
    debug_assert!(io_block > 0 && io_block.is_multiple_of(ALIGNMENT));
    let target = total.div_ceil(workers).div_ceil(io_block) * io_block;
    (0..total)
        .step_by(target)
        .map(|start| start..(start + target).min(total))
        .collect()
}

fn checksum_file_regions(
    file: &File,
    path: &Path,
    file_bytes: usize,
    region_ranges: &[Range<usize>],
    direct: bool,
) -> Result<Vec<u32>> {
    let bands = read_bands(file_bytes, DIRECT_READ_WORKERS, DIRECT_IO_BLOCK);
    std::thread::scope(|scope| -> Result<Vec<u32>> {
        let mut readers = Vec::new();
        for band in bands {
            readers
                .push(scope.spawn(move || checksum_band(file, path, band, region_ranges, direct)));
        }
        let mut combined: Vec<Option<crc32fast::Hasher>> =
            (0..region_ranges.len()).map(|_| None).collect();
        for reader in readers {
            let parts = reader
                .join()
                .map_err(|_| anyhow::anyhow!("prefix-record checksum worker panicked"))??;
            for (target, part) in combined.iter_mut().zip(parts) {
                let Some(part) = part else { continue };
                if let Some(target) = target.as_mut() {
                    target.combine(&part);
                } else {
                    *target = Some(part);
                }
            }
        }
        combined
            .into_iter()
            .map(|value| {
                value
                    .map(crc32fast::Hasher::finalize)
                    .ok_or_else(|| anyhow::anyhow!("record region was not read"))
            })
            .collect()
    })
}

fn checksum_band(
    file: &File,
    path: &Path,
    band: Range<usize>,
    region_ranges: &[Range<usize>],
    direct: bool,
) -> Result<Vec<Option<crc32fast::Hasher>>> {
    let mut checksums: Vec<Option<crc32fast::Hasher>> =
        (0..region_ranges.len()).map(|_| None).collect();
    let mut scratch = AlignedBuffer::new_uninit(DIRECT_IO_BLOCK)?;
    for offset in (band.start..band.end).step_by(DIRECT_IO_BLOCK) {
        let len = (band.end - offset).min(DIRECT_IO_BLOCK);
        let destination = &mut scratch.as_uninit_mut_slice()[..len];
        read_chunk(file, path, destination, offset, direct)?;
        let initialized = unsafe {
            std::slice::from_raw_parts(destination.as_ptr().cast::<u8>(), destination.len())
        };
        for (index, region) in region_ranges.iter().enumerate() {
            let start = offset.max(region.start);
            let end = (offset + len).min(region.end);
            if start < end {
                checksums[index]
                    .get_or_insert_with(crc32fast::Hasher::new)
                    .update(&initialized[start - offset..end - offset]);
            }
        }
    }
    Ok(checksums)
}

fn read_band(
    file: &File,
    path: &Path,
    destination: &mut [MaybeUninit<u8>],
    band: Range<usize>,
    region_ranges: &[Range<usize>],
    direct: bool,
) -> Result<Vec<Option<crc32fast::Hasher>>> {
    let mut checksums: Vec<Option<crc32fast::Hasher>> =
        (0..region_ranges.len()).map(|_| None).collect();
    for (relative, chunk) in destination.chunks_mut(DIRECT_IO_BLOCK).enumerate() {
        let offset = band.start + relative * DIRECT_IO_BLOCK;
        read_chunk(file, path, chunk, offset, direct)?;
        let chunk_end = offset + chunk.len();
        // SAFETY: `read_chunk` initialized the entire chunk before this view.
        let initialized =
            unsafe { std::slice::from_raw_parts(chunk.as_ptr().cast::<u8>(), chunk.len()) };
        for (index, region) in region_ranges.iter().enumerate() {
            let start = offset.max(region.start);
            let end = chunk_end.min(region.end);
            if start < end {
                checksums[index]
                    .get_or_insert_with(crc32fast::Hasher::new)
                    .update(&initialized[start - offset..end - offset]);
            }
        }
    }
    Ok(checksums)
}

fn read_chunk(
    file: &File,
    path: &Path,
    destination: &mut [MaybeUninit<u8>],
    offset: usize,
    direct: bool,
) -> Result<()> {
    let mut done = 0usize;
    while done < destination.len() {
        // SAFETY: the destination suffix is writable for the requested count;
        // `pread` retains neither pointer nor file descriptor.
        let result = unsafe {
            libc::pread(
                file.as_raw_fd(),
                destination[done..].as_mut_ptr().cast::<libc::c_void>(),
                destination.len() - done,
                (offset + done) as libc::off_t,
            )
        };
        if result < 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("pread prefix record {}", path.display()));
        }
        let count = result as usize;
        if count == 0 {
            bail!(
                "short prefix-record read at {}/{}",
                offset + done,
                offset + destination.len()
            );
        }
        done += count;
        if direct && done < destination.len() && !done.is_multiple_of(ALIGNMENT) {
            bail!("unaligned short O_DIRECT read at {}", offset + done);
        }
    }
    Ok(())
}

pub(crate) fn read_header_and_tokens(path: &Path) -> Result<(RecordHeader, Vec<u32>)> {
    let mut file = File::open(path)?;
    let file_bytes = usize::try_from(file.metadata()?.len())?;
    let mut raw_header = vec![0u8; HEADER_BYTES];
    file.read_exact(&mut raw_header)?;
    let header = RecordHeader::decode(&raw_header)?;
    if header.file_bytes != file_bytes {
        bail!("prefix-record index file size mismatch");
    }
    if header.token_count > 16 * 1024 * 1024 {
        bail!("prefix-record token count is implausibly large");
    }
    file.seek(SeekFrom::Start(header.tokens_offset as u64))?;
    let token_bytes = header
        .token_count
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("prefix-record token size overflow"))?;
    let mut raw_tokens = vec![0u8; token_bytes];
    file.read_exact(&mut raw_tokens)?;
    if crc32fast::hash(&raw_tokens) != header.tokens_crc32 {
        bail!("prefix-record token CRC mismatch");
    }
    let tokens = raw_tokens
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("chunk of 4")))
        .collect();
    Ok((header, tokens))
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::io::Write;

    use super::*;
    use crate::format::RegionDescription;

    #[test]
    fn read_bands_cover_the_file_once_and_stay_aligned() {
        let total = 19 * ALIGNMENT;
        let bands = read_bands(total, 4, 4 * ALIGNMENT);
        assert_eq!(bands.first().map(|band| band.start), Some(0));
        assert_eq!(bands.last().map(|band| band.end), Some(total));
        for pair in bands.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        for band in bands {
            assert!(band.start.is_multiple_of(ALIGNMENT));
            assert!(band.end.is_multiple_of(ALIGNMENT));
        }
    }

    #[test]
    fn checked_read_validates_all_region_crcs() {
        let path =
            std::env::temp_dir().join(format!("letsinfer-prefix-io-test-{}", std::process::id()));
        let mut header = RecordHeader::layout(
            [9; 32],
            17,
            4,
            vec![RegionDescription {
                name: "state".into(),
                byte_count: 2 * ALIGNMENT as u64,
            }],
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_bytes];
        for (index, token) in [1u32, 2, 3, 4].iter().enumerate() {
            let offset = header.tokens_offset + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&token.to_le_bytes());
        }
        bytes[header.region_range(0).unwrap()].fill(0x5a);
        header.tokens_crc32 = crc32fast::hash(&bytes[header.tokens_range()]);
        header.region_crc32 = vec![crc32fast::hash(&bytes[header.region_range(0).unwrap()])];
        header.encode(&mut bytes[..HEADER_BYTES]).unwrap();
        let mut file = File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let loaded = read_record_checked(&path, &header, false).unwrap();
        assert_eq!(loaded.as_slice(), bytes);
        fs::remove_file(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn large_checked_read_uses_reclaimable_file_mapping() {
        let path = std::env::temp_dir().join(format!(
            "letsinfer-prefix-mapped-read-test-{}",
            std::process::id()
        ));
        let mut header = RecordHeader::layout(
            [7; 32],
            29,
            4,
            vec![RegionDescription {
                name: "state".into(),
                byte_count: FILE_BACKED_RECORD_MIN_BYTES as u64,
            }],
        )
        .unwrap();
        let mut bytes = vec![0u8; header.file_bytes];
        for (index, token) in [5u32, 6, 7, 8].iter().enumerate() {
            let offset = header.tokens_offset + index * 4;
            bytes[offset..offset + 4].copy_from_slice(&token.to_le_bytes());
        }
        let region = header.region_range(0).unwrap();
        bytes[region.start] = 0xA5;
        bytes[region.end - 1] = 0x5A;
        header.tokens_crc32 = crc32fast::hash(&bytes[header.tokens_range()]);
        header.region_crc32 = vec![crc32fast::hash(&bytes[region.clone()])];
        header.encode(&mut bytes[..HEADER_BYTES]).unwrap();
        let mut file = File::create(&path).unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let loaded = read_record_checked(&path, &header, true).unwrap();
        assert!(loaded.is_file_backed());
        assert_eq!(loaded.as_slice()[region.start], 0xA5);
        assert_eq!(loaded.as_slice()[region.end - 1], 0x5A);
        drop(loaded);
        fs::remove_file(path).unwrap();
    }
}
