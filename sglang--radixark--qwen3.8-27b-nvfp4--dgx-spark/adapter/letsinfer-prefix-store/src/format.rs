// SPDX-License-Identifier: AGPL-3.0-only

//! Versioned, page-aligned record format for opaque named state regions.
//!
//! Instead of hard-coding tokens, KV, SSM, or hidden-state sections, a record
//! contains exact token IDs plus N opaque named regions, each page-aligned and
//! individually CRC'd:
//!
//! - offsets are encoded on disk but must equal the canonical layout
//!   recomputed by [`RecordHeader::validate_layout`] — hostile or corrupt
//!   headers cannot point regions outside the file;
//! - the header CRC is computed with its own field zeroed, so decode
//!   verifies the entire fixed-size header in one pass;
//! - all arithmetic is checked; oversized geometry is an error, never a
//!   wraparound.
//!
//! ```text
//! 0                          4 KiB
//! ├──────── header ───────────┤
//! ├─ token IDs (u32 LE) ─┤ pad ├─ region 0 ─┤ pad ├─ region 1 ─┤ ... pad
//! ```

use anyhow::{bail, Result};

pub const ALIGNMENT: usize = 4096;
pub const HEADER_BYTES: usize = ALIGNMENT;
const MAGIC: &[u8; 8] = b"LIPFX001";
const VERSION: u32 = 1;
/// Bounded per-record region table: fits fixed-size header encoding and is
/// above the qualified adapters' per-layer state-region count (currently
/// fewer than 64 regions). 96 × (12+8+8+4) bytes = 3,072, which fits
/// the 4 KiB header alongside the fixed fields.
pub const MAX_REGIONS: usize = 96;
const MAX_REGION_NAME_BYTES: usize = 12;

/// One opaque state region: the store never interprets its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionDescription {
    /// Adapter-chosen identifier (e.g. "paged_kv", "gdn_ssm", "scale_basis").
    /// ASCII, non-empty, at most [`MAX_REGION_NAME_BYTES`] bytes.
    pub name: String,
    pub byte_count: u64,
}

/// Decoded record header. Numeric fields are fixed little-endian on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordHeader {
    /// Full compatibility fingerprint digest (format version, state ABI,
    /// model identity/revision, tokenizer identity, layout, dtype, adapter
    /// identity, salt). The store treats it as opaque; equality is the only
    /// operation.
    pub fingerprint: [u8; 32],
    /// Chained hash over fingerprint + exact token IDs; index discriminator
    /// only — exact token comparison stays authoritative.
    pub key_hash: u64,
    pub token_count: usize,
    pub regions: Vec<RegionDescription>,
    pub tokens_offset: usize,
    pub region_offsets: Vec<usize>,
    pub logical_bytes: usize,
    pub file_bytes: usize,
    /// CRC32 of the token section.
    pub tokens_crc32: u32,
    /// CRC32 per region, same order as `regions`.
    pub region_crc32: Vec<u32>,
}

impl RecordHeader {
    /// Derive every offset from semantic sizes; callers never choose offsets.
    pub fn layout(
        fingerprint: [u8; 32],
        key_hash: u64,
        token_count: usize,
        regions: Vec<RegionDescription>,
    ) -> Result<Self> {
        if token_count == 0 {
            bail!("prefix record needs a non-zero token count");
        }
        if regions.is_empty() || regions.len() > MAX_REGIONS {
            bail!("invalid prefix-record region count {}", regions.len());
        }
        for region in &regions {
            validate_region_name(&region.name)?;
            if region.byte_count == 0 {
                bail!("prefix-record region '{}' is empty", region.name);
            }
        }
        let tokens_offset = HEADER_BYTES;
        let tokens_end = checked_add(tokens_offset, checked_mul(token_count, 4)?)?;
        let mut region_offsets = Vec::with_capacity(regions.len());
        let mut cursor = align_up(tokens_end)?;
        for region in &regions {
            region_offsets.push(cursor);
            let byte_count = as_usize(region.byte_count)?;
            cursor = align_up(checked_add(cursor, byte_count)?)?;
        }
        // After the final align_up, cursor is both the logical end padded to
        // a page and therefore the file size.
        let logical_bytes = {
            let last_offset = *region_offsets.last().expect("non-empty regions");
            let last_bytes = as_usize(regions.last().expect("non-empty").byte_count)?;
            checked_add(last_offset, last_bytes)?
        };
        let file_bytes = align_up(logical_bytes)?;
        Ok(Self {
            fingerprint,
            key_hash,
            token_count,
            regions,
            tokens_offset,
            region_offsets,
            logical_bytes,
            file_bytes,
            tokens_crc32: 0,
            region_crc32: Vec::new(),
        })
    }

    pub fn encode(&self, destination: &mut [u8]) -> Result<()> {
        if destination.len() < HEADER_BYTES {
            bail!("prefix-record header destination is too small");
        }
        if self.region_crc32.len() != self.regions.len() {
            bail!("prefix-record region CRC count mismatch");
        }
        destination[..HEADER_BYTES].fill(0);
        let mut cursor = 0usize;
        put_bytes(destination, &mut cursor, MAGIC)?;
        put_u32(destination, &mut cursor, VERSION)?;
        put_u32(destination, &mut cursor, HEADER_BYTES as u32)?;
        put_bytes(destination, &mut cursor, &self.fingerprint)?;
        put_u64(destination, &mut cursor, self.key_hash)?;
        put_u64(destination, &mut cursor, self.token_count as u64)?;
        put_u32(destination, &mut cursor, self.regions.len() as u32)?;
        put_u32(destination, &mut cursor, self.tokens_crc32)?;
        for value in [
            self.tokens_offset as u64,
            self.logical_bytes as u64,
            self.file_bytes as u64,
        ] {
            put_u64(destination, &mut cursor, value)?;
        }
        // Header CRC field is zero during checksum computation.
        let header_crc_offset = cursor;
        put_u32(destination, &mut cursor, 0)?;
        for index in 0..MAX_REGIONS {
            let mut name = [0u8; MAX_REGION_NAME_BYTES];
            let mut byte_count = 0u64;
            let mut offset = 0u64;
            let mut crc = 0u32;
            if index < self.regions.len() {
                let region = &self.regions[index];
                name[..region.name.len()].copy_from_slice(region.name.as_bytes());
                byte_count = region.byte_count;
                offset = self.region_offsets[index] as u64;
                crc = self.region_crc32[index];
            }
            put_bytes(destination, &mut cursor, &name)?;
            put_u64(destination, &mut cursor, byte_count)?;
            put_u64(destination, &mut cursor, offset)?;
            put_u32(destination, &mut cursor, crc)?;
        }
        let checksum = header_checksum(destination, header_crc_offset);
        destination[header_crc_offset..header_crc_offset + 4]
            .copy_from_slice(&checksum.to_le_bytes());
        Ok(())
    }

    pub fn decode(source: &[u8]) -> Result<Self> {
        if source.len() < HEADER_BYTES {
            bail!("short prefix-record header");
        }
        let mut cursor = 0usize;
        if take(source, &mut cursor, MAGIC.len())? != MAGIC {
            bail!("bad prefix-record magic");
        }
        if get_u32(source, &mut cursor)? != VERSION {
            bail!("unsupported prefix-record format version");
        }
        if get_u32(source, &mut cursor)? as usize != HEADER_BYTES {
            bail!("bad prefix-record header size");
        }
        let mut fingerprint = [0u8; 32];
        fingerprint.copy_from_slice(take(source, &mut cursor, 32)?);
        let key_hash = get_u64(source, &mut cursor)?;
        let token_count = as_usize(get_u64(source, &mut cursor)?)?;
        let region_count = get_u32(source, &mut cursor)? as usize;
        let tokens_crc32 = get_u32(source, &mut cursor)?;
        let tokens_offset = as_usize(get_u64(source, &mut cursor)?)?;
        let logical_bytes = as_usize(get_u64(source, &mut cursor)?)?;
        let file_bytes = as_usize(get_u64(source, &mut cursor)?)?;
        let header_crc_offset = cursor;
        let expected_header_crc = get_u32(source, &mut cursor)?;
        if expected_header_crc != header_checksum(source, header_crc_offset) {
            bail!("prefix-record header CRC mismatch");
        }
        if region_count == 0 || region_count > MAX_REGIONS {
            bail!("invalid prefix-record region count {region_count}");
        }
        let mut regions = Vec::with_capacity(region_count);
        let mut region_offsets = Vec::with_capacity(region_count);
        let mut region_crc32 = Vec::with_capacity(region_count);
        for index in 0..MAX_REGIONS {
            let name_bytes = take(source, &mut cursor, MAX_REGION_NAME_BYTES)?;
            let byte_count = get_u64(source, &mut cursor)?;
            let offset = as_usize(get_u64(source, &mut cursor)?)?;
            let crc = get_u32(source, &mut cursor)?;
            if index < region_count {
                let end = name_bytes
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(MAX_REGION_NAME_BYTES);
                let name = std::str::from_utf8(&name_bytes[..end])
                    .map_err(|_| anyhow::anyhow!("bad prefix-record region name"))?
                    .to_string();
                regions.push(RegionDescription { name, byte_count });
                region_offsets.push(offset);
                region_crc32.push(crc);
            }
        }
        // Construct only after bounded parsing; canonical layout validation
        // is the final trust boundary before any payload access.
        let header = Self {
            fingerprint,
            key_hash,
            token_count,
            regions,
            tokens_offset,
            region_offsets,
            logical_bytes,
            file_bytes,
            tokens_crc32,
            region_crc32,
        };
        header.validate_layout()?;
        Ok(header)
    }

    pub fn validate_layout(&self) -> Result<()> {
        if self.token_count == 0
            || self.tokens_offset != HEADER_BYTES
            || self.file_bytes < self.logical_bytes
            || !self.file_bytes.is_multiple_of(ALIGNMENT)
            || self.region_offsets.len() != self.regions.len()
            || self.region_crc32.len() != self.regions.len()
        {
            bail!("invalid prefix-record geometry");
        }
        let expected = Self::layout(
            self.fingerprint,
            self.key_hash,
            self.token_count,
            self.regions.clone(),
        )?;
        if self.region_offsets != expected.region_offsets
            || self.logical_bytes != expected.logical_bytes
            || self.file_bytes != expected.file_bytes
        {
            bail!("non-canonical prefix-record layout");
        }
        Ok(())
    }

    /// State-ABI compatibility: same fingerprint and identical region
    /// geometry. Per-record identity (tokens, key hash) is checked
    /// separately by the exact-token comparison.
    pub fn compatible_with(&self, expected: &Self) -> bool {
        self.fingerprint == expected.fingerprint && self.regions == expected.regions
    }

    pub fn tokens_range(&self) -> std::ops::Range<usize> {
        self.tokens_offset..self.tokens_offset + self.token_count * 4
    }

    pub fn region_range(&self, index: usize) -> Option<std::ops::Range<usize>> {
        let offset = *self.region_offsets.get(index)?;
        let bytes = usize::try_from(self.regions.get(index)?.byte_count).ok()?;
        Some(offset..offset + bytes)
    }
}

fn validate_region_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > MAX_REGION_NAME_BYTES
        || !name.bytes().all(|b| b.is_ascii_graphic())
    {
        bail!("invalid prefix-record region name {name:?}");
    }
    Ok(())
}

fn header_checksum(source: &[u8], checksum_offset: usize) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&source[..checksum_offset]);
    hasher.update(&[0; 4]);
    hasher.update(&source[checksum_offset + 4..HEADER_BYTES]);
    hasher.finalize()
}

pub(crate) fn align_up(value: usize) -> Result<usize> {
    let with_slack = checked_add(value, ALIGNMENT - 1)?;
    Ok(with_slack / ALIGNMENT * ALIGNMENT)
}

pub(crate) fn checked_add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| anyhow::anyhow!("prefix-record size overflow"))
}

pub(crate) fn checked_mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| anyhow::anyhow!("prefix-record size overflow"))
}

fn as_usize(value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| anyhow::anyhow!("prefix-record value exceeds usize"))
}

fn put_bytes(destination: &mut [u8], cursor: &mut usize, bytes: &[u8]) -> Result<()> {
    let target = destination
        .get_mut(*cursor..*cursor + bytes.len())
        .ok_or_else(|| anyhow::anyhow!("prefix-record header overflow"))?;
    target.copy_from_slice(bytes);
    *cursor += bytes.len();
    Ok(())
}

fn put_u32(destination: &mut [u8], cursor: &mut usize, value: u32) -> Result<()> {
    put_bytes(destination, cursor, &value.to_le_bytes())
}

fn put_u64(destination: &mut [u8], cursor: &mut usize, value: u64) -> Result<()> {
    put_bytes(destination, cursor, &value.to_le_bytes())
}

fn take<'a>(source: &'a [u8], cursor: &mut usize, count: usize) -> Result<&'a [u8]> {
    let bytes = source
        .get(*cursor..*cursor + count)
        .ok_or_else(|| anyhow::anyhow!("short prefix-record header"))?;
    *cursor += count;
    Ok(bytes)
}

fn get_u32(source: &[u8], cursor: &mut usize) -> Result<u32> {
    Ok(u32::from_le_bytes(take(source, cursor, 4)?.try_into()?))
}

fn get_u64(source: &[u8], cursor: &mut usize) -> Result<u64> {
    Ok(u64::from_le_bytes(take(source, cursor, 8)?.try_into()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> RecordHeader {
        let mut header = RecordHeader::layout(
            [7u8; 32],
            0xDEAD_BEEF_1234_5678,
            1000,
            vec![
                RegionDescription {
                    name: "paged_kv".into(),
                    byte_count: 1 << 20,
                },
                RegionDescription {
                    name: "gdn_ssm".into(),
                    byte_count: 64 << 10,
                },
                RegionDescription {
                    name: "scale_basis".into(),
                    byte_count: 128,
                },
            ],
        )
        .unwrap();
        header.tokens_crc32 = 11;
        header.region_crc32 = vec![22, 33, 44];
        header
    }

    #[test]
    fn round_trips_through_encode_and_decode() {
        let header = sample_header();
        let mut bytes = vec![0u8; HEADER_BYTES];
        header.encode(&mut bytes).unwrap();
        let decoded = RecordHeader::decode(&bytes).unwrap();
        assert_eq!(decoded.fingerprint, header.fingerprint);
        assert_eq!(decoded.key_hash, header.key_hash);
        assert_eq!(decoded.token_count, header.token_count);
        assert_eq!(decoded.regions, header.regions);
        assert_eq!(decoded.region_offsets, header.region_offsets);
        assert_eq!(decoded.region_crc32, header.region_crc32);
        assert_eq!(decoded.file_bytes, header.file_bytes);
    }

    #[test]
    fn all_offsets_are_page_aligned() {
        let header = sample_header();
        assert_eq!(header.tokens_offset % ALIGNMENT, 0);
        for offset in &header.region_offsets {
            assert_eq!(offset % ALIGNMENT, 0);
        }
        assert_eq!(header.file_bytes % ALIGNMENT, 0);
    }

    #[test]
    fn corrupted_header_fails_crc() {
        let header = sample_header();
        let mut bytes = vec![0u8; HEADER_BYTES];
        header.encode(&mut bytes).unwrap();
        bytes[64] ^= 0x01;
        assert!(RecordHeader::decode(&bytes).is_err());
    }

    #[test]
    fn tampered_offset_fails_canonical_validation() {
        let mut header = sample_header();
        // Point a region past its canonical position, then re-encode with a
        // fresh CRC: decode must still reject via canonical re-derivation.
        header.region_offsets[2] += ALIGNMENT;
        header.logical_bytes += ALIGNMENT;
        header.file_bytes += ALIGNMENT;
        let mut bytes = vec![0u8; HEADER_BYTES];
        header.encode(&mut bytes).unwrap();
        assert!(RecordHeader::decode(&bytes).is_err());
    }

    #[test]
    fn rejects_empty_and_oversized_region_tables() {
        assert!(RecordHeader::layout([0; 32], 1, 10, vec![]).is_err());
        let too_many = (0..MAX_REGIONS + 1)
            .map(|i| RegionDescription {
                name: format!("r{i}"),
                byte_count: 1,
            })
            .collect();
        assert!(RecordHeader::layout([0; 32], 1, 10, too_many).is_err());
    }
}
