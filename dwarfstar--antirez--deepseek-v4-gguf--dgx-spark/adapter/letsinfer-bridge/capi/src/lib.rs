// SPDX-License-Identifier: AGPL-3.0-only

//! C ABI for `letsinfer_prefix_store` — the engine-neutral persistent prefix
//! cache extracted from the validated Let's Infer implementation.
//!
//! Contract (mirrors `source/letsinfer/include/letsinfer/PrefixCache.hpp`):
//!
//! - `letsinfer_prefix_store_open` / `letsinfer_prefix_store_close` own the store lifetime;
//! - `letsinfer_prefix_store_begin_capture` is admission-checked (bounded writers, byte
//!   capacity, dedup) and returns NULL with a rejection code on refusal;
//! - `letsinfer_prefix_store_longest_prefix` returns NULL as a safe miss; a non-NULL
//!   reader pins its record against eviction until closed;
//! - region reads are CRC-verified; any integrity failure is reported as an
//!   error AND the store evicts the record (fail closed);
//! - every function catches Rust panics and converts them to error returns:
//!   a cache problem must never crash the inference server.
//!
//! Thread rules: handles are plain heap pointers. The store handle may be
//! shared across threads (the underlying store is internally synchronized).
//! A writer or reader handle must be used by one thread at a time.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;

use letsinfer_prefix_store::{
    CaptureRejection, LoadedRegion, PrefixStore, RecordReader, RecordWriter, RegionDescription,
    StoreConfig,
};

/// ABI version reported by `letsinfer_prefix_abi_version`. Bump on any signature or
/// semantic change; the C loader refuses a mismatched library.
pub const LETSINFER_PREFIX_ABI_VERSION: u32 = 2;

pub struct LetsInferPrefixStore {
    store: Arc<PrefixStore>,
}

pub struct LetsInferPrefixWriter {
    writer: Option<RecordWriter>,
}

pub struct LetsInferPrefixReader {
    store: Arc<PrefixStore>,
    reader: RecordReader,
    pinned_region: Option<LoadedRegion>,
}

// Rejection codes (letsinfer_prefix_store_begin_capture). 0 is "accepted" and is never
// written; keep in sync with ds4_letsinfer_cache.c.
pub const LETSINFER_PREFIX_REJECT_DISABLED: c_int = 1;
pub const LETSINFER_PREFIX_REJECT_WRITER_BUSY: c_int = 2;
pub const LETSINFER_PREFIX_REJECT_ALREADY_STORED: c_int = 3;
pub const LETSINFER_PREFIX_REJECT_BELOW_MIN_TOKENS: c_int = 4;
pub const LETSINFER_PREFIX_REJECT_RECORD_TOO_LARGE: c_int = 5;
pub const LETSINFER_PREFIX_REJECT_INSUFFICIENT_CAPACITY: c_int = 6;
pub const LETSINFER_PREFIX_REJECT_ALLOCATION_FAILURE: c_int = 7;
pub const LETSINFER_PREFIX_REJECT_INVALID_PLAN: c_int = 8;
pub const LETSINFER_PREFIX_REJECT_INVALID_ARGUMENT: c_int = 9;
pub const LETSINFER_PREFIX_REJECT_PANIC: c_int = 10;

fn rejection_code(rejection: CaptureRejection) -> c_int {
    match rejection {
        CaptureRejection::Disabled => LETSINFER_PREFIX_REJECT_DISABLED,
        CaptureRejection::WriterBusy => LETSINFER_PREFIX_REJECT_WRITER_BUSY,
        CaptureRejection::AlreadyStored => LETSINFER_PREFIX_REJECT_ALREADY_STORED,
        CaptureRejection::BelowMinimumTokenCount => LETSINFER_PREFIX_REJECT_BELOW_MIN_TOKENS,
        CaptureRejection::RecordTooLarge => LETSINFER_PREFIX_REJECT_RECORD_TOO_LARGE,
        CaptureRejection::InsufficientCapacity => LETSINFER_PREFIX_REJECT_INSUFFICIENT_CAPACITY,
        CaptureRejection::AllocationFailure => LETSINFER_PREFIX_REJECT_ALLOCATION_FAILURE,
        CaptureRejection::InvalidPlan => LETSINFER_PREFIX_REJECT_INVALID_PLAN,
    }
}

/// Copy a message into a caller-provided, always-NUL-terminated buffer.
fn set_err(err: *mut c_char, err_len: usize, message: &str) {
    if err.is_null() || err_len == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let copy = bytes.len().min(err_len - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), err.cast::<u8>(), copy);
        *err.add(copy) = 0;
    }
}

unsafe fn tokens_slice<'a>(tokens: *const u32, count: usize) -> Option<&'a [u32]> {
    if count == 0 || tokens.is_null() {
        return None;
    }
    Some(std::slice::from_raw_parts(tokens, count))
}

#[no_mangle]
pub extern "C" fn letsinfer_prefix_abi_version() -> u32 {
    LETSINFER_PREFIX_ABI_VERSION
}

/// Open (creating if needed) the store rooted at `root`. Returns NULL on
/// failure with a message in `err`. `capacity_bytes` bounds committed +
/// reserved records (0 = lookups only, capture disabled). `direct_reads`
/// non-zero uses O_DIRECT bulk reads on Linux.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_store_open(
    root: *const c_char,
    capacity_bytes: u64,
    ttl_seconds: u64,
    minimum_token_count: u64,
    resident_capacity_bytes: u64,
    direct_reads: c_int,
    err: *mut c_char,
    err_len: usize,
) -> *mut LetsInferPrefixStore {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if root.is_null() {
            set_err(err, err_len, "root path is NULL");
            return std::ptr::null_mut();
        }
        let root = match unsafe { CStr::from_ptr(root) }.to_str() {
            Ok(text) if !text.is_empty() => PathBuf::from(text),
            _ => {
                set_err(err, err_len, "root path is not valid UTF-8 or is empty");
                return std::ptr::null_mut();
            }
        };
        let config = StoreConfig {
            root,
            capacity_bytes,
            ttl_seconds,
            minimum_token_count: minimum_token_count as usize,
            resident_capacity_bytes,
            direct_reads: direct_reads != 0,
        };
        match PrefixStore::open(config) {
            Ok(store) => Box::into_raw(Box::new(LetsInferPrefixStore { store })),
            Err(error) => {
                set_err(err, err_len, &format!("open failed: {error:#}"));
                std::ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(pointer) => pointer,
        Err(_) => {
            set_err(err, err_len, "panic in letsinfer_prefix_store_open");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn letsinfer_prefix_store_close(store: *mut LetsInferPrefixStore) {
    if store.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let boxed = unsafe { Box::from_raw(store) };
        // Drain the bounded background writer before dropping the last Arc:
        // the writer thread only holds a Weak, so queued-but-uncommitted
        // records die with the store (observed 2026-08-06: a single capture
        // queued at server shutdown was lost, `captures_admitted=1
        // captures_committed=0`). Bounded wait so a wedged writer can never
        // hang server shutdown.
        for _ in 0..1000 {
            if boxed.store.statistics().outstanding_writer_count == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        drop(boxed);
    }));
}

/// Admission-checked capture of one record keyed by (fingerprint, tokens).
/// `region_names` / `region_byte_counts` describe `region_count` opaque
/// regions. On refusal returns NULL and stores a LETSINFER_PREFIX_REJECT_* code in
/// `rejection_out`.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_store_begin_capture(
    store: *mut LetsInferPrefixStore,
    fingerprint: *const u8,
    tokens: *const u32,
    token_count: usize,
    region_names: *const *const c_char,
    region_byte_counts: *const u64,
    region_count: usize,
    rejection_out: *mut c_int,
) -> *mut LetsInferPrefixWriter {
    let set_rejection = |code: c_int| {
        if !rejection_out.is_null() {
            unsafe { *rejection_out = code };
        }
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        if store.is_null()
            || fingerprint.is_null()
            || region_names.is_null()
            || region_byte_counts.is_null()
            || region_count == 0
        {
            set_rejection(LETSINFER_PREFIX_REJECT_INVALID_ARGUMENT);
            return std::ptr::null_mut();
        }
        let Some(tokens) = (unsafe { tokens_slice(tokens, token_count) }) else {
            set_rejection(LETSINFER_PREFIX_REJECT_INVALID_ARGUMENT);
            return std::ptr::null_mut();
        };
        let mut print = [0u8; 32];
        print.copy_from_slice(unsafe { std::slice::from_raw_parts(fingerprint, 32) });
        let mut regions = Vec::with_capacity(region_count);
        for index in 0..region_count {
            let name = unsafe { *region_names.add(index) };
            if name.is_null() {
                set_rejection(LETSINFER_PREFIX_REJECT_INVALID_ARGUMENT);
                return std::ptr::null_mut();
            }
            let Ok(name) = unsafe { CStr::from_ptr(name) }.to_str() else {
                set_rejection(LETSINFER_PREFIX_REJECT_INVALID_ARGUMENT);
                return std::ptr::null_mut();
            };
            regions.push(RegionDescription {
                name: name.to_string(),
                byte_count: unsafe { *region_byte_counts.add(index) },
            });
        }
        let handle = unsafe { &*store };
        match handle.store.begin_capture(print, tokens, regions) {
            Ok(writer) => Box::into_raw(Box::new(LetsInferPrefixWriter {
                writer: Some(writer),
            })),
            Err(rejection) => {
                set_rejection(rejection_code(rejection));
                std::ptr::null_mut()
            }
        }
    }));
    match result {
        Ok(pointer) => pointer,
        Err(_) => {
            set_rejection(LETSINFER_PREFIX_REJECT_PANIC);
            std::ptr::null_mut()
        }
    }
}

/// Mutable pointer to region `index` inside the writer's record buffer, and
/// its byte count. NULL on any invalid input. The pointer stays valid until
/// commit/cancel/free.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_writer_region(
    writer: *mut LetsInferPrefixWriter,
    index: usize,
    byte_count_out: *mut u64,
) -> *mut u8 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if writer.is_null() {
            return std::ptr::null_mut();
        }
        let handle = unsafe { &mut *writer };
        let Some(writer) = handle.writer.as_mut() else {
            return std::ptr::null_mut();
        };
        match writer.writable_region(index) {
            Some(region) => {
                if !byte_count_out.is_null() {
                    unsafe { *byte_count_out = region.len() as u64 };
                }
                region.as_mut_ptr()
            }
            None => std::ptr::null_mut(),
        }
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Queue the captured record to the store's bounded background writer and
/// free the writer handle. Returns 0 on success. `promote_resident` non-zero
/// keeps the committed blob in the byte-bounded host-RAM tier.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_writer_commit_async(
    writer: *mut LetsInferPrefixWriter,
    promote_resident: c_int,
    err: *mut c_char,
    err_len: usize,
) -> c_int {
    if writer.is_null() {
        set_err(err, err_len, "writer is NULL");
        return -1;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut handle = unsafe { Box::from_raw(writer) };
        let Some(writer) = handle.writer.take() else {
            set_err(err, err_len, "writer already consumed");
            return -1;
        };
        match writer.commit_async(promote_resident != 0) {
            Ok(()) => 0,
            Err(error) => {
                set_err(err, err_len, &format!("commit failed: {error:#}"));
                -1
            }
        }
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            set_err(err, err_len, "panic in letsinfer_prefix_writer_commit_async");
            -1
        }
    }
}

/// Cancel a capture, releasing its reservation, and free the writer handle.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_writer_cancel(writer: *mut LetsInferPrefixWriter) {
    if writer.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let mut handle = unsafe { Box::from_raw(writer) };
        if let Some(writer) = handle.writer.take() {
            writer.cancel();
        }
    }));
}

/// Longest stored exact prefix of `tokens` with at least
/// `minimum_token_count` tokens, or NULL as a safe miss. The returned reader
/// pins the record until `letsinfer_prefix_reader_close`.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_store_longest_prefix(
    store: *mut LetsInferPrefixStore,
    fingerprint: *const u8,
    tokens: *const u32,
    token_count: usize,
    minimum_token_count: usize,
) -> *mut LetsInferPrefixReader {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if store.is_null() || fingerprint.is_null() {
            return std::ptr::null_mut();
        }
        let Some(tokens) = (unsafe { tokens_slice(tokens, token_count) }) else {
            return std::ptr::null_mut();
        };
        let mut print = [0u8; 32];
        print.copy_from_slice(unsafe { std::slice::from_raw_parts(fingerprint, 32) });
        let handle = unsafe { &*store };
        match handle
            .store
            .longest_prefix(print, tokens, minimum_token_count)
        {
            Some(reader) => Box::into_raw(Box::new(LetsInferPrefixReader {
                store: handle.store.clone(),
                reader,
                pinned_region: None,
            })),
            None => std::ptr::null_mut(),
        }
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Number of key tokens the record was stored under.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_reader_token_count(reader: *const LetsInferPrefixReader) -> usize {
    if reader.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        unsafe { &*reader }.reader.token_count()
    }))
    .unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn letsinfer_prefix_reader_region_count(reader: *const LetsInferPrefixReader) -> usize {
    if reader.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        unsafe { &*reader }.reader.region_count()
    }))
    .unwrap_or(0)
}

/// Byte count of region `index`; 0 for an invalid index (regions with zero
/// bytes are not used by the DS4 adapter).
#[no_mangle]
pub extern "C" fn letsinfer_prefix_reader_region_byte_count(reader: *const LetsInferPrefixReader, index: usize) -> u64 {
    if reader.is_null() {
        return 0;
    }
    catch_unwind(AssertUnwindSafe(|| {
        unsafe { &*reader }
            .reader
            .region_description(index)
            .map_or(0, |region| region.byte_count)
    }))
    .unwrap_or(0)
}

/// Copy region `index`'s name into `name_out` (NUL-terminated). Returns 0 on
/// success, -1 on invalid index/arguments.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_reader_region_name(
    reader: *const LetsInferPrefixReader,
    index: usize,
    name_out: *mut c_char,
    name_capacity: usize,
) -> c_int {
    if reader.is_null() || name_out.is_null() || name_capacity == 0 {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        match unsafe { &*reader }.reader.region_description(index) {
            Some(region) => {
                set_err(name_out, name_capacity, &region.name);
                0
            }
            None => -1,
        }
    }))
    .unwrap_or(-1)
}

/// Read region `index` into `destination` (must be exactly the region's byte
/// count), verifying CRCs. Returns 0 on success; on failure the store also
/// evicts the record (fail closed) and the caller must treat it as a miss.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_reader_read_region(
    reader: *mut LetsInferPrefixReader,
    index: usize,
    destination: *mut u8,
    destination_len: u64,
    err: *mut c_char,
    err_len: usize,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if reader.is_null() || destination.is_null() {
            set_err(err, err_len, "NULL reader or destination");
            return -1;
        }
        let Ok(len) = usize::try_from(destination_len) else {
            set_err(err, err_len, "destination length exceeds usize");
            return -1;
        };
        let destination = unsafe { std::slice::from_raw_parts_mut(destination, len) };
        match unsafe { &*reader }.reader.read_region(index, destination) {
            Ok(()) => 0,
            Err(error) => {
                set_err(err, err_len, &format!("read failed: {error:#}"));
                -1
            }
        }
    }));
    match result {
        Ok(code) => code,
        Err(_) => {
            set_err(err, err_len, "panic in letsinfer_prefix_reader_read_region");
            -1
        }
    }
}

/// Return a zero-copy immutable view of a CRC-verified region. The pointer is
/// valid until the next call for this reader or until the reader is closed.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_reader_region_data(
    reader: *mut LetsInferPrefixReader,
    index: usize,
    byte_count_out: *mut u64,
    err: *mut c_char,
    err_len: usize,
) -> *const u8 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if !byte_count_out.is_null() {
            unsafe { *byte_count_out = 0 };
        }
        if reader.is_null() {
            set_err(err, err_len, "reader is NULL");
            return std::ptr::null();
        }
        let handle = unsafe { &mut *reader };
        match handle.reader.load_region(index) {
            Ok(region) => {
                handle.pinned_region = Some(region);
                let bytes = handle.pinned_region.as_ref().unwrap().as_slice();
                if !byte_count_out.is_null() {
                    unsafe { *byte_count_out = bytes.len() as u64 };
                }
                bytes.as_ptr()
            }
            Err(error) => {
                set_err(err, err_len, &format!("region view failed: {error:#}"));
                std::ptr::null()
            }
        }
    }));
    match result {
        Ok(pointer) => pointer,
        Err(_) => {
            set_err(err, err_len, "panic in letsinfer_prefix_reader_region_data");
            std::ptr::null()
        }
    }
}

/// Constant-time last-used touch of the reader's record (one per request).
#[no_mangle]
pub extern "C" fn letsinfer_prefix_reader_touch(reader: *const LetsInferPrefixReader) {
    if reader.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let handle = unsafe { &*reader };
        handle.store.touch(&handle.reader);
    }));
}

/// Drop the record's host-RAM residency (the caller's engine tier now owns
/// the state); the reader itself stays valid until closed.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_reader_release_resident(reader: *const LetsInferPrefixReader) {
    if reader.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let handle = unsafe { &*reader };
        handle.store.release_resident(&handle.reader);
    }));
}

/// Close a reader, unpinning its record.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_reader_close(reader: *mut LetsInferPrefixReader) {
    if reader.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(reader) });
    }));
}

/// Remove records idle past the configured TTL. Returns 0 on success.
#[no_mangle]
pub extern "C" fn letsinfer_prefix_store_reclaim_expired(
    store: *mut LetsInferPrefixStore,
    unix_time_seconds: u64,
) -> c_int {
    if store.is_null() {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        match unsafe { &*store }.store.reclaim_expired(unix_time_seconds) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    }))
    .unwrap_or(-1)
}

/// One-line human-readable statistics summary for logs. Returns 0 on
/// success and fills `line_out` (NUL-terminated, truncated to capacity).
#[no_mangle]
pub extern "C" fn letsinfer_prefix_store_stats_line(
    store: *const LetsInferPrefixStore,
    line_out: *mut c_char,
    line_capacity: usize,
) -> c_int {
    if store.is_null() || line_out.is_null() || line_capacity == 0 {
        return -1;
    }
    catch_unwind(AssertUnwindSafe(|| {
        let stats = unsafe { &*store }.store.statistics();
        let line = format!(
            "records={} committed_bytes={} reserved_bytes={} lookups={} hits={} \
             captures_admitted={} captures_committed={} captures_failed={} writer_busy={} \
             evictions_capacity={} evictions_expiry={} evictions_integrity={} \
             resident_bytes={}/{} nvme_loads={}",
            stats.record_count,
            stats.committed_byte_count,
            stats.reserved_byte_count,
            stats.lookup_count,
            stats.hit_count,
            stats.admitted_capture_count,
            stats.committed_capture_count,
            stats.failed_capture_count,
            stats.writer_busy_rejection_count,
            stats.capacity_eviction_count,
            stats.expiry_eviction_count,
            stats.integrity_eviction_count,
            stats.resident_used_bytes,
            stats.resident_capacity_bytes,
            stats.nvme_load_count,
        );
        set_err(line_out, line_capacity, &line);
        0
    }))
    .unwrap_or(-1)
}
