// SPDX-License-Identifier: AGPL-3.0-only

//! Engine-neutral persistent prefix store.
//!
//! Mirrors the `PrefixStore` contract in
//! `source/letsinfer/include/letsinfer/PrefixCache.hpp`: capture admission with
//! a bounded writer count, longest-exact-prefix lookup where a null result
//! is a safe miss, constant-time touch, TTL reclamation, and statistics.
//!
//! Correctness rules enforced by the validated implementation:
//!
//! - exact token comparison is authoritative; hashes only index;
//! - corrupt, incomplete, or incompatible records are ordinary misses and
//!   are evicted, never returned;
//! - commits are atomic: same-directory temp file, `fsync(file)`, rename,
//!   `fsync(dir)`, then a best-effort `POSIX_FADV_DONTNEED`;
//! - capacity is exact-byte accounted including in-flight reservations,
//!   evicted in deterministic last-used order;
//! - records with active readers or writers are never evicted;
//! - at most one active and one queued multi-GiB writer (admission happens
//!   before any buffer allocation).
//!
//! The data path uses the validated durable mechanics: aligned
//! queue-depth `O_DIRECT` reads, per-path single-flight RAM residency, and one
//! background writer with one queued record. Engine-native block leasing lives
//! in the vLLM adapter because only the engine owns its paged allocator.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use parking_lot::Mutex;

use crate::format::{RecordHeader, RegionDescription, ALIGNMENT, HEADER_BYTES};
#[cfg(target_os = "linux")]
use crate::io::MappedCapture;
use crate::io::{
    read_header_and_tokens, read_record_checked, AlignedBuffer, RecordBuffer,
    FILE_BACKED_RECORD_MIN_BYTES,
};
use crate::ram_tier::{RecordBlob, ResidentTier};

const RECORD_EXTENSION: &str = "letsinfer_prefix";
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
const MAX_OUTSTANDING_COMMITS: usize = 2;
const COMMIT_QUEUE_CAPACITY: usize = MAX_OUTSTANDING_COMMITS - 1;

// FNV-1a; index discriminator only (exact token comparison is authoritative,
// so a collision can cost performance but never correctness).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = seed;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Chained key hash over the fingerprint and every token ID, matching the
/// "parent hash then exact token IDs" chain of the design record.
pub fn key_hash(fingerprint: &[u8; 32], tokens: &[u32]) -> u64 {
    let mut hash = fnv1a(FNV_OFFSET, fingerprint);
    for token in tokens {
        hash = fnv1a(hash, &token.to_le_bytes());
    }
    hash
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureRejection {
    Disabled,
    WriterBusy,
    AlreadyStored,
    BelowMinimumTokenCount,
    RecordTooLarge,
    InsufficientCapacity,
    AllocationFailure,
    InvalidPlan,
}

#[derive(Clone, Debug, Default)]
pub struct StoreStatistics {
    pub committed_byte_count: u64,
    pub reserved_byte_count: u64,
    pub record_count: u64,
    pub active_reader_count: u64,
    pub outstanding_writer_count: u64,
    pub admitted_capture_count: u64,
    pub committed_capture_count: u64,
    pub failed_capture_count: u64,
    pub writer_busy_rejection_count: u64,
    pub lookup_count: u64,
    pub hit_count: u64,
    pub capacity_eviction_count: u64,
    pub expiry_eviction_count: u64,
    pub integrity_eviction_count: u64,
    pub resident_capacity_bytes: u64,
    pub resident_used_bytes: u64,
    pub resident_record_count: u64,
    pub resident_hit_count: u64,
    pub nvme_load_count: u64,
}

#[derive(Clone, Debug)]
pub struct StoreConfig {
    pub root: PathBuf,
    /// Hard byte budget for committed + reserved records. Zero disables
    /// capture (lookups still work against existing records).
    pub capacity_bytes: u64,
    /// Sliding expiry applied by `reclaim_expired`.
    pub ttl_seconds: u64,
    /// Captures below this many tokens are rejected (restore would not
    /// beat recompute).
    pub minimum_token_count: usize,
    /// Byte-bounded complete validated records kept in host RAM. Zero disables
    /// residency while retaining direct durable reads.
    pub resident_capacity_bytes: u64,
    /// Use Linux `O_DIRECT` for bulk reads. Tests and non-Linux builds may
    /// disable it while retaining identical validation.
    pub direct_reads: bool,
}

struct RecordMeta {
    header: RecordHeader,
    tokens: Arc<Vec<u32>>,
    path: PathBuf,
    file_bytes: u64,
    last_used_unix: u64,
    active_readers: u64,
}

struct IndexState {
    /// key_hash → record. One record per exact (fingerprint, tokens).
    records: HashMap<u64, RecordMeta>,
    reserved_hashes: HashSet<u64>,
    committed_bytes: u64,
    reserved_bytes: u64,
    statistics: StoreStatistics,
}

pub struct PrefixStore {
    config: StoreConfig,
    state: Mutex<IndexState>,
    resident: ResidentTier,
    writer: CommitWriter,
}

struct WriterSlots {
    outstanding: AtomicUsize,
    limit: usize,
}

impl WriterSlots {
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            outstanding: AtomicUsize::new(0),
            limit,
        })
    }

    fn try_acquire(self: &Arc<Self>) -> Option<WriterPermit> {
        self.outstanding
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                (current < self.limit).then_some(current + 1)
            })
            .ok()
            .map(|_| WriterPermit {
                slots: self.clone(),
            })
    }

    fn outstanding(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }
}

struct WriterPermit {
    slots: Arc<WriterSlots>,
}

impl Drop for WriterPermit {
    fn drop(&mut self) {
        let previous = self.slots.outstanding.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "writer permit underflow");
    }
}

struct CommitWriter {
    sender: SyncSender<PreparedCommit>,
    slots: Arc<WriterSlots>,
}

impl CommitWriter {
    fn new() -> (Self, Receiver<PreparedCommit>) {
        let (sender, receiver) = mpsc::sync_channel(COMMIT_QUEUE_CAPACITY);
        (
            Self {
                sender,
                slots: WriterSlots::new(MAX_OUTSTANDING_COMMITS),
            },
            receiver,
        )
    }

    fn try_reserve(&self) -> Option<WriterPermit> {
        self.slots.try_acquire()
    }

    fn outstanding(&self) -> usize {
        self.slots.outstanding()
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl PrefixStore {
    /// Open the store, scan and validate existing records, remove stale
    /// temporaries, and enforce capacity by last-used order.
    pub fn open(config: StoreConfig) -> Result<Arc<Self>> {
        fs::create_dir_all(&config.root)
            .with_context(|| format!("create prefix-store root {}", config.root.display()))?;
        let mut state = IndexState {
            records: HashMap::new(),
            reserved_hashes: HashSet::new(),
            committed_bytes: 0,
            reserved_bytes: 0,
            statistics: StoreStatistics::default(),
        };
        for entry in fs::read_dir(&config.root)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.contains(".tmp-") {
                let _ = fs::remove_file(&path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some(RECORD_EXTENSION) {
                continue;
            }
            match load_record_meta(&path) {
                Ok(meta) => {
                    state.committed_bytes += meta.file_bytes;
                    state.records.insert(meta.header.key_hash, meta);
                }
                Err(error) => {
                    // Fail closed: an unreadable record is an ordinary miss;
                    // remove it so it cannot occupy capacity forever.
                    state.statistics.integrity_eviction_count += 1;
                    let _ = fs::remove_file(&path);
                    let _ = error;
                }
            }
        }
        let resident_capacity = usize::try_from(config.resident_capacity_bytes)
            .context("resident capacity exceeds usize")?;
        let (writer, receiver) = CommitWriter::new();
        let store = Arc::new(Self {
            config,
            state: Mutex::new(state),
            resident: ResidentTier::new(resident_capacity),
            writer,
        });
        store.enforce_capacity_for(0)?;
        spawn_writer(&store, receiver);
        Ok(store)
    }

    /// Admission-checked capture. Large records stage directly in an atomic
    /// same-filesystem temporary file so they do not consume a second copy of
    /// scarce unified memory; small records retain the lower-latency aligned
    /// memory path.
    pub fn begin_capture(
        self: &Arc<Self>,
        fingerprint: [u8; 32],
        tokens: &[u32],
        regions: Vec<RegionDescription>,
    ) -> std::result::Result<RecordWriter, CaptureRejection> {
        if self.config.capacity_bytes == 0 {
            return Err(CaptureRejection::Disabled);
        }
        if tokens.len() < self.config.minimum_token_count {
            return Err(CaptureRejection::BelowMinimumTokenCount);
        }
        let hash = key_hash(&fingerprint, tokens);
        let header = RecordHeader::layout(fingerprint, hash, tokens.len(), regions)
            .map_err(|_| CaptureRejection::InvalidPlan)?;
        let file_bytes = header.file_bytes as u64;
        if file_bytes > self.config.capacity_bytes {
            return Err(CaptureRejection::RecordTooLarge);
        }
        // Admission before any allocation or device readback.
        let Some(permit) = self.writer.try_reserve() else {
            self.state.lock().statistics.writer_busy_rejection_count += 1;
            return Err(CaptureRejection::WriterBusy);
        };
        {
            let mut state = self.state.lock();
            if state.reserved_hashes.contains(&hash)
                || state
                    .records
                    .get(&hash)
                    .is_some_and(|existing| existing.tokens.as_slice() == tokens)
            {
                return Err(CaptureRejection::AlreadyStored);
            }
            state.reserved_bytes += file_bytes;
            state.reserved_hashes.insert(hash);
            state.statistics.admitted_capture_count += 1;
        }
        // The reservation above already counts the incoming bytes.
        if self.enforce_capacity_for(0).is_err() {
            let mut state = self.state.lock();
            state.reserved_bytes -= file_bytes;
            state.reserved_hashes.remove(&hash);
            state.statistics.failed_capture_count += 1;
            return Err(CaptureRejection::InsufficientCapacity);
        }
        let mut buffer = match CaptureBuffer::new(self.capture_temp_path(hash), header.file_bytes) {
            Ok(buffer) => buffer,
            Err(_) => {
                let mut state = self.state.lock();
                state.reserved_bytes -= file_bytes;
                state.reserved_hashes.remove(&hash);
                state.statistics.failed_capture_count += 1;
                return Err(CaptureRejection::AllocationFailure);
            }
        };
        for (index, token) in tokens.iter().enumerate() {
            let offset = header.tokens_offset + index * 4;
            buffer.as_mut_slice()[offset..offset + 4].copy_from_slice(&token.to_le_bytes());
        }
        Ok(RecordWriter {
            store: self.clone(),
            header: Some(header),
            tokens: tokens.to_vec(),
            buffer: Some(buffer),
            permit: Some(permit),
        })
    }

    /// Longest exact prefix of `tokens` at a captured boundary, or a safe
    /// miss. The reader pins the record against eviction until dropped.
    pub fn longest_prefix(
        self: &Arc<Self>,
        fingerprint: [u8; 32],
        tokens: &[u32],
        minimum_token_count: usize,
    ) -> Option<RecordReader> {
        // Rolling chained hashes for every prefix length, computed once.
        let mut state = self.state.lock();
        state.statistics.lookup_count += 1;
        // Collect candidate lengths (all stored records with this
        // fingerprint whose token_count fits), longest first.
        let mut candidates: Vec<(usize, u64)> = state
            .records
            .values()
            .filter(|meta| {
                meta.header.fingerprint == fingerprint
                    && meta.header.token_count <= tokens.len()
                    && meta.header.token_count >= minimum_token_count
            })
            .map(|meta| (meta.header.token_count, meta.header.key_hash))
            .collect();
        candidates.sort_unstable_by_key(|candidate| std::cmp::Reverse(candidate.0));
        for (count, stored_hash) in candidates {
            if key_hash(&fingerprint, &tokens[..count]) != stored_hash {
                continue;
            }
            let Some(meta) = state.records.get_mut(&stored_hash) else {
                continue;
            };
            // Exact token comparison is authoritative.
            if meta.tokens.as_slice() != &tokens[..count] {
                continue;
            }
            meta.active_readers += 1;
            meta.last_used_unix = now_unix();
            let header = meta.header.clone();
            let path = meta.path.clone();
            let record_tokens = meta.tokens.clone();
            state.statistics.hit_count += 1;
            state.statistics.active_reader_count += 1;
            drop(state);
            return Some(RecordReader {
                store: self.clone(),
                header,
                path,
                tokens: record_tokens,
                blob: Mutex::new(None),
            });
        }
        None
    }

    /// Compatible startup candidates in exact priority order:
    /// most recently used, final-hidden capsule first, then longer prefix.
    /// Selection is greedy and never exceeds the native byte budget.
    pub fn prewarm_candidates(
        &self,
        fingerprint: [u8; 32],
        capacity_bytes: u64,
    ) -> Vec<(Vec<u32>, bool, u64)> {
        if capacity_bytes == 0 {
            return Vec::new();
        }
        let state = self.state.lock();
        let mut records = state
            .records
            .values()
            .filter(|meta| meta.header.fingerprint == fingerprint)
            .map(|meta| {
                (
                    meta.last_used_unix,
                    meta.header
                        .regions
                        .iter()
                        .any(|region| region.name == "hidden"),
                    meta.header.token_count,
                    meta.file_bytes,
                    meta.tokens.as_ref().clone(),
                )
            })
            .collect::<Vec<_>>();
        records.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| right.1.cmp(&left.1))
                .then_with(|| right.2.cmp(&left.2))
        });
        let mut remaining = capacity_bytes;
        let mut selected = Vec::new();
        for (_, has_hidden, _, file_bytes, tokens) in records {
            if file_bytes > remaining {
                continue;
            }
            remaining -= file_bytes;
            selected.push((tokens, has_hidden, file_bytes));
        }
        selected
    }

    /// Constant-time last-used update (one marker touch per request).
    pub fn touch(&self, reader: &RecordReader) {
        let mut state = self.state.lock();
        if let Some(meta) = state.records.get_mut(&reader.header.key_hash) {
            meta.last_used_unix = now_unix();
        }
    }

    /// Drop future host-RAM reachability after the engine-native tier owns the
    /// state. An active reader keeps its immutable `Arc` until it closes.
    pub fn release_resident(&self, reader: &RecordReader) {
        self.resident.invalidate(&reader.path);
    }

    /// Remove records idle past the TTL. Active records are protected.
    pub fn reclaim_expired(&self, unix_time_seconds: u64) -> Result<()> {
        let ttl = self.config.ttl_seconds;
        let mut victims = Vec::new();
        {
            let mut state = self.state.lock();
            let expired: Vec<u64> = state
                .records
                .values()
                .filter(|meta| {
                    meta.active_readers == 0
                        && meta.last_used_unix.saturating_add(ttl) < unix_time_seconds
                })
                .map(|meta| meta.header.key_hash)
                .collect();
            for hash in expired {
                if let Some(meta) = state.records.remove(&hash) {
                    state.committed_bytes -= meta.file_bytes;
                    state.statistics.expiry_eviction_count += 1;
                    victims.push(meta.path);
                }
            }
        }
        for path in victims {
            self.resident.invalidate(&path);
            let _ = fs::remove_file(path);
        }
        Ok(())
    }

    pub fn statistics(&self) -> StoreStatistics {
        let state = self.state.lock();
        let mut statistics = state.statistics.clone();
        statistics.committed_byte_count = state.committed_bytes;
        statistics.reserved_byte_count = state.reserved_bytes;
        statistics.record_count = state.records.len() as u64;
        statistics.outstanding_writer_count = self.writer.outstanding() as u64;
        let resident = self.resident.stats();
        statistics.resident_capacity_bytes = resident.capacity_bytes as u64;
        statistics.resident_used_bytes = resident.used_bytes as u64;
        statistics.resident_record_count = resident.entries as u64;
        statistics.resident_hit_count = resident.hit_count;
        statistics.nvme_load_count = resident.nvme_load_count;
        statistics
    }

    /// Deterministic byte-LRU: evict oldest inactive records until
    /// committed + reserved + incoming fits the budget.
    fn enforce_capacity_for(&self, incoming_bytes: u64) -> Result<()> {
        let mut victims = Vec::new();
        let mut exhausted = false;
        {
            let mut state = self.state.lock();
            loop {
                let used = state.committed_bytes + state.reserved_bytes + incoming_bytes;
                if used <= self.config.capacity_bytes || self.config.capacity_bytes == 0 {
                    break;
                }
                let victim = state
                    .records
                    .values()
                    .filter(|meta| meta.active_readers == 0)
                    .min_by_key(|meta| (meta.last_used_unix, meta.header.key_hash))
                    .map(|meta| meta.header.key_hash);
                let Some(hash) = victim else {
                    // Nothing evictable and still over budget: fail the
                    // incoming reservation, never an existing record.
                    exhausted = true;
                    break;
                };
                if let Some(meta) = state.records.remove(&hash) {
                    state.committed_bytes -= meta.file_bytes;
                    state.statistics.capacity_eviction_count += 1;
                    victims.push(meta.path);
                }
            }
        }
        for path in victims {
            self.resident.invalidate(&path);
            let _ = fs::remove_file(path);
        }
        if exhausted {
            bail!("prefix-store capacity exhausted by active records");
        }
        Ok(())
    }

    fn record_path(&self, hash: u64) -> PathBuf {
        self.config
            .root
            .join(format!("{hash:016x}.{RECORD_EXTENSION}"))
    }

    fn capture_temp_path(&self, hash: u64) -> PathBuf {
        let temp_id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        self.config.root.join(format!(
            "{hash:016x}.{RECORD_EXTENSION}.tmp-{}-{temp_id}",
            std::process::id()
        ))
    }
}

enum CaptureBuffer {
    Memory(AlignedBuffer),
    #[cfg(target_os = "linux")]
    Mapped(MappedCapture),
}

impl CaptureBuffer {
    fn new(temp_path: PathBuf, len: usize) -> Result<Self> {
        #[cfg(target_os = "linux")]
        if len >= FILE_BACKED_RECORD_MIN_BYTES {
            return MappedCapture::new(temp_path, len).map(Self::Mapped);
        }
        let _ = temp_path;
        AlignedBuffer::new_zeroed(len).map(Self::Memory)
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            Self::Memory(buffer) => buffer.as_mut_slice(),
            #[cfg(target_os = "linux")]
            Self::Mapped(buffer) => buffer.as_mut_slice(),
        }
    }
}

/// Capture writer; `commit` makes the record durable atomically.
pub struct RecordWriter {
    store: Arc<PrefixStore>,
    header: Option<RecordHeader>,
    tokens: Vec<u32>,
    buffer: Option<CaptureBuffer>,
    permit: Option<WriterPermit>,
}

impl RecordWriter {
    pub fn region_count(&self) -> usize {
        self.header.as_ref().map_or(0, |h| h.regions.len())
    }

    /// Mutable view of one region's bytes inside the record buffer.
    pub fn writable_region(&mut self, index: usize) -> Option<&mut [u8]> {
        let range = self.header.as_ref()?.region_range(index)?;
        self.buffer.as_mut()?.as_mut_slice().get_mut(range)
    }

    /// Synchronous durable commit, retained for Rust callers and tests.
    pub fn commit(mut self) -> Result<()> {
        let job = self.prepare(true)?;
        let store = self.store.clone();
        store.commit_prepared(job)
    }

    /// Queue the already-captured record to the bounded background writer.
    /// Admission occurred before allocating memory or file-backed staging.
    pub fn commit_async(mut self, promote_resident: bool) -> Result<()> {
        let job = self.prepare(promote_resident)?;
        match self.store.writer.sender.try_send(job) {
            Ok(()) => Ok(()),
            Err(error) => {
                let (reason, job) = match error {
                    TrySendError::Full(job) => ("queue full", job),
                    TrySendError::Disconnected(job) => ("writer disconnected", job),
                };
                self.store.cancel_prepared(&job);
                bail!("prefix-record background {reason}")
            }
        }
    }

    fn prepare(&mut self, promote_resident: bool) -> Result<PreparedCommit> {
        let mut header = self.header.take().expect("writer used once");
        let mut buffer = self.buffer.take().expect("writer used once");
        let bytes = buffer.as_mut_slice();
        header.tokens_crc32 = crc32fast::hash(&bytes[header.tokens_range()]);
        header.region_crc32 = (0..header.regions.len())
            .map(|index| {
                let range = header.region_range(index).expect("validated layout");
                crc32fast::hash(&bytes[range])
            })
            .collect();
        if let Err(error) = header.encode(&mut bytes[..HEADER_BYTES]) {
            self.store.cancel_reservation(&header);
            return Err(error);
        }
        Ok(PreparedCommit {
            tokens: Arc::new(std::mem::take(&mut self.tokens)),
            buffer,
            _permit: self.permit.take().expect("writer permit used once"),
            promote_resident,
            header,
        })
    }

    pub fn cancel(mut self) {
        if let Some(header) = self.header.take() {
            let mut state = self.store.state.lock();
            state.reserved_bytes -= header.file_bytes as u64;
            state.reserved_hashes.remove(&header.key_hash);
        }
    }
}

struct PreparedCommit {
    header: RecordHeader,
    tokens: Arc<Vec<u32>>,
    buffer: CaptureBuffer,
    _permit: WriterPermit,
    promote_resident: bool,
}

impl PrefixStore {
    fn cancel_reservation(&self, header: &RecordHeader) {
        let mut state = self.state.lock();
        state.reserved_bytes = state
            .reserved_bytes
            .saturating_sub(header.file_bytes as u64);
        state.reserved_hashes.remove(&header.key_hash);
        state.statistics.failed_capture_count += 1;
    }

    fn cancel_prepared(&self, job: &PreparedCommit) {
        self.cancel_reservation(&job.header);
    }

    fn commit_prepared(&self, job: PreparedCommit) -> Result<()> {
        let PreparedCommit {
            header,
            tokens,
            buffer,
            _permit,
            promote_resident,
        } = job;
        let path = self.record_path(header.key_hash);
        let mut resident_blob = None;
        let write_result = match buffer {
            CaptureBuffer::Memory(buffer) => {
                let temp = self.capture_temp_path(header.key_hash);
                let blob = Arc::new(RecordBlob::new(RecordBuffer::from_memory(buffer)));
                let result = write_atomically(&self.config.root, &temp, &path, blob.as_slice());
                if result.is_err() {
                    let _ = fs::remove_file(&temp);
                } else if promote_resident {
                    resident_blob = Some(blob);
                }
                result
            }
            #[cfg(target_os = "linux")]
            CaptureBuffer::Mapped(buffer) => buffer.commit(&self.config.root, &path),
        };
        if let Err(error) = write_result {
            self.cancel_reservation(&header);
            return Err(error);
        }

        if promote_resident
            && resident_blob.is_none()
            && header.file_bytes <= self.resident.stats().capacity_bytes
        {
            match read_record_checked(&path, &header, self.config.direct_reads) {
                Ok(buffer) => resident_blob = Some(Arc::new(RecordBlob::new(buffer))),
                Err(error) => {
                    let _ = fs::remove_file(&path);
                    self.cancel_reservation(&header);
                    return Err(error);
                }
            }
        }

        let file_bytes = header.file_bytes as u64;
        {
            let mut state = self.state.lock();
            state.reserved_bytes = state.reserved_bytes.saturating_sub(file_bytes);
            state.reserved_hashes.remove(&header.key_hash);
            if let Some(prior) = state.records.remove(&header.key_hash) {
                state.committed_bytes = state.committed_bytes.saturating_sub(prior.file_bytes);
                self.resident.invalidate(&prior.path);
            }
            state.committed_bytes += file_bytes;
            state.statistics.committed_capture_count += 1;
            state.records.insert(
                header.key_hash,
                RecordMeta {
                    tokens,
                    path: path.clone(),
                    file_bytes,
                    last_used_unix: now_unix(),
                    active_readers: 0,
                    header,
                },
            );
        }
        if promote_resident {
            if let Some(blob) = resident_blob {
                self.resident.promote(path, blob);
            }
        }
        Ok(())
    }
}

fn spawn_writer(store: &Arc<PrefixStore>, receiver: Receiver<PreparedCommit>) {
    let store = Arc::downgrade(store);
    std::thread::Builder::new()
        .name("letsinfer-prefix-writer".to_string())
        .spawn(move || writer_loop(store, receiver))
        .expect("spawn Let's Infer prefix writer");
}

fn writer_loop(store: Weak<PrefixStore>, receiver: Receiver<PreparedCommit>) {
    while let Ok(job) = receiver.recv() {
        let Some(store) = store.upgrade() else {
            return;
        };
        let _ = store.commit_prepared(job);
    }
}

impl Drop for RecordWriter {
    fn drop(&mut self) {
        // A dropped (neither committed nor cancelled) writer still releases
        // its reservation; the permit releases via its own Drop.
        if let Some(header) = self.header.take() {
            let mut state = self.store.state.lock();
            state.reserved_bytes -= header.file_bytes as u64;
            state.reserved_hashes.remove(&header.key_hash);
        }
    }
}

/// Pinned reader for one record; the complete immutable blob is loaded and
/// CRC-verified once, then all region reads reuse that resident allocation.
pub struct RecordReader {
    store: Arc<PrefixStore>,
    header: RecordHeader,
    path: PathBuf,
    tokens: Arc<Vec<u32>>,
    blob: Mutex<Option<Arc<RecordBlob>>>,
}

/// Immutable, CRC-verified view of one record region. Keeping this value alive
/// pins the underlying record bytes so an FFI adapter can consume a large
/// region without allocating and copying it a second time.
pub struct LoadedRegion {
    blob: Arc<RecordBlob>,
    range: std::ops::Range<usize>,
}

impl LoadedRegion {
    pub fn as_slice(&self) -> &[u8] {
        &self.blob.as_slice()[self.range.clone()]
    }
}

impl RecordReader {
    pub fn token_count(&self) -> usize {
        self.header.token_count
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn region_count(&self) -> usize {
        self.header.regions.len()
    }

    pub fn region_description(&self, index: usize) -> Option<&RegionDescription> {
        self.header.regions.get(index)
    }

    /// Read one region into `destination`, verifying its CRC. Any failure
    /// is a miss for the caller; the store additionally drops the record
    /// on integrity failure (fail closed).
    pub fn read_region(&self, index: usize, destination: &mut [u8]) -> Result<()> {
        let region = self
            .header
            .regions
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("region index {index} out of range"))?;
        let expected_bytes = usize::try_from(region.byte_count)?;
        if destination.len() != expected_bytes {
            bail!(
                "region '{}' needs {} bytes, destination has {}",
                region.name,
                expected_bytes,
                destination.len()
            );
        }
        let range = self
            .header
            .region_range(index)
            .ok_or_else(|| anyhow::anyhow!("invalid region range"))?;
        let read_result = (|| -> Result<()> {
            let blob = self.load_blob()?;
            destination.copy_from_slice(&blob.as_slice()[range]);
            Ok(())
        })();
        if read_result.is_err() {
            self.store.evict_for_integrity(self.header.key_hash);
        }
        read_result
    }

    /// Load and validate a region once, then return a pinned immutable view.
    /// Integrity failures evict the record exactly like `read_region`.
    pub fn load_region(&self, index: usize) -> Result<LoadedRegion> {
        let range = self
            .header
            .region_range(index)
            .ok_or_else(|| anyhow::anyhow!("region index {index} out of range"))?;
        match self.load_blob() {
            Ok(blob) => Ok(LoadedRegion { blob, range }),
            Err(error) => {
                self.store.evict_for_integrity(self.header.key_hash);
                Err(error)
            }
        }
    }

    #[cfg(feature = "python")]
    pub(crate) fn loaded_region(
        &self,
        index: usize,
    ) -> Result<(Arc<RecordBlob>, std::ops::Range<usize>)> {
        let range = self
            .header
            .region_range(index)
            .ok_or_else(|| anyhow::anyhow!("region index {index} out of range"))?;
        match self.load_blob() {
            Ok(blob) => Ok((blob, range)),
            Err(error) => {
                self.store.evict_for_integrity(self.header.key_hash);
                Err(error)
            }
        }
    }

    fn load_blob(&self) -> Result<Arc<RecordBlob>> {
        let mut local = self.blob.lock();
        if let Some(blob) = local.as_ref() {
            return Ok(blob.clone());
        }
        let (blob, _) = self
            .store
            .resident
            .load(&self.path, self.header.file_bytes, || {
                read_record_checked(&self.path, &self.header, self.store.config.direct_reads)
                    .map(RecordBlob::new)
            })?;
        *local = Some(blob.clone());
        Ok(blob)
    }
}

impl Drop for RecordReader {
    fn drop(&mut self) {
        let mut state = self.store.state.lock();
        state.statistics.active_reader_count =
            state.statistics.active_reader_count.saturating_sub(1);
        if let Some(meta) = state.records.get_mut(&self.header.key_hash) {
            meta.active_readers = meta.active_readers.saturating_sub(1);
        }
    }
}

impl PrefixStore {
    fn evict_for_integrity(&self, hash: u64) {
        let victim = {
            let mut state = self.state.lock();
            state.records.remove(&hash).map(|meta| {
                state.committed_bytes -= meta.file_bytes;
                state.statistics.integrity_eviction_count += 1;
                meta.path
            })
        };
        if let Some(path) = victim {
            self.resident.invalidate(&path);
            let _ = fs::remove_file(path);
        }
    }
}

fn load_record_meta(path: &Path) -> Result<RecordMeta> {
    let (header, tokens) = read_header_and_tokens(path)?;
    if key_hash(&header.fingerprint, &tokens) != header.key_hash {
        bail!("record {} key-hash mismatch", path.display());
    }
    let last_used_unix = fs::metadata(path)?
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let file_bytes = header.file_bytes as u64;
    Ok(RecordMeta {
        header,
        tokens: Arc::new(tokens),
        path: path.to_path_buf(),
        file_bytes,
        last_used_unix,
        active_readers: 0,
    })
}

fn write_atomically(root: &Path, temp: &Path, path: &Path, bytes: &[u8]) -> Result<()> {
    debug_assert!(bytes.len().is_multiple_of(ALIGNMENT));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(temp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(temp, path)?;
    OpenOptions::new().read(true).open(root)?.sync_all()?;
    // Optimization hint only; the bytes are already durable and visible.
    let _ = release_clean_pages(&file);
    Ok(())
}

#[cfg(target_os = "linux")]
fn release_clean_pages(file: &File) -> Result<()> {
    let status = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if status != 0 {
        return Err(std::io::Error::from_raw_os_error(status).into());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn release_clean_pages(_file: &File) -> Result<()> {
    Ok(())
}
