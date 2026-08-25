// SPDX-License-Identifier: AGPL-3.0-only

//! Byte-bounded residency for validated immutable records.
//!
//! Per-path single-flight elects one NVMe reader while concurrent followers
//! wait for the same `Arc`. LRU eviction removes future reachability without
//! invalidating active readers.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Result};
use parking_lot::{Condvar, Mutex};

#[cfg(test)]
use crate::io::AlignedBuffer;
use crate::io::RecordBuffer;

pub(crate) struct RecordBlob {
    buffer: RecordBuffer,
}

impl RecordBlob {
    pub(crate) fn new(buffer: RecordBuffer) -> Self {
        Self { buffer }
    }

    pub(crate) fn len(&self) -> usize {
        self.buffer.len()
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoadSource {
    Resident,
    Nvme,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TierStats {
    pub(crate) capacity_bytes: usize,
    pub(crate) used_bytes: usize,
    pub(crate) entries: usize,
    pub(crate) loading: usize,
    pub(crate) hit_count: u64,
    pub(crate) nvme_load_count: u64,
}

#[derive(Default)]
struct TierState {
    entries: HashMap<PathBuf, Arc<RecordBlob>>,
    lru: VecDeque<PathBuf>,
    loading: HashSet<PathBuf>,
    used_bytes: usize,
    hit_count: u64,
    nvme_load_count: u64,
}

pub(crate) struct ResidentTier {
    capacity_bytes: usize,
    state: Mutex<TierState>,
    changed: Condvar,
}

impl ResidentTier {
    pub(crate) fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            state: Mutex::new(TierState::default()),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn load<F>(
        &self,
        path: &Path,
        expected_bytes: usize,
        loader: F,
    ) -> Result<(Arc<RecordBlob>, LoadSource)>
    where
        F: FnOnce() -> Result<RecordBlob>,
    {
        if self.capacity_bytes == 0 || expected_bytes > self.capacity_bytes {
            let blob = loader()?;
            validate_size(&blob, expected_bytes)?;
            self.state.lock().nvme_load_count += 1;
            return Ok((Arc::new(blob), LoadSource::Nvme));
        }

        let key = path.to_path_buf();
        loop {
            let mut state = self.state.lock();
            if let Some(blob) = state.entries.get(&key).cloned() {
                touch(&mut state.lru, &key);
                state.hit_count += 1;
                return Ok((blob, LoadSource::Resident));
            }
            if state.loading.insert(key.clone()) {
                break;
            }
            self.changed.wait(&mut state);
        }

        let loaded = loader().and_then(|blob| {
            validate_size(&blob, expected_bytes)?;
            Ok(Arc::new(blob))
        });
        let mut state = self.state.lock();
        state.loading.remove(&key);
        if let Ok(blob) = loaded.as_ref() {
            insert(&mut state, self.capacity_bytes, key, blob.clone());
            state.nvme_load_count += 1;
        }
        self.changed.notify_all();
        loaded.map(|blob| (blob, LoadSource::Nvme))
    }

    pub(crate) fn promote(&self, path: PathBuf, blob: Arc<RecordBlob>) {
        if self.capacity_bytes == 0 || blob.len() > self.capacity_bytes {
            return;
        }
        let mut state = self.state.lock();
        insert(&mut state, self.capacity_bytes, path, blob);
        self.changed.notify_all();
    }

    pub(crate) fn invalidate(&self, path: &Path) {
        let mut state = self.state.lock();
        if let Some(blob) = state.entries.remove(path) {
            state.used_bytes -= blob.len();
            remove_from_lru(&mut state.lru, path);
        }
    }

    pub(crate) fn stats(&self) -> TierStats {
        let state = self.state.lock();
        TierStats {
            capacity_bytes: self.capacity_bytes,
            used_bytes: state.used_bytes,
            entries: state.entries.len(),
            loading: state.loading.len(),
            hit_count: state.hit_count,
            nvme_load_count: state.nvme_load_count,
        }
    }
}

fn validate_size(blob: &RecordBlob, expected_bytes: usize) -> Result<()> {
    if blob.len() != expected_bytes {
        bail!(
            "resident record size mismatch: got {}, expected {expected_bytes}",
            blob.len()
        );
    }
    Ok(())
}

fn insert(state: &mut TierState, capacity: usize, key: PathBuf, blob: Arc<RecordBlob>) {
    if let Some(prior) = state.entries.remove(&key) {
        state.used_bytes -= prior.len();
        remove_from_lru(&mut state.lru, &key);
    }
    while state.used_bytes.saturating_add(blob.len()) > capacity {
        let Some(victim) = state.lru.pop_front() else {
            break;
        };
        if let Some(removed) = state.entries.remove(&victim) {
            state.used_bytes -= removed.len();
        }
    }
    debug_assert!(state.used_bytes.saturating_add(blob.len()) <= capacity);
    state.used_bytes += blob.len();
    state.lru.push_back(key.clone());
    state.entries.insert(key, blob);
}

fn touch(lru: &mut VecDeque<PathBuf>, key: &Path) {
    remove_from_lru(lru, key);
    lru.push_back(key.to_path_buf());
}

fn remove_from_lru(lru: &mut VecDeque<PathBuf>, key: &Path) {
    if let Some(position) = lru.iter().position(|candidate| candidate == key) {
        lru.remove(position);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn blob(bytes: usize, fill: u8) -> RecordBlob {
        let mut buffer = AlignedBuffer::new_zeroed(bytes).unwrap();
        buffer.as_mut_slice().fill(fill);
        RecordBlob::new(RecordBuffer::from_memory(buffer))
    }

    #[test]
    fn resident_hit_runs_loader_once() {
        let tier = ResidentTier::new(4 * 4096);
        let loads = AtomicUsize::new(0);
        let path = Path::new("record-a");
        let first = tier
            .load(path, 4096, || {
                loads.fetch_add(1, Ordering::Relaxed);
                Ok(blob(4096, 7))
            })
            .unwrap();
        let second = tier
            .load(path, 4096, || {
                loads.fetch_add(1, Ordering::Relaxed);
                Ok(blob(4096, 9))
            })
            .unwrap();
        assert_eq!(first.1, LoadSource::Nvme);
        assert_eq!(second.1, LoadSource::Resident);
        assert_eq!(loads.load(Ordering::Relaxed), 1);
        assert_eq!(second.0.as_slice()[0], 7);
    }

    #[test]
    fn byte_lru_evicts_oldest() {
        let tier = ResidentTier::new(2 * 4096);
        for (name, fill) in [("a", 1), ("b", 2), ("c", 3)] {
            tier.load(Path::new(name), 4096, || Ok(blob(4096, fill)))
                .unwrap();
        }
        assert_eq!(tier.stats().entries, 2);
        assert!(tier.state.lock().entries.contains_key(Path::new("b")));
        assert!(tier.state.lock().entries.contains_key(Path::new("c")));
    }
}
