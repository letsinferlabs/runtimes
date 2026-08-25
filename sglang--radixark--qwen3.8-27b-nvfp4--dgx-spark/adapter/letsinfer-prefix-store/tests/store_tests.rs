// SPDX-License-Identifier: AGPL-3.0-only

//! Store-level acceptance tests mirroring the feature record's list:
//! round-trip, fingerprint mismatch → miss, corruption → miss + eviction,
//! restart survival, capacity LRU, TTL expiry, writer admission.

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use letsinfer_prefix_store::{CaptureRejection, PrefixStore, RegionDescription, StoreConfig};

const FP_A: [u8; 32] = [0xA1; 32];
const FP_B: [u8; 32] = [0xB2; 32];

fn config(root: &std::path::Path, capacity: u64) -> StoreConfig {
    StoreConfig {
        root: root.to_path_buf(),
        capacity_bytes: capacity,
        ttl_seconds: 3600,
        minimum_token_count: 4,
        resident_capacity_bytes: 1 << 20,
        direct_reads: false,
    }
}

fn regions() -> Vec<RegionDescription> {
    vec![
        RegionDescription {
            name: "paged_kv".into(),
            byte_count: 8192,
        },
        RegionDescription {
            name: "gdn_ssm".into(),
            byte_count: 512,
        },
    ]
}

fn capture(
    store: &std::sync::Arc<PrefixStore>,
    fingerprint: [u8; 32],
    tokens: &[u32],
    kv_fill: u8,
) {
    let mut writer = store
        .begin_capture(fingerprint, tokens, regions())
        .expect("capture admitted");
    writer.writable_region(0).unwrap().fill(kv_fill);
    writer.writable_region(1).unwrap().fill(kv_fill ^ 0xFF);
    writer.commit().expect("commit");
}

#[test]
fn exact_hit_round_trips_all_regions() {
    let dir = tempfile::tempdir().unwrap();
    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    capture(&store, FP_A, &tokens, 0x5A);

    // Longer request sharing the captured prefix must hit at 64 tokens.
    let request: Vec<u32> = (0..100).collect();
    let reader = store.longest_prefix(FP_A, &request, 4).expect("hit");
    assert_eq!(reader.token_count(), 64);
    assert_eq!(reader.tokens(), &tokens[..]);
    let mut kv = vec![0u8; 8192];
    reader.read_region(0, &mut kv).unwrap();
    assert!(kv.iter().all(|&b| b == 0x5A));
    let mut ssm = vec![0u8; 512];
    reader.read_region(1, &mut ssm).unwrap();
    assert!(ssm.iter().all(|&b| b == 0xA5));
}

#[test]
fn longest_of_multiple_captured_boundaries_wins() {
    let dir = tempfile::tempdir().unwrap();
    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    let tokens: Vec<u32> = (0..128).collect();
    capture(&store, FP_A, &tokens[..32], 1);
    capture(&store, FP_A, &tokens[..96], 2);
    let reader = store.longest_prefix(FP_A, &tokens, 4).expect("hit");
    assert_eq!(reader.token_count(), 96);
}

#[test]
fn fingerprint_mismatch_is_a_miss() {
    let dir = tempfile::tempdir().unwrap();
    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    capture(&store, FP_A, &tokens, 0x11);
    assert!(store.longest_prefix(FP_B, &tokens, 4).is_none());
}

#[test]
fn diverging_tokens_are_a_miss() {
    let dir = tempfile::tempdir().unwrap();
    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    capture(&store, FP_A, &tokens, 0x11);
    let mut diverged = tokens.clone();
    diverged[10] = 9999;
    assert!(store.longest_prefix(FP_A, &diverged, 4).is_none());
}

#[test]
fn store_survives_reopen_restart() {
    let dir = tempfile::tempdir().unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    {
        let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
        capture(&store, FP_A, &tokens, 0x77);
    }
    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    let reader = store.longest_prefix(FP_A, &tokens, 4).expect("restart hit");
    let mut kv = vec![0u8; 8192];
    reader.read_region(0, &mut kv).unwrap();
    assert!(kv.iter().all(|&b| b == 0x77));
}

#[test]
fn corrupted_payload_fails_closed_and_is_evicted() {
    let dir = tempfile::tempdir().unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    {
        let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
        capture(&store, FP_A, &tokens, 0x42);
    }
    // Flip one payload byte in the sole committed record.
    let record = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().and_then(|e| e.to_str()) == Some("letsinfer_prefix"))
        .expect("record file");
    let mut file = OpenOptions::new().write(true).open(&record).unwrap();
    file.seek(SeekFrom::Start(8192)).unwrap(); // inside region 0
    file.write_all(&[0xFF]).unwrap();
    drop(file);

    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    // Header and tokens still validate, so the record indexes; the region
    // read must fail closed and evict.
    if let Some(reader) = store.longest_prefix(FP_A, &tokens, 4) {
        let mut kv = vec![0u8; 8192];
        assert!(reader.read_region(0, &mut kv).is_err());
        drop(reader);
        assert_eq!(store.statistics().integrity_eviction_count, 1);
        assert_eq!(store.statistics().record_count, 0);
    } else {
        // Startup validation already rejected it — equally fail-closed.
        assert_eq!(store.statistics().record_count, 0);
    }
}

#[test]
fn truncated_record_is_rejected_at_startup() {
    let dir = tempfile::tempdir().unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    {
        let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
        capture(&store, FP_A, &tokens, 0x42);
    }
    let record = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().and_then(|e| e.to_str()) == Some("letsinfer_prefix"))
        .unwrap();
    let full = std::fs::metadata(&record).unwrap().len();
    let file = OpenOptions::new().write(true).open(&record).unwrap();
    file.set_len(full - 4096).unwrap();
    drop(file);

    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    assert_eq!(store.statistics().record_count, 0);
    assert!(store.longest_prefix(FP_A, &tokens, 4).is_none());
}

#[test]
fn capacity_evicts_oldest_record_first() {
    let dir = tempfile::tempdir().unwrap();
    // Each record: 4 KiB header + 4 KiB tokens + 8 KiB kv + 4 KiB ssm = 20 KiB.
    let store = PrefixStore::open(config(dir.path(), 45 << 10)).unwrap();
    let first: Vec<u32> = (0..64).collect();
    let second: Vec<u32> = (1000..1064).collect();
    let third: Vec<u32> = (2000..2064).collect();
    capture(&store, FP_A, &first, 1);
    std::thread::sleep(std::time::Duration::from_millis(1100));
    capture(&store, FP_A, &second, 2);
    // Third capture must evict `first` (oldest last-used) to fit.
    capture(&store, FP_A, &third, 3);
    assert!(store.longest_prefix(FP_A, &first, 4).is_none());
    assert!(store.longest_prefix(FP_A, &second, 4).is_some());
    assert!(store.longest_prefix(FP_A, &third, 4).is_some());
    assert!(store.statistics().capacity_eviction_count >= 1);
}

#[test]
fn ttl_reclaims_idle_records_only() {
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), 1 << 20);
    cfg.ttl_seconds = 10;
    let store = PrefixStore::open(cfg).unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    capture(&store, FP_A, &tokens, 1);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    store.reclaim_expired(now + 5).unwrap();
    assert_eq!(store.statistics().record_count, 1, "within TTL");
    store.reclaim_expired(now + 60).unwrap();
    assert_eq!(store.statistics().record_count, 0, "past TTL");
    assert_eq!(store.statistics().expiry_eviction_count, 1);
}

#[test]
fn writer_admission_allows_two_then_rejects() {
    let dir = tempfile::tempdir().unwrap();
    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    let first: Vec<u32> = (0..32).collect();
    let second: Vec<u32> = (100..132).collect();
    let third: Vec<u32> = (200..232).collect();
    let writer_one = store.begin_capture(FP_A, &first, regions()).unwrap();
    let writer_two = store.begin_capture(FP_A, &second, regions()).unwrap();
    assert!(matches!(
        store.begin_capture(FP_A, &third, regions()),
        Err(CaptureRejection::WriterBusy)
    ));
    drop(writer_one);
    let replacement = store.begin_capture(FP_A, &third, regions());
    assert!(replacement.is_ok());
    drop(writer_two);
    assert_eq!(store.statistics().reserved_byte_count, 20 << 10);
}

#[cfg(target_os = "linux")]
#[test]
fn large_capture_stages_in_store_file_and_cancel_cleans_it() {
    let dir = tempfile::tempdir().unwrap();
    let store = PrefixStore::open(config(dir.path(), 128 << 20)).unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    let large = vec![RegionDescription {
        name: "paged_kv".into(),
        byte_count: 65 << 20,
    }];
    let mut writer = store.begin_capture(FP_A, &tokens, large).unwrap();
    writer.writable_region(0).unwrap()[0] = 0xA5;

    let temporary = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .contains(".tmp-")
        })
        .expect("large capture should use file-backed staging");
    assert!(std::fs::metadata(temporary).unwrap().len() >= 65 << 20);

    writer.cancel();
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    assert_eq!(store.statistics().reserved_byte_count, 0);
    assert_eq!(store.statistics().record_count, 0);
}

#[cfg(target_os = "linux")]
#[test]
fn large_file_backed_capture_commits_async_and_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    let large = vec![RegionDescription {
        name: "paged_kv".into(),
        byte_count: 65 << 20,
    }];
    {
        let store = PrefixStore::open(config(dir.path(), 128 << 20)).unwrap();
        let mut writer = store.begin_capture(FP_A, &tokens, large).unwrap();
        let region = writer.writable_region(0).unwrap();
        region[0] = 0xA5;
        region[region.len() - 1] = 0x5A;
        writer.commit_async(false).unwrap();
        for _ in 0..500 {
            if store.statistics().record_count == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(store.statistics().record_count, 1);
    }

    let store = PrefixStore::open(config(dir.path(), 128 << 20)).unwrap();
    let reader = store.longest_prefix(FP_A, &tokens, 4).expect("restart hit");
    let mut region = vec![0u8; 65 << 20];
    reader.read_region(0, &mut region).unwrap();
    assert_eq!(region[0], 0xA5);
    assert_eq!(region[region.len() - 1], 0x5A);
    assert_eq!(store.statistics().record_count, 1);
}

#[test]
fn duplicate_capture_is_rejected_as_already_stored() {
    let dir = tempfile::tempdir().unwrap();
    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    let tokens: Vec<u32> = (0..64).collect();
    capture(&store, FP_A, &tokens, 1);
    assert!(matches!(
        store.begin_capture(FP_A, &tokens, regions()),
        Err(CaptureRejection::AlreadyStored)
    ));
}

#[test]
fn prewarm_candidates_filter_fingerprint_prioritize_capsules_and_fit_budget() {
    let dir = tempfile::tempdir().unwrap();
    let store = PrefixStore::open(config(dir.path(), 1 << 20)).unwrap();
    let plain: Vec<u32> = (0..32).collect();
    let capsule: Vec<u32> = (100..116).collect();
    let incompatible: Vec<u32> = (200..264).collect();
    capture(&store, FP_A, &plain, 1);

    let capsule_regions = vec![
        RegionDescription {
            name: "paged_kv".into(),
            byte_count: 8192,
        },
        RegionDescription {
            name: "hidden".into(),
            byte_count: 512,
        },
    ];
    let mut writer = store
        .begin_capture(FP_A, &capsule, capsule_regions)
        .unwrap();
    writer.writable_region(0).unwrap().fill(2);
    writer.writable_region(1).unwrap().fill(3);
    writer.commit().unwrap();
    capture(&store, FP_B, &incompatible, 4);

    let candidates = store.prewarm_candidates(FP_A, u64::MAX);
    assert_eq!(candidates.len(), 2);
    assert_eq!(candidates[0].0, capsule);
    assert!(candidates[0].1);
    assert_eq!(candidates[1].0, plain);
    assert!(!candidates[1].1);

    let exact_budget = candidates[0].2;
    let limited = store.prewarm_candidates(FP_A, exact_budget);
    assert_eq!(limited.len(), 1);
    assert_eq!(limited[0].0, capsule);
}

#[test]
fn below_minimum_and_oversized_captures_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store = PrefixStore::open(config(dir.path(), 16 << 10)).unwrap();
    let short: Vec<u32> = vec![1, 2];
    assert!(matches!(
        store.begin_capture(FP_A, &short, regions()),
        Err(CaptureRejection::BelowMinimumTokenCount)
    ));
    let tokens: Vec<u32> = (0..64).collect();
    // 20 KiB record > 16 KiB capacity.
    assert!(matches!(
        store.begin_capture(FP_A, &tokens, regions()),
        Err(CaptureRejection::RecordTooLarge)
    ));
}
