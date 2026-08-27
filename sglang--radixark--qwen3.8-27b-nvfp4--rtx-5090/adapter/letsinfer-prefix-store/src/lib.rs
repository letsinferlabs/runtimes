// SPDX-License-Identifier: AGPL-3.0-only

//! Engine-neutral persistent prefix store (Let's Infer on vLLM, M1).
//!
//! See `source/letsinfer-vllm/plugins/README.md` for the architecture and
//! `context/features/0001_persistent_prefix_cache.md` for the validated
//! design this extracts.

pub mod format;
mod io;
mod ram_tier;
pub mod store;

pub use format::{RegionDescription, ALIGNMENT, HEADER_BYTES, MAX_REGIONS};
pub use store::{
    key_hash, CaptureRejection, LoadedRegion, PrefixStore, RecordReader, RecordWriter, StoreConfig,
    StoreStatistics,
};

#[cfg(feature = "python")]
mod py;
