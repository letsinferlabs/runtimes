// SPDX-License-Identifier: AGPL-3.0-only

//! Engine-neutral persistent prefix store used by the Let's Infer vLLM
//! connector. Records bind exact tokens and opaque named state regions, use
//! page-aligned CRC-protected storage, and commit atomically.

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
