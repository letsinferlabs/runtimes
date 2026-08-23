# letsinfer-bridge: persistent prefix cache for DwarfStar

This runtime-owned bridge gives `ds4-server` access to the engine-neutral Let's
Infer NVMe prefix store through a thin, versioned C ABI. The runtime carries the
bridge source and a pinned build snapshot; core remains the owner of the cache
contract.

## Features

- **Exact restored replay** — opaque named state regions are validated before
  reuse and corruption is treated as a cache miss.
- **Crash-safe durability** — page-aligned CRC records commit through temp file,
  `fsync`, atomic rename, and directory `fsync`.
- **Bounded storage** — exact-byte LRU capacity, sliding TTL, and a bounded
  background writer prevent unbounded growth.
- **Unified-memory aware** — large captures and restores use same-filesystem
  mappings instead of a second record-sized anonymous allocation.
- **Optional and fail-open** — DwarfStar continues serving with the persistent
  cache disabled when the verified bridge is unavailable.
- **Versioned ABI** — the server checks the C ABI version before using the
  bridge; the runtime pins and verifies the built artifact.

```
ds4_server.c ──> ds4_letsinfer_cache.c ──dlopen──> libletsinfer_prefix_capi.so
                                                (capi/, Rust cdylib)
                                                        │
                                          vendor/letsinfer_prefix_store
                                          (Let's Infer-core build snapshot)
```

## Layout

- `capi/` — the `letsinfer_prefix_capi` crate: a panic-safe C ABI
  (`letsinfer_prefix_*` symbols), versioned through
  `letsinfer_prefix_abi_version()` and built as a cdylib.
- `vendor/letsinfer_prefix_store/` — an exact Let's Infer-owned build snapshot of
  the authoritative `cache/letsinfer_prefix_store` logic. It lets this runtime
  build the native bridge as a closed, pinned workspace and provides the region
  view used by C ABI v2, avoiding a second payload-sized allocation during
  DwarfStar restore. This is the validated engine-neutral `persistent_prefix`
  store:
  CRC'd page-aligned records of opaque named state regions, exact-token
  authority, atomic commits (temp file + fsync + rename + dir fsync),
  exact-byte LRU capacity, sliding TTL, a bounded background writer (one
  active + one queued), optional host-RAM residency, and O_DIRECT reads.
  Captures and durable restores at or above 64 MiB use same-filesystem file
  mappings rather than record-sized anonymous allocations, preserving
  unified-memory safety while retaining CRC validation and atomic
  fsync/rename commits.

## Build

```
cd letsinfer-bridge && cargo build --release
# or from the repo root: make letsinfer-bridge
```

produces `target/release/libletsinfer_prefix_capi.so`. No part of the normal
`make cuda` / `make cpu` build depends on this: the five ds4 binaries build
and run without a Rust toolchain, and without the .so present ds4-server
simply logs one line and runs with the cache disabled (fail-open).

Let's Infer builds the bridge with the runtime manifest's digest-pinned builder
image for the selected target and verifies the declared artifact SHA-256 before
installation. A manual host build or file copy is not part of deployment. For
local development, use an architecture-matched pinned Rust image:

```
docker run --rm -v "$PWD":/work -w /work \
    registry.example/rust-builder@sha256:<exact-digest> \
    cargo build --release
```

## Runtime

`ds4-server` dlopens the library only when `DS4_LETSINFER_CACHE=1`; default
path is `libletsinfer_prefix_capi.so` next to the ds4-server binary, override
with `DS4_LETSINFER_CACHE_LIB`. All flags are documented in
the DwarfStar runtime's `ds4_letsinfer_cache.h`.

## Tests

```
cd letsinfer-bridge && cargo test --workspace --release
```

runs the Let's Infer store snapshot's validation suite: record round-trips,
capacity LRU, TTL, corruption-as-miss, and writer admission.
