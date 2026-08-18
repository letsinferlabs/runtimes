# Provenance

## Source

- Upstream project: `antirez/ds4`
- Performance fork: `Entrpi/ds4`
- Exact imported revision:
  `6cd114626b91f2d87d3dc349707263beef58cca6`

The complete compute and CUDA/MMQ kernel source is retained from that exact
revision. The runtime owns its target-specific DwarfStar modifications rather
than depending on a hidden engine copy in Let's Infer core.

## Let's Infer integration

The native server adds:

- the ABI-2 DwarfStar adapter for Let's Infer's engine-neutral prefix store;
- projected unified-memory admission and memory-aware FIFO retry;
- a context-scaled transient host allowance that is zero at the proven-safe
  32K boundary and reaches 256 MiB for the candidate and every busy row at
  64K and above, plus one context-scaled allowance while admissions overlap;
  an idle candidate pays only its own allowance;
- one-row idle progress above the external memory-stop line, while concurrent
  admissions preserve a 13 GiB normal floor above Watchdog's 12 GiB warning
  line;
- up to 128 logical execution banks whose request-owned raw, packed-scale, and
  DSpark storage is physically backed only for admitted request extents;
- constant-time full-allocation VMM residency accounting from the allocator's
  maintained mapped-page total plus direct reservation metadata on tensors and
  views, preserving the same admission budget and range checks;
- a no-allocation admission fast path when the selected bank already covers
  the complete request extent and the request needs no transient host reserve;
  any nonzero demand retains every VMM, host-floor, trim, and FIFO gate;
- an internal short-cohort DSpark policy that reproduces the retained
  512-token live-prefill interleave and concurrency-specific verifier-row,
  capture, D2R, and sorted-MoE boundaries during admission and verification,
  while leaving C1 and requests above 24,576 total tokens on the manifest's
  long-context path;
- memory-aware FIFO queueing when the admitted extents do not fit together;
- idle-bank VMM reclaim and deep-bank reuse;
- in-place use of a matching live or durable prefix without duplicate banks.

Active banks are never selected for restore or reclaim. Let's Infer supplies the
gateway, Rust prefix store and C ABI bridge, Watchdog, and benchmark runners at
installation and qualification time; those components are not vendored here.

The exact built server SHA-256 is
`6c8e9e73cdae6a437f5dd7dc3a6c23479e00f12bb0eec12ee602367fd4c5968f`.
Its normalized DGX Spark image is
`sha256:587f5d315db74c48915781e453af725cdae59c7a1d9beaf9b8627bf4065015b7`.
The server was built from the source retained here and includes the exact
rendered-chat token-count endpoint used to materialize the standard benchmark.

## Distribution boundary

The runtime includes only the release configuration, image recipe,
corresponding engine and kernel source, licenses, concise provenance, and the
structured public benchmark result. It excludes container layers, model data,
caches, raw benchmark evidence and prompts, operational scripts, unrelated
build outputs, private instructions, and nested Git metadata.

Let's Infer runtime integration and target-specific modifications are distributed
under the top-level AGPL-3.0-only `LICENSE`. The inherited engine source keeps
its MIT terms in `engine/LICENSE`. Preserve both licenses and the original
attribution when publishing a fork.
