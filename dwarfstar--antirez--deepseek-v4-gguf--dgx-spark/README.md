# DeepSeek V4 Flash / DwarfStar / DGX Spark

## Hugging Face artifacts

- Primary model: [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)
- Drafter: [bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF)

`runtime.json` pins the exact revision, filename, byte count, and SHA-256 for
each artifact. These links identify the source repositories; they do not
replace the immutable runtime pins.

This repository is the complete, forkable source form of one Let's Infer runtime
target. It is not a thin configuration wrapper around code hidden in Let's Infer.
It contains:

- Let's Infer runtime and release configuration;
- the exact retained-fast `6cd1146` DwarfStar engine and
  CUDA/MMQ/Metal source;
- the Let's Infer server integration and the proven ABI-2 zero-copy cache adapter;
- a pinned image recipe that reproducibly builds the DGX Spark server;
- the small DwarfStar-to-Let's Infer cache ABI shim;
- a declarative standard benchmark contract in `runtime.json` and its sealed
  machine-readable result in `benchmark.json`;
- license and concise source provenance.

Private development notes, experiment scripts, benchmark runners, materialized
prompts, evidence, model-conversion utilities, operational scripts, Watchdog,
the Let's Infer prefix store, and its Rust C ABI bridge are not vendored into this
runtime. Let's Infer core owns the versioned model-neutral prompt generator and
runner. This runtime declares only workloads, token counts, concurrency,
request limits, seeds, and the exact engine/model identity used to count the
rendered prompts. The engine implementation and its target-specific
optimizations are authoritative here, not in Let's Infer core.

The production recipe uses multi-request segmented prefill through 32K and the
single-bank path for deeper prompts. Its immutable image and server binary are
reproducible from the pinned source. The manifest exposes up to 128 admitted
requests with memory-aware FIFO queueing and a maximum context of 557,056
tokens. `benchmark.json` is the structured performance record for the exact
engine, model, target, serving recipe, and standard workload contract. The
release manifest is the authority for activation and qualification state.
Public distribution requires pullable immutable OCI identities for the exact
image and runtime pack.

A fork can change any engine source, kernel, cache adapter, dependency, or
image-build step and then build and pack that fork with Let's Infer. Build the
target image from the repository root:

```bash
docker buildx build --pull=false --provenance=false \
  --build-arg SOURCE_DATE_EPOCH=0 \
  --output type=docker,rewrite-timestamp=true -f image/Dockerfile .
```

Let's Infer performs that same build for a local candidate when its manifest pins
the expected local image ID. The recipe builds the retained engine source and
normalizes the resulting layer metadata. Two independent no-cache builds must
produce the same image ID. A published registry release may distribute that
image by OCI digest while retaining this source and Dockerfile for audit and
offline reconstruction.

## Reproduce this

After installing the runtime, Let's Infer materializes the standard prompts,
starts every cell with a fresh process and prefix store, collects Watchdog
telemetry, and validates the resulting JSON:

```bash
letsinfer benchmark deepseek-v4-flash --c1 --c2 --c4 --c8
```

To run the C16 32K, 64K, and 128K cells:

```bash
letsinfer benchmark deepseek-v4-flash \
  --c16 --32k --64k --128k
```

The sealed `benchmark.json` contains 24 neutral `ppN,tgN,cN` rows. Metrics that
are unavailable remain `null` or `-1`; Let's Infer does not invent telemetry.
Its version-2 telemetry schema records NVMe capacity usage, temperature, and
read/write throughput directly from Watchdog when the host exposes each value.

## License

Let's Infer runtime integration and target-specific modifications are distributed
under `AGPL-3.0-only`; the complete terms are in the top-level `LICENSE`.
Retained upstream DwarfStar/DS4 engine code remains under its MIT terms in
`engine/LICENSE`. The runtime distribution therefore contains both
`AGPL-3.0-only` and MIT-licensed source, with the original attributions
preserved in `PROVENANCE.md`.

The manifest exposes one serving recipe rather than user-facing profiles. It
fixes the 557,056-token context ceiling, the engine-default 4,096-token
coalescing scratch, 2,048-token continuous-prefill chunks, depth-2 recursive
top-2 rejection sampling, no non-expert weight cache, the automatic VMM budget,
the default 1,024-column D2R crossover, and no verifier-row cap. The context
ceiling is a validity limit, not a 557,056-token physical allocation for every
request. Benchmark selection changes only workload shape; it does not create a
serving profile or alter the manifest's engine settings.

## Request scheduling

Let's Infer supplies the fixed serving contract and safety envelope; DwarfStar
owns request scheduling inside that envelope. The scheduler behaves as
follows:

- The API and engine expose 128 logical request slots. Each admitted request's
  physical footprint is sized from its actual prompt plus requested generation
  budget. Unused logical banks own no physical pages. Sixty 8K requests can
  therefore be admitted concurrently when their combined state and runtime
  scratch fit; the 557,056-token ceiling does not reserve six or any other
  fixed number of banks. The only fixed ceilings are 128 slots and available
  memory.
- The continuous scheduler maintains a rolling active set. It admits from one
  FIFO, immediately reuses a bank released by a completed request, and keeps
  each request's sampling state, stream, stop conditions, tools, and output
  independent. A memory-blocked FIFO head is retried before newer work. It
  remains queued until memory admission succeeds, the client disconnects, or
  shutdown begins; there is no payload-size-dependent retry deadline.
- Admission checks both the request-sized VMM budget and the current host-memory
  floor. Deep cold and durable-restore admissions reuse idle banks rather than
  accumulating duplicate mappings. If necessary, other idle banks release VMM
  pages; active banks are never reclaimed. While another row is active, the DGX
  Spark engine preserves a 13 GiB host-memory floor: 1 GiB above Watchdog's
  warning line and 3 GiB above its graceful-stop line. An otherwise-idle queue
  may admit one row against the lower 10.125 GiB progress floor. Further work
  remains queued until the active row releases enough capacity; active banks
  are never reclaimed.
- Short-request compute tuning is cohort-wide because one GPU batch shares its
  kernel policy. It activates only when at least two requests overlap and the
  largest known prompt-plus-generation extent is at most 24,576 tokens. The
  runtime then selects verifier-fit rows, capture rows, D2R crossover, sorted
  wide-MoE dispatch, and a 512-token live-prefill interleave from the current
  cohort width. C1, an unknown extent, or any larger request fails closed to the
  manifest's long-context settings.
- Consequently, a mixed 8K and 64K pair gets independent request-sized memory,
  but the shared compute cohort uses the long-context policy because the 64K
  request crosses the short-cohort boundary. The 8K request receives the short
  compute policy when it overlaps only eligible short requests. This is dynamic
  request allocation plus cohort-aware compute tuning, not two incompatible
  kernel configurations inside one GPU batch.
- A cold overlapping cohort may share projection, FFN, and MoE work while
  attention remains bank-local for every request. Packed prefill shares the
  existing 4,096-row graph capacity, never exceeds 2,048 rows for one bank, and
  publishes all final prompt seeds before decode. The server waits at most 20 ms
  once for the initial burst so ordinary HTTP arrival skew does not split it.
  C1 never enters this segmented path. Packed cohorts stop at the 32K prompt
  frontier; deeper chunks use the same bank-local path as C1 because the deep
  streaming top-k kernel is not qualified with segmented attention.
- Requests supported by the continuous OpenAI path can batch even with
  per-request temperature, sampling, streaming, thinking, stop strings, and
  tool-call behavior. Requests requiring the richer serial path remain FIFO
  ordered and use a request-fit serial graph where possible rather than silently
  changing semantics.

A cold admission plan is recomputed when released capacity changes its target.
Once paid, warm, or idle-settle work begins, retries retain the exact
target/source/cache/seed state. The Let's Infer gateway reserves two worker slots
for health and lifecycle traffic outside the 128 inference-connection ceiling,
so full inference occupancy cannot starve the container healthcheck.

The request-sized allocator reads its maintained total of mapped VMM pages in
constant time and carries direct reservation metadata on tensors and views;
when an existing bank already covers the request and no transient host reserve
is required, it skips the no-op global projections and mapping walk. Admission
limits and every nonzero-demand safety gate are unchanged.
`DS4_DSPARK_AUTO_BATCH=0` disables the short-cohort policy, and
`DS4_CONT_PREFILL_SEGMENTED_COHORT=0` disables packed cold-cohort prefill.

`engine/ds4_letsinfer_cache.c` only translates DwarfStar bank payloads to the
versioned Let's Infer ABI. It is part of this modified engine, not a cache
implementation. At runtime Let's Infer mounts
`libletsinfer_prefix_capi.so`, which contains the authoritative engine-neutral
store. Its production capture path syncs and unmaps each durable file-backed
record before asking Linux to discard the completed writeback pages, so cache
persistence does not accumulate against DwarfStar's admission floor.
