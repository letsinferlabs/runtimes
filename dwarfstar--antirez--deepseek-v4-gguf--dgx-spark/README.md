# DeepSeek V4 Flash / DwarfStar / DGX Spark

Run DeepSeek V4 Flash on one NVIDIA DGX Spark through Let's Infer's stable
local OpenAI-compatible API.

## Features

- **One-command installation** — Let's Infer acquires both exact GGUF
  artifacts, the runtime pack, and Engine OCI, then starts the API.
- **557K context** — a 557,056-token context ceiling without reserving that
  physical allocation for every request.
- **High-concurrency serving** — up to 128 active requests with request-sized
  memory admission and FIFO queueing.
- **Dynamic batching** — short overlapping requests use cohort-aware tuning;
  long and mixed-context cohorts retain the qualified long-context policy.
- **Speculative decoding** — the pinned DwarfStar drafter uses the sealed
  recursive rejection-sampling configuration.
- **Persistent prefix cache** — exact restored-prefix replay with a 64 GiB
  durable store and engine-neutral Let's Infer cache ABI.
- **Reproducible evidence** — the canonical code-and-prose contract binds every
  verifier result to the exact model, engine, target, cache, and serving recipe.

## Install

```bash
letsinfer install deepseek-v4-flash
```

Let's Infer detects the DGX Spark target, verifies the signed catalog,
downloads the exact model and drafter, installs the qualified runtime, and
waits for the local API to become ready.

## Hugging Face artifacts

- Primary model: [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)
- Drafter: [bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF](https://huggingface.co/bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF)

`runtime.json` pins each artifact's exact revision, filename, byte count, and
SHA-256. These links identify the source repositories; they do not replace the
immutable runtime pins.

## Serving contract

The candidate exposes one qualified recipe:

| Capability | Contract |
| --- | --- |
| Target | One NVIDIA DGX Spark, Linux/arm64, SM 12.1 |
| Context ceiling | 557,056 tokens |
| Gateway connections | 128 |
| Active request slots | 128 |
| Prefix cache | Persistent, 64 GiB durable capacity |
| Benchmark output | 128 tokens, deterministic sampling |

The context ceiling is a validity limit, not a fixed per-request allocation.
The rolling scheduler admits from one FIFO and sizes physical state from the
actual prompt and requested generation budget. Completed requests release
their banks immediately. Work that can fit later remains queued until capacity
is available or the client disconnects; active banks are never reclaimed.

Memory admission preserves the runtime's qualified host-memory floors. Short
cohort tuning is enabled only when at least two eligible requests overlap and
the largest known prompt-plus-generation extent is at most 24,576 tokens. C1,
unknown extents, and larger or mixed cohorts use the long-context settings.
Packed multi-request prefill is qualified through 32K; deeper prompts use the
bank-local path.

## Reproduce the benchmark

Run the canonical C1 code-and-prose contexts through the Let's Infer gateway:

```bash
letsinfer benchmark deepseek-v4-flash --c1
```

Add selected concurrency cells without changing the serving recipe:

```bash
letsinfer benchmark deepseek-v4-flash --c2 --c4 --c8
```

Let's Infer creates a fresh process and prefix state for every measured cell,
captures Watchdog telemetry, validates identities and output, and writes the
machine-readable result. Unavailable metrics remain `null` or `-1`; they are
never estimated.

## Build and develop

This directory contains the complete forkable source for the runtime target:

- the pinned DwarfStar engine and CUDA/MMQ/Metal source;
- the Engine protocol adapter and server integration;
- the versioned Let's Infer prefix-cache ABI shim;
- the deterministic Engine OCI recipe;
- target-specific tests and runtime configuration; and
- model, source, benchmark, and license provenance.

Build the Engine image from this directory:

```bash
docker buildx build --pull=false --provenance=false \
  --build-arg SOURCE_DATE_EPOCH=0 \
  --output type=docker,rewrite-timestamp=true -f image/Dockerfile .
```

A fork may change engine source, kernels, cache integration, dependencies,
artifacts, or configuration. Any changed serving identity invalidates the old
qualification evidence and must be benchmarked again.

## License

Let's Infer integration and target-specific modifications are
`AGPL-3.0-only`. Retained DwarfStar engine source remains MIT-licensed under
`engine/LICENSE`; original attributions are recorded in `PROVENANCE.md`.
