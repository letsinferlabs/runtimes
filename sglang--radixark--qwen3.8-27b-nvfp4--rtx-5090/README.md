> **Run this model with [Let's Infer](https://letsinfer.ai/).**
>
> Install Let's Infer first:
>
> ```sh
> curl -fsSL https://letsinfer.ai/install.sh | sh
> ```
>
> Then install this model:
>
> ```sh
> letsinfer model install qwen3.8-27b
> ```

# Qwen3.8 27B NVFP4 / SGLang / RTX 5090

Run Qwen3.8 27B on one NVIDIA GeForce RTX 5090 through Let's Infer's stable local
OpenAI-compatible API.

## Features

- **One-command installation** -- Let's Infer downloads the exact model,
  runtime pack, and Engine OCI, then starts the API.
- **128K context** -- a 131,200-token serving ceiling reserves the required
  128-token completion after a nominal 128K prompt. FP8 target and draft KV
  cache preserve the model's complete vision capability on one 32 GiB GPU.
- **Native SGLang scheduling** -- up to four active requests behind 64 gateway
  connections, with dynamic admission and queueing.
- **RTX 5090 tuning** -- the linux/amd64 CUDA 12.9 SGLang Engine, SM120
  TRT-LLM decode attention, FlashInfer prefill, FP8 KV cache, depth-8 NEXTN,
  ReplaySSM, chunked prefill, P-core affinity, and a 16-slot Mamba state
  envelope are sealed together.
- **Full multimodal model** -- the vision tower remains enabled. The embedding
  and vision BF16 weights use fail-closed CUDA host mappings to reserve enough
  VRAM for the 128K KV pool without changing their values.
- **Safe cache policy** -- SGLang process-local prefix reuse is enabled; this
  candidate makes no persistent-restart cache claim.
- **Reproducible evidence** -- the canonical code-and-prose contract binds every
  verifier result to the exact model, engine, target, and serving recipe.

## Hugging Face artifacts

- Primary model: [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
`runtime.json` pins the exact model revision. This link identifies the source
repository; it does not replace that immutable pin.

## Runtime candidate

This directory is the complete source form of one Let's Infer runtime
candidate. It binds that checkpoint to one digest-pinned SGLang Engine OCI and
the RTX 5090 target. The adapter, Engine source closure, image recipe,
benchmark contract, qualification source, and target-specific settings live
beside `runtime.json`; generic gateway, Watchdog, prompt, and benchmark-runner
code stays in Let's Infer core.

After qualification and catalog publication, install the RTX 5090 runtime with:

```bash
letsinfer model install qwen3.8-27b
```

Let's Infer then downloads the runtime pack, this exact model revision, and
the pinned Engine OCI. You do not download or place the weights separately.

## Reproduce this

The complete schema-8 author run passed all 26 declared cells on one RTX 5090.
Fresh-context code C1 decode measured 244.773959, 202.089955, and 179.417978
tokens per second at 32K, 64K, and 128K, with TTFT of 4.670389, 11.433039,
and 33.946240 seconds. At 128K, C2 and C4 median stream decode measured
167.121942 and 166.052119 tokens per second. The standard 64K cache lane reduced
TTFT from 10.972137 seconds cold to 0.268811 seconds warm. Benchmark ID is
`0f170ef7…d5594`; record SHA-256 is `abbb02d2…17874` and results SHA-256 is
`fc22a98d…d2431`.

The same immutable Engine had previously measured 260.621477 tokens per second
at 32K and 152.276412 at 128K in a focused code-C1 screen. That narrower result
is method-separated from the complete fresh-context matrix above.

After installing the published runtime, run its complete cache-aware benchmark:

```bash
letsinfer benchmark qwen3.8-27b
```

The schema-8 contract gives short, 32K, 64K, 128K, and the TTFT lane independent
fresh processes. It runs code and prose at C1/C2/C4 inside each context tier,
then the standard 64K cold/warm cache-TTFT pair. Let's Infer owns the prompts, shared-prefix
ordering inside each tier, lifecycle, Watchdog telemetry, cache-hit validation,
and the final `benchmark.json`.
