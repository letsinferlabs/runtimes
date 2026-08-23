# Qwen3.8 27B NVFP4 / SGLang / DGX Spark

Run Qwen3.8 27B on one NVIDIA DGX Spark through Let's Infer's stable local
OpenAI-compatible API.

## Features

- **One-command installation** — Let's Infer downloads the exact model,
  DFlash2 drafter, runtime pack, and Engine OCI, then starts the API.
- **262K context** — qualified with a 262,144-token context ceiling.
- **Native SGLang scheduling** — up to 10 active requests behind 128 gateway
  connections, with dynamic admission and queueing.
- **Speculative decoding** — the pinned DFlash2 drafter produces up to eight
  draft tokens per step.
- **DGX Spark tuning** — FlashInfer attention, FP8 KV cache, chunked prefill,
  Mamba cache policy, and target-specific Engine configuration are sealed in
  the candidate.
- **Reproducible evidence** — the canonical code-and-prose contract binds every
  verifier result to the exact model, engine, target, and serving recipe.

## Hugging Face artifacts

- Primary model: [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- DFlash drafter: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)

`runtime.json` pins both exact revisions. These links identify the source
repositories; they do not replace those immutable pins.

## Runtime candidate

This directory is the complete source form of one Let's Infer runtime
candidate. It binds that checkpoint to one digest-pinned SGLang Engine OCI and
the DGX Spark target. The adapter, Engine source closure, image recipe,
benchmark contract, qualification source, and target-specific settings live
beside `runtime.json`; generic gateway, Watchdog, prompt, and benchmark-runner
code stays in Let's Infer core.

Install the qualified DGX Spark candidate with:

```bash
letsinfer install qwen3.8-27b
```

Let's Infer then downloads the runtime pack, this exact model revision, and
the pinned Engine OCI. You do not download or place the weights separately.

## Reproduce this

After installing the qualified runtime, run its complete canonical C1
code-and-prose context set through the unified gateway:

```bash
letsinfer benchmark qwen3.8-27b --c1
```

Let's Infer materializes the standard prompts with the exact Engine tokenizer,
starts each cell with an isolated lifecycle and cache, records Watchdog
telemetry, and validates the generated `benchmark.json`.
