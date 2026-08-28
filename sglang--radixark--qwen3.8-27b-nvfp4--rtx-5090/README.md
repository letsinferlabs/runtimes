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
- **64K context** -- benchmarked with 65,536-token prompts and a 66,048-token
  serving ceiling that preserves the declared completion budget.
- **Concurrent serving recipe** -- three hardware-backed active requests behind
  64 gateway connections, with excess work queued by the managed service.
- **RTX 5090 tuning** -- the linux/amd64 CUDA 12.9 SGLang Engine, SM120
  FlashInfer attention, exact live-M=9 FP8 tactics, FP8 KV cache, depth-8
  NEXTN, ReplaySSM, chunked prefill, P-core affinity, and a three-request Mamba
  state envelope without the redundant decode lock are sealed together; excess
  admissions remain queued.
  The measured 0.979 static-memory fraction retains the runtime's explicit
  2 GiB physical free-VRAM safety floor.
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

The exact 32K code C1 prequalification cell produced 128 reasoning tokens with
the required marker and restart-reproduced at 250.91--252.34 decode tokens per
second on the declared RTX 5090 target. This local screen is not community
qualification; the complete contract below remains authoritative.

After installing the published runtime, run its complete cache-aware benchmark:

```bash
letsinfer benchmark run qwen3.8-27b
```

The schema-8 contract runs short code and prose plus 32K and 64K code/prose
contexts at C1/C2/C4, then one 64K cold/warm TTFT pair. Its `fresh-context`
lifecycle gives short, 32K, 64K, and the TTFT pair independent processes while
preserving shared-prefix ordering inside each group. Let's Infer owns the
prompts, lifecycle, Watchdog telemetry, cache-hit validation, and the final
`benchmark.json`.
