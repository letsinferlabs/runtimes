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
> letsinfer model install nemotron-3.5-lightning
> ```

# Nemotron 3.5 Lightning 30B-A3B NVFP4 / vLLM / DGX Spark

Run NVIDIA Nemotron 3.5 Lightning on one NVIDIA DGX Spark through Let's
Infer's stable local OpenAI-compatible API.

## Features

- **One-command installation** -- Let's Infer downloads the exact target and
  DSpark checkpoints, runtime pack, and Engine OCI, then starts the API.
- **1M context** -- serves the model's native 1,048,576-token window.
- **DSpark speculative decoding** -- NVIDIA's immutable draft checkpoint
  proposes three-token blocks while the target preserves output correctness.
- **Low-latency Marlin path** -- atomic reduction accelerates the checkpoint's
  W4A16 routed experts on GB10.
- **Full-graph target verification** -- Triton target attention avoids the
  piecewise-only graph fallback during speculative decoding.
- **Hybrid-cache tuning** -- FlashInfer Mamba state uses aligned prefix caching,
  FP16 SSM storage, and seeded stochastic rounding.
- **Responsive scheduling** -- asynchronous vLLM scheduling serves up to 48
  active requests behind 128 Let's Infer gateway connections.
- **Reproducible deployment** -- model, draft, vLLM image, adapter, target, and
  benchmark contract are all immutable and reviewable.

## Hugging Face artifacts

- Primary model: [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
- DSpark drafter: [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark)

`runtime.json` pins both repositories to exact revisions. Let's Infer acquires
both automatically; you do not preinstall or move model files.

## Engine and recipe

The Engine pins vLLM 0.27.1 commit
`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` through the immutable official
ARM64 image manifest. Serving follows NVIDIA's Nemotron 3.5 Lightning vLLM
recipe with target-specific DGX Spark scheduling and cache settings. Exact
source identities and licenses are recorded in `engine/PROVENANCE.json` and
`THIRD_PARTY.md`.

## Benchmark performance

A sealed payload-bound Let's Infer `pp32768,tg128,c1` workload on one DGX Spark measured:

| Metric | Result |
|---|---:|
| Aggregate throughput | 19.815 tok/s |
| Decode throughput | 124.663 tok/s |
| TTFT | 5.438 s |

The prompt contained 31,126 rendered tokens and is bound to the same canonical
prompt-set SHA used by the prior catalog baseline. Decode throughput improved
16.96%, aggregate throughput improved 60.07%, and TTFT was 40.55% faster.

## Reproduce this

After installing the qualified runtime, run its complete cache-aware benchmark:

```bash
letsinfer benchmark run nemotron-3.5-lightning
```

The schema-8 contract runs short code and prose at C1/C2/C4, plus code at
C1/C2/C4 across 32K, 64K, 128K, and 256K. It finishes with an exact 64K
cold/warm TTFT pair. The runtime supports the full 1M serving window even
though the standard qualification matrix stops at 256K.
