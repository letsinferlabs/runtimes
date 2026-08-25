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
> letsinfer install nemotron-3.5-lightning
> ```

# Nemotron 3.5 Lightning 30B-A3B NVFP4 / SGLang / DGX Spark

Run NVIDIA Nemotron 3.5 Lightning on one NVIDIA DGX Spark through Let's
Infer's stable local OpenAI-compatible API.

## Features

- **One-command installation** -- Let's Infer downloads the exact target and
  DSpark checkpoints, runtime pack, and Engine OCI, then starts the API.
- **1M context** -- serves the model's native 1,048,576-token window.
- **DSpark speculative decoding** -- the official NVIDIA draft checkpoint
  proposes three-token blocks for low-concurrency latency and throughput.
- **Hybrid-cache tuning** -- FP16 Mamba state and the target's native attention
  cache fit the complete model and draft on one 128 GiB unified-memory GB10.
- **Native SGLang scheduling** -- up to 48 active requests behind 128 gateway
  connections, with Let's Infer admission and queueing.
- **Reproducible evidence** -- canonical code-and-prose measurements bind the
  exact checkpoints, Engine image, target, and serving recipe.

## Hugging Face artifacts

- Primary model: [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
- DSpark drafter: [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark)

`runtime.json` pins both repositories to exact revisions. Let's Infer acquires
both automatically; you do not preinstall or move model files.

## Recipe lineage

The serving flags follow NVIDIA's Nemotron 3.5 Lightning SGLang cookbook and
the DGX Spark integration published by MiaAI-Lab. The Engine pins SGLang commit
`d59c1ddf70ee17fcc41c053ed38bd60bc6cc28cc` through the immutable official
ARM64 image manifest. Exact source identities and licenses are recorded in
`engine/PROVENANCE.json` and `THIRD_PARTY.md`.

## Reproduce this

After installing the qualified runtime, run its complete cache-aware
benchmark:

```bash
letsinfer benchmark nemotron-3.5-lightning
```

The schema-7 contract runs short code and prose at C1/C2/C4, plus code at
C1/C2/C4 across 32K, 64K, 128K, and 256K. It finishes with an exact 64K
cold/warm TTFT pair. The runtime supports the full 1M serving window even
though the standard qualification matrix stops at 256K.
