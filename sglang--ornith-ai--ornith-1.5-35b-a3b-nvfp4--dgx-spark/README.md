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
> letsinfer install ornith-1.5-35b-a3b
> ```

# Ornith 1.5 35B-A3B NVFP4 / SGLang / DGX Spark

Run the official mixed-precision
[`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)
checkpoint on one NVIDIA DGX Spark through Let's Infer's local
OpenAI-compatible API.

## Recipe

The runtime pins the exact 23.4 GB ModelOpt checkpoint without requantizing it.
Its Qwen3.5-MoE text backbone has 256 experts with eight active per token,
alternates three linear-attention layers with one full-attention layer, and
uses FP8 attention/linear weights, W4A16 NVFP4 MoE and shared-expert weights,
an NVFP4 LM head, FP8 KV cache, and BF16 Mamba state. The native context limit
is clamped to 262,144 tokens.

The candidate reuses the exact SGLang Engine already qualified on DGX Spark.
That Engine includes Qwen3.5-MoE mixed-ModelOpt support, SM121 FlashInfer
kernels, hybrid Mamba scheduling, reasoning/tool parsers, and Let's Infer's
persistent NVMe HiCache backend. The initial production recipe uses four
active requests, chunked prefill, the FlashInfer CuteDSL MoE runner, and no
separate draft checkpoint.

## Persistent cache

SGLang's hierarchical cache writes complete attention KV and hybrid state to
Let's Infer's CRC-checked, atomic, byte-LRU PrefixStore. The durable tier is
bounded to 64 GiB with a seven-day TTL, direct reads, a 1 GiB host staging
tier, and no separate resident PrefixStore tier on unified memory. Incomplete,
corrupt, stale, or identity-mismatched records are misses.

## Benchmark

The schema-7 qualification contract runs short code and prose at C1, C2, and
C4, followed by code at C1, C2, and C4 for 32K, 64K, 128K, and a 260,000-token
prompt. It finishes with an exact-repeat 64K cold/warm TTFT pair. Every cell is
run once in one fresh shared-prefix matrix, and the 262,144-token ceiling is
never exceeded.

After publication:

```bash
letsinfer install ornith-1.5-35b-a3b \
  --runtime sglang--ornith-ai--ornith-1.5-35b-a3b-nvfp4--dgx-spark
letsinfer benchmark ornith-1.5-35b-a3b
```

## Exact sources

- Model revision: `0f0b1b59b879ccde1353e6ebd0fb10c204d4c544` (MIT).
- Engine OCI:
  `ghcr.io/letsinferlabs/engines/sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark@sha256:f70057989f8207323ee499671496fdf8054af3b89ebb69e4a6bd799fca864d74`.
- Engine configuration:
  `sha256:cf79d71ed72d94d9acf35bb8602969c5c159db8abe802ff3399925f9270d8650`.

The runtime source is AGPL-3.0-only. The model and reused Engine retain their
own licenses and provenance.
