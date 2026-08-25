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

# Ornith 1.5 35B-A3B NVFP4 / vLLM B12X / DGX Spark

Serve the exact
[`ornith-ai/Ornith-1.5-35B-A3B-NVFP4`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4)
checkpoint on one NVIDIA DGX Spark through a digest-pinned vLLM/B12X Engine.

## Recipe

This runtime ports the
[`sparkbench` B12X/EUGR recipe](https://github.com/shawnmarck/sparkbench/blob/253c5e44d83c79db18daf0c0e40d73c27940e682/recipes/ornith-ai-ornith-1-5-35b-a3b-nvfp4-b12x-eugr.yaml)
and the upstream
[`MiaAI-Lab` DGX Spark implementation](https://github.com/MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark/tree/9fd7fe7896230550fad1c773e85ad765292cfde5).
It preserves their text-only B12X W4A16 path, 8,192-token chunked prefill,
in-checkpoint MTP with one speculative token, 24 sequence slots, 0.85 memory
fraction, FP8 KV cache, and native 262,144-token context ceiling.

The Engine starts from the `nightly-20260815` arm64 image that was current for
the August 20 SparkBench measurements. It verifies and replaces exactly two
B12X files with Mia's MIT-licensed CUDA-graph fixes: the W4A16 output-drain
metadata correction and the stable grow-only route-packing workspace.

SparkBench reports 70.8 tok/s at 4K, 36.5 at 50K, and 27.2 at 100K under its
PBM method, while Mia reports 86.3 tok/s for its short single-stream request.
Those are upstream results with different prompt/output methods; the Let's
Infer benchmark below is the release authority for this runtime.

## Persistent cache

The Engine includes Let's Infer's host-copy-only vLLM connector and atomic
CRC-checked PrefixStore. It binds replay to the exact model, tokenizer, cache
dtype, hybrid KV/SSM layout, Engine configuration, and token prefix. The 64
GiB NVMe tier is byte-LRU bounded with a seven-day TTL; incomplete, stale,
corrupt, or incompatible records are complete misses. The canonical 64K TTFT
pair captures only its final chunk boundary so the throughput matrix does not
become a synchronous NVMe-write benchmark.

## Benchmark

The schema-7 contract runs short code and prose at C1/C2/C4, code at
32K/64K/128K/260K at C1/C2/C4, and one exact-repeat 64K cold/warm TTFT pair.
Each cell runs once and the serving context is clamped to 262,144 tokens.

After publication:

```bash
letsinfer install ornith-1.5-35b-a3b \
  --runtime vllm-b12x--ornith-ai--ornith-1.5-35b-a3b-nvfp4--dgx-spark
letsinfer benchmark ornith-1.5-35b-a3b
```

## Exact sources

- Model revision: `0f0b1b59b879ccde1353e6ebd0fb10c204d4c544` (MIT).
- EUGR image: `docker.io/eugr/spark-vllm-b12x@sha256:25fe41c2e85993b4e0534b3c72f68bf327c5d2726fbe1640cf6b220715d3b0e3`.
- vLLM fork commit: `ad848fc4141f201489db18d5453c50b312245a0a`.
- B12X commit: `a63f07e90fd449b693cafbe6aef1a73309595bf7`.
- FlashInfer commit: `8044d94bf9acc5369857baf88d28906bb32bf264`.

The Let's Infer integration is AGPL-3.0-only. Vendored overlays and upstream
Engine components retain their own licenses and notices in `third_party/`.
