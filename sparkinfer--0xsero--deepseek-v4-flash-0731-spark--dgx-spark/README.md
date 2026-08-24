# DeepSeek V4 Flash EXL3 / SparkInfer / DGX Spark

Run the `0xSero/deepseek-v4-flash-0731-spark` 3.0-bpw EXL3 checkpoint on
one NVIDIA DGX Spark through Let's Infer's local OpenAI-compatible API.

## Recipe

This candidate turns MiaAI-Lab's single-Spark recipe into an immutable Engine
OCI and runtime pack. It preserves the REAP-K216 quant, losslessly coalesces
the rank-sliced checkpoint to TP1, builds the K176 DSpark drafter, and serves
with SparkInfer's SM121 kernels. Native 432-byte NVFP4 MLA cache records,
DSpark K6, prefix caching, a 4,096-token prefill workspace, and CUDA graph
size 7 are fixed in `runtime.json`.

The runtime includes Mia's native-NVFP4 dual-cache and DSpark writer fixes,
her cross-device coalescer and default-thinking integration, plus the
SparkInfer route-histogram and inactive-route backports. Every patched base
file is SHA-256 checked before patching. The xgrammar tool-calling repair is
an exact arm64 wheel pinned by its release hash instead of a boot-time upgrade.

## Boundaries

The context ceiling is exactly 262,144 tokens. The production recipe admits
one active request; the gateway queues additional connections. Let’s Infer
persists the downloaded source checkpoint, coalesced TP1 checkpoint, K176
draft, and compilation caches. vLLM prefix-cache entries are intentionally
declared non-persistent because this Engine does not implement Let's Infer's
portable persistent-prefix connector.

## Reproduce

After this candidate is independently verified and published:

```bash
letsinfer install deepseek-v4-flash \
  --runtime sparkinfer--0xsero--deepseek-v4-flash-0731-spark--dgx-spark
letsinfer benchmark deepseek-v4-flash
```

The schema-6 contract first runs fixed short-code and short-prose workloads at
C1, C2, and C4 with 512-token completions. It then runs the canonical code
workload once at C1, C2, and C4 for 32K, 64K, 128K, and a 260,000-token prompt
under the 262,144-token cap: 18 cells total. One fresh process/store
serves the complete matrix and intentionally retains prefix state between
cells. Each long context runs C1, C2, then C4 while all four distinct streams
share the complete ledger prefix, so the matrix measures real prefix-cache
reuse without repeating four cold long prefills. Benchmark claims are added
only from a complete immutable evidence directory bound to the measured
commit.

## Upstream sources

- Recipe: `MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark` at
  `76c51c7defffd01025c75daaf241ea323dda4734` (MIT).
- Model: [0xSero/deepseek-v4-flash-0731-spark](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) at
  `22f28d32b9b29b4352eaa380ff8c2c170b2847ab`.
- Engine base: `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer` at
  platform digest `sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`.
- Kernel source: `local-inference-lab/b12x` base
  `272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f` (Apache-2.0).

See `THIRD_PARTY.md` and `engine/PROVENANCE.json` for the source and license
closure. Model weights and the upstream base image retain their own licenses.
