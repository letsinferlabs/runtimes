# Qwen3.8 27B runtimes

This directory contains independently versioned Let's Infer runtimes for
Qwen3.8 27B.

The initial target is an SGLang candidate for NVIDIA DGX Spark:

```text
sglang/targets/dgx-spark/
```

The candidate pins its model revision, arm64 image, serving recipe, and target
contract. It remains unqualified until the target-specific installation,
performance, cache, restart, pressure, and crash gates are complete.
