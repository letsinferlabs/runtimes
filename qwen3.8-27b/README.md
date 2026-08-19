# Qwen3.8 27B runtimes

This directory contains independently versioned Let's Infer runtimes for
Qwen3.8 27B.

The initial target is a qualified release candidate for NVIDIA DGX Spark:

```text
sglang/targets/dgx-spark/
```

The target pins the exact model revision, arm64 image digest, 1M-token YaRN
recipe, EAGLE speculative decoding settings, serving limits, and hardware
contract. It accepts up to 128 connections; SGLang schedules up to ten active
requests and queues excess accepted work.

The production recipe uses SGLang's in-process Radix cache. Let's Infer's Rust
prefix store was also measured in a cache-compatible 262K, non-speculative
lane: its warm 32,768-token lookup was 4.6% faster, while cold performance was
equivalent. The pinned SGLang revision cannot combine its external HiCache
backend with this runtime's 1M-token EAGLE configuration, and neither external
backend restored cache hits across a fresh SGLang process. Radix therefore
preserves the qualified long-context and speculative-decoding behavior.

The structured `benchmark.json` contains the sealed C1 results for 32K, 64K,
128K, and 256K prompts, including Watchdog telemetry. Reproduce those workloads
through the installed runtime and generic core runner:

```bash
letsinfer benchmark qwen3.8-27b --c1
```
