# Qwen3.8 27B runtimes

This directory contains independently versioned Let's Infer runtimes for
Qwen3.8 27B.

The initial target is a release candidate for NVIDIA DGX Spark:

```text
sglang/targets/dgx-spark/
```

The target pins the exact target and DFlash 2 draft revisions, arm64 image
identity, 1M-token YaRN recipe, serving limits, and hardware contract. The
runtime-owned SGLang overlay adds DFlash 2 to the optimized Spark image, keeps
target-only configuration out of the draft checkpoint, and runs selector
logits through the target's quantized output head. It accepts up to 128
connections; SGLang schedules up to ten active requests and queues excess
accepted work.

The candidate uses SGLang's in-process Radix cache. Persistent external cache
integration remains a separate qualification item and does not block DFlash 2
validation.

The candidate intentionally has no `benchmark.json`: benchmark evidence belongs
to the final image and serving recipe and will be added only after qualification.
After qualification, the generic core runner owns workload generation and execution:

```bash
letsinfer benchmark qwen3.8-27b --c1
```
