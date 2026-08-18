# DeepSeek V4 Flash runtimes

This repository contains independently versioned Let's Infer runtimes for
DeepSeek V4 Flash.

The first target is the complete DwarfStar runtime for NVIDIA DGX Spark:

```text
dwarfstar/targets/dgx-spark/
```

Each target owns its engine source, kernels, image recipe, immutable release
manifest, and structured benchmark record. Model weights, container layers,
private evidence, generated prompts, caches, and credentials are not stored in
this repository.
