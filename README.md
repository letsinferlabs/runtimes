# Let's Infer runtimes

This repository contains independently versioned model runtimes for Let's
Infer. Runtime implementations are organized by model, engine, and hardware
target:

```text
<model>/<engine>/targets/<target>/
```

Each target directory is self-contained and may own its engine source, kernels,
image recipe, immutable release manifest, and structured benchmark record.
Shared repository tooling validates the same runtime contract for every model.

Model weights, container layers, credentials, private evidence, generated
prompts, caches, and machine-specific state are not stored here.

## Available runtimes

- `deepseek-v4-flash/dwarfstar/targets/dgx-spark`
- `qwen3.8-27b/sglang/targets/dgx-spark`

## License

AGPL-3.0-only. See `LICENSE`.
