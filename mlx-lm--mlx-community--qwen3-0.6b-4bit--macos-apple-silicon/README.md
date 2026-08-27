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
> letsinfer model install qwen3-0.6b
> ```

# MLX-LM / Qwen3 0.6B / macOS Apple Silicon

This unqualified candidate runs the exact [MLX Qwen3 0.6B 4-bit](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit) revision using MLX-LM 0.31.3 and a hash-locked, self-contained CPython environment.

MLX-LM is the Apple-Silicon-specialized lane. Its server has continuous batching and an in-memory prompt cache, but no performance claim or recommendation is made until it is compared with llama.cpp on the same Mac, model workload, context, and commit.

Reproduce the source checks with:

```bash
python3 tools/candidate_policy.py audit --candidate mlx-lm--mlx-community--qwen3-0.6b-4bit--macos-apple-silicon --mode build-native-engine
python3 tools/generate_manifest.py --validate-only
```
