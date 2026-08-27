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

# llama.cpp / Qwen3 0.6B / macOS Apple Silicon

This unqualified candidate runs the exact [Qwen3 0.6B GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) revision through the official llama.cpp 0.3.0 macOS arm64 archive and a runtime-owned Engine protocol 2 adapter.

It uses Metal through llama.cpp, keeps the native backend on loopback, and exposes only the authenticated TLS Engine frontend on the allocated node port. The candidate has no transferred benchmark evidence and cannot be recommended before physical Apple Silicon qualification.

Reproduce the source checks with:

```bash
python3 tools/candidate_policy.py audit --candidate llamacpp--qwen--qwen3-0.6b-gguf--macos-apple-silicon --mode build-native-engine
python3 tools/generate_manifest.py --validate-only
```
