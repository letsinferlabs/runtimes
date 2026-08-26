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

# llama.cpp / Qwen3 0.6B / iOS Apple GPU

This unqualified candidate runs the exact [Qwen3 0.6B GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) revision through llama.cpp 0.3.0 compiled into the signed Let's Infer iOS application.

It is the broad GGUF compatibility lane for current iPhone and iPad hardware. The application must remain foregrounded through Guided Access or supervised Single App Mode; no performance or availability claim exists before physical qualification.

Reproduce the source checks with:

```bash
python3 tools/candidate_policy.py audit --candidate llamacpp--qwen--qwen3-0.6b-gguf--ios-apple-gpu --mode build-native-engine
python3 tools/generate_manifest.py --validate-only
```
