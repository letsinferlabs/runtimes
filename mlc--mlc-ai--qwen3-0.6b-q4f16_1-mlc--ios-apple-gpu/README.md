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

# MLC LLM / Qwen3 0.6B / iOS Apple GPU

This unqualified candidate uses [MLC LLM](https://github.com/mlc-ai/mlc-llm) to compile the exact [Qwen3 0.6B MLC snapshot](https://huggingface.co/mlc-ai/Qwen3-0.6B-q4f16_1-MLC) for Metal on iPhone and iPad. It is the compiled-model lane intended for Apple A-series devices where MLX is unavailable or unsuitable.

The model library and MLC runtime are embedded executable inputs of the signed iOS application. Model weights remain an exact separately downloaded Hugging Face snapshot. No performance, background-service, or device-family claim exists until physical kiosk-mode qualification is complete.

The build recipe requires both an exact recursive MLC LLM checkout and an
exact Git/LFS checkout of model revision
`8c14ce481d4c692769976ad52afea453a102df19`. It verifies the source commit,
submodule state, model commit, and every large model object before compiling;
the model weights are used from a read-only cache and are not bundled.

Reproduce the source checks with:

```bash
python3 tools/candidate_policy.py audit --candidate mlc--mlc-ai--qwen3-0.6b-q4f16_1-mlc--ios-apple-gpu --mode build-native-engine
python3 tools/generate_manifest.py --validate-only
```
