# Qwen3.8 27B NVFP4 / SGLang / DGX Spark

## Hugging Face artifacts

- Primary model: [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)
- DFlash drafter: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)

`runtime.json` pins both exact revisions. These links identify the source
repositories; they do not replace those immutable pins.

This directory is the complete source form of one Let's Infer runtime
candidate. It binds that checkpoint to one digest-pinned SGLang Engine OCI and
the DGX Spark target. The adapter, Engine source closure, image recipe,
benchmark contract, qualification evidence, and target-specific settings live
beside `runtime.json`; generic gateway, Watchdog, prompt, and benchmark-runner
code stays in Let's Infer core.

After this exact candidate passes qualification and becomes the recommended
DGX Spark choice, you install it with:

```bash
letsinfer install qwen3.8-27b
```

Let's Infer then downloads the runtime pack, this exact model revision, and
the pinned Engine OCI. You do not download or place the weights separately.

## Reproduce this

After installing the qualified runtime, run its complete canonical C1
code-and-prose context set through the unified gateway:

```bash
letsinfer benchmark qwen3.8-27b --c1
```

Let's Infer materializes the standard prompts with the exact Engine tokenizer,
starts each cell with an isolated lifecycle and cache, records Watchdog
telemetry, and validates the generated `benchmark.json`.
