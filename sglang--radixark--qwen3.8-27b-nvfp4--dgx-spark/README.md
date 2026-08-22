# Qwen3.8 27B NVFP4 / SGLang / DGX Spark

## Hugging Face model

[RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4)

`runtime.json` pins the exact model revision. The link identifies the source
repository; it does not replace that immutable pin.

This directory is the complete source form of one Let's Infer runtime
candidate. It binds that checkpoint to one digest-pinned SGLang Engine OCI and
the DGX Spark target. The adapter, Engine source closure, image recipe,
benchmark contract, sealed evidence, and target-specific settings live beside
`runtime.json`; generic gateway, Watchdog, prompt, and benchmark-runner code
stays in Let's Infer core.

After this exact candidate passes qualification and becomes the recommended
DGX Spark choice, you install it with:

```bash
letsinfer install qwen3.8-27b
```

Let's Infer then downloads the runtime pack, this exact model revision, and
the pinned Engine OCI. You do not download or place the weights separately.
