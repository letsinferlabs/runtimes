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
> letsinfer install ling-3.0-flash
> ```

# Ling 3.0 Flash INT4 / SGLang / DGX Spark

Run inclusionAI Ling 3.0 Flash on one NVIDIA DGX Spark through Let's Infer's
stable local OpenAI-compatible API.

## Features

- **One-command installation** -- Let's Infer downloads the exact target and
  DSpark checkpoints, runtime pack, and Engine OCI, then starts the API.
- **256K context** -- serves the checkpoint's native 262,144-token window.
- **DSpark speculative decoding** -- the external 1.36B Ling draft proposes
  eight-token blocks through SGLang's ReplaySSM verification path.
- **Spark-safe memory profile** -- 75% static allocation, FP8 KV, a 32-entry
  Mamba cache, and a 10 GiB host-availability safety floor.
- **Measured concurrency envelope** -- up to six active requests behind 128
  gateway connections, with Let's Infer admission and queueing.
- **Process-local prefix reuse** -- SGLang radix caching is enabled, but this
  runtime does not claim persistent inference-state restoration.

## Hugging Face artifacts

- Primary model: [inclusionAI/Ling-3.0-flash-int4](https://huggingface.co/inclusionAI/Ling-3.0-flash-int4)
- DSpark drafter: [inclusionAI/Ling-3.0-flash-dspark](https://huggingface.co/inclusionAI/Ling-3.0-flash-dspark)

`runtime.json` pins both repositories to exact revisions. Let's Infer acquires
both automatically; you do not preinstall or move model files.

## Recipe lineage

The base launch and parser configuration follows inclusionAI's official Ling
3.0 Flash SGLang DGX Spark cookbook. The external DSpark draft, ReplaySSM
settings, FP8 KV cache, memory profile, and six-request envelope derive from
MiaAI-Lab's DGX Spark integration at commit
`ca840cb8d032353e24648aeee06312b0938348f6`.

The Engine uses the immutable official ARM64 SGLang image built from commit
`0e5e40d8f1460976cd7190ae479c210f0642c120`. Exact source identities and
license boundaries are recorded in `engine/PROVENANCE.json` and
`THIRD_PARTY.md`.

## Reproduce this

After installing the qualified runtime, run its complete cache-aware benchmark:

```bash
letsinfer benchmark ling-3.0-flash
```

The schema-7 contract runs short code and prose at C1/C2/C4, code at C1/C2/C4
across 32K, 64K, 128K, and 256K, then an exact 64K cold/warm TTFT pair.
