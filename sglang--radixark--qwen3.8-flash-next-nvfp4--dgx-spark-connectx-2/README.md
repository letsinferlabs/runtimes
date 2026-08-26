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
> letsinfer model install qwen3.8-flash-next
> ```

# Qwen3.8 Flash Next NVFP4 / SGLang / two DGX Sparks

This unqualified candidate serves the exact
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
checkpoint as one tensor-parallel group across two NVIDIA DGX Sparks.

## Execution contract

- Linux arm64 and one full GB10 (`sm_121`) device per node.
- Two nodes connected by a verified 200 Gbit/s ConnectX RDMA link with at
  least 1500-byte MTU.
- SGLang TP=2 with task-0 as the main-node endpoint owner.
- NCCL is forced to its internal IB transport and the Engine consumes only the
  interface and HCA sealed by Core. Socket fallback cannot satisfy readiness.
- The initial correctness recipe uses a 65,536-token ceiling, four active
  requests, no speculative decoding, and no persistent inference-state cache.

The context, concurrency, CUDA graph, MTP, and cache settings are deliberately
conservative until the exact two-Spark load and RDMA evidence pass. Increasing
them changes the qualification subject and must be measured as one new recipe.

## Immutable inputs

- Model revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- SGLang arm64 base manifest:
  `sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4`.
- Base SGLang source: `d91c3682b0b429e4c70df63cd57f819588ce29b0`
  plus the Qwen3.8 Flash Next overlay recorded in `engine/PROVENANCE.json`.

The model card currently describes the checkpoint as a candidate release and
delegates its terms to the Qwen Community License 1.0. See `THIRD_PARTY.md`.

## Qualification

Before publication, the exact finalized proposal must prove:

- finite and coherent text, reasoning, tool-call, streaming, and usage output;
- exact Engine-rendered token counts and structured context rejection;
- NCCL `NET/IB` selection with no socket fallback, increasing RDMA counters,
  and GPUDirect RDMA evidence;
- whole-group unavailability on task or link failure and explicit recovery;
- safe unified-memory, pressure, thermal, restart, and OOM behavior; and
- the complete declared code/prose context and concurrency matrix.

Authors validate source locally with the repository candidate-policy,
manifest, Engine builder, tests, and deterministic pack tools. Independent
qualification uses the finalized pull-request artifact through
`letsinfer benchmark verification run <pull-request-url>`.
