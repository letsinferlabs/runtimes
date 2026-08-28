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

This candidate serves the exact
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
checkpoint as one SGLang tensor-parallel group across two NVIDIA DGX Sparks.
Public version `0.1.0-rc.1` packages the retained local RC.43 execution
subject; the local RC number was an experiment sequence and is not a public
version lineage.

## Production execution contract

- Linux arm64 with one full NVIDIA GB10 (`sm_121`) device on each of two
  nodes.
- A verified 200 Gbit/s ConnectX RDMA link with at least 1500-byte MTU.
  Core seals the interface, HCA, route, and verbs devices. The adapter forces
  NCCL `NET/IB`, rejects socket fallback, and requires IB logs plus RDMA
  counter progress before readiness.
- SGLang TP=2 with task-0 as the main-node endpoint owner.
- Native 262,144-token context, a 600,000-token total pool ceiling, six active
  requests, 32 connections, 0.80 static memory fraction, 97 Mamba slots, and
  PLE embedding offload.
- Built-in NEXTN MTP speculative decoding with three speculative steps, top-k
  one, and four draft tokens; ReplaySSM-spec and decode-mode speculative
  attention are enabled.
- Exact-batch decode CUDA graphs through batch eight, graph padding disabled,
  and prefill CUDA graphs disabled.
- Model-default sampling with the PyTorch sampling backend, temperature-zero
  benchmark requests, and seed 42042.
- Thinking remains enabled. The runtime does not install Tony's
  `enable_thinking=false` chat-template override.
- SGLang's process-local radix prefix cache remains enabled for the declared
  shared-prefix and cold/warm benchmark cells. Persistent inference-state
  restore is not implemented or claimed.

The Engine receives `--allow-auto-truncate`, while Core owns admission at the
declared 262,144-token boundary. Qualification must verify that requests above
that public boundary receive the structured Core rejection rather than being
silently shortened.

## Immutable model and Engine inputs

- Model revision:
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
- SGLang arm64 base manifest:
  `sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4`.
- Base image configuration:
  `sha256:64c58f100438fa5f036bdfbeb3edd3136fb12c5d22d8ae52786c4a701263c55d`.
- Base SGLang source:
  `d91c3682b0b429e4c70df63cd57f819588ce29b0`, with the exact upstream
  overlay recorded in `engine/PROVENANCE.json`.
- Engine OCI manifest:
  `sha256:363c364ee5029e9a49c4735ada372f430fd14f566b4cb2ca2cec70515b786ae4`.
- Engine image configuration:
  `sha256:7be16441d906930f8babaf30ebe8779587845ce7da8503ba7b1d59e271a6d5e4`.
- Normalized Engine execution payload:
  `sha256:8dd2e3e209847ffd75c6960d22cb98b14cca7f43c44ad22189f0fd0f97c4ff43`.

The serving recipe is materially derived from
[tonyd2wild's two-Spark recipe at commit
`65ea3883b8dd80438b58ace56eb7979c52fa6ea6`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/tree/65ea3883b8dd80438b58ace56eb7979c52fa6ea6).
Let's Infer removes its thinking-off override and binds the recipe to Core's
managed TP2 placement and exact ConnectX resources.

## SM121 Engine corrections

The Engine adds only hash-guarded source corrections against the pinned SGLang
base:

- enables the FlashInfer TRT-LLM QSA sparse-decode path after its real SM121
  head-shape guard passes;
- canonicalizes QSA top-k selection order;
- carries the measured SM121 BF16 single-row decode tactics; and
- replaces the HC mix device-atomic accumulation with a deterministic two-way
  split-K reduction.

Candidate CUDA tests cover QSA eager/graph behavior and deterministic HC mix
on SM121. The protocol adapter separately binds the exact two-node resources,
protects distributed arguments, and proves RDMA readiness.

## Complete benchmark contract

The schema-8 contract uses `fresh-context` isolation:

- short code and prose at C1, C2, and C4 with 512 output tokens;
- long code at 32,768, 65,536, 131,072, and 260,000 prompt tokens, each at C1,
  C2, and C4 with 128 output tokens; and
- one unique 64,000-token code prompt run cold and immediately warm with a
  one-token response budget.

Every unrelated context tier receives a fresh process and store. C1/C2/C4
inside one tier retain shared prefix state, and the cold/warm pair must report
a larger warm cache hit. The complete contract is 18 workload cells plus the
two-request TTFT cache pair. A focused local 32K C1 screen is diagnostic only
and cannot qualify or score this release.

## Qualification

Before publication, the exact finalized proposal must prove:

- finite and coherent text, reasoning, tool-call, streaming, and usage output;
- exact Engine-rendered token counts and structured context rejection;
- all 20 declared benchmark cells without shortened output;
- NCCL `NET/IB` selection with no socket fallback, ConnectX HCA/QP evidence,
  and increasing RDMA counters;
- whole-group unavailability on task or link failure and explicit recovery;
- safe unified-memory, pressure, thermal, restart, and OOM behavior; and
- restoration of the previously installed service on every terminal
  benchmark path.

Authors validate source with the repository candidate policy, manifest,
Engine builder, candidate tests, and deterministic pack tools. Independent
qualification uses the finalized pull-request artifact through
`letsinfer benchmark verification run <pull-request-url>`.
