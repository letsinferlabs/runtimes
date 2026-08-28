# Third-party provenance

This candidate downloads model weights separately from its runtime pack and
does not redistribute them.

- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
  is pinned at `7b719225242aacd3dbd3f9407468c2ee9a9d2594`. Its model card
  delegates terms to the source model.
- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
  is distributed under the Qwen Community License 1.0. Operators must review
  those terms for their use, especially third-party hosted services.
- [SGLang](https://github.com/sgl-project/sglang) is Apache-2.0 licensed. The
  complete upstream license is retained at `engine/sglang/LICENSE`.
- The pinned day-zero SGLang image incorporates the exact
  [Qiaolin-Yu/sglang-qwen-next](https://github.com/Qiaolin-Yu/sglang-qwen-next)
  overlay identified in `engine/PROVENANCE.json`; its SGLang source remains
  covered by the retained upstream license and image inventory.
- The MTP4, CUDA-graph, memory, PLE, Mamba-slot, and NCCL serving recipe is
  materially derived from
  [tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark at
  `65ea3883b8dd80438b58ace56eb7979c52fa6ea6`](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/tree/65ea3883b8dd80438b58ace56eb7979c52fa6ea6).
  No file from that repository is copied into this candidate; the source here
  retains only the attributed serving configuration adapted to Let's Infer's
  managed placement and thinking-enabled benchmark contract.
- The immutable SGLang CUDA 13 arm64 base carries its own NVIDIA CUDA, cuDNN,
  NCCL, and operating-system notices and package inventory. The trusted
  finalizer preserves that inventory in the Engine SBOM.

The candidate source and Let's Infer adapter are AGPL-3.0-only as declared by
the candidate `LICENSE` and `release.json`.
