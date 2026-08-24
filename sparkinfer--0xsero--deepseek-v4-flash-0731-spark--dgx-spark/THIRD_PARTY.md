# Third-party notices

The runtime integration is AGPL-3.0-only. The following incorporated or
patched upstream components retain their original licenses:

- MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark, commit
  `76c51c7defffd01025c75daaf241ea323dda4734`: MIT.
- local-inference-lab/b12x (SparkInfer), base commit
  `272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`: Apache-2.0.
- vLLM source contained in the pinned base image: Apache-2.0.
- xgrammar 0.2.4 arm64 wheel: Apache-2.0.

The model checkpoint and the pinned Engine base are acquired separately and
remain subject to their upstream license terms. Exact identities and the
applied fix list are recorded in `engine/PROVENANCE.json`.
