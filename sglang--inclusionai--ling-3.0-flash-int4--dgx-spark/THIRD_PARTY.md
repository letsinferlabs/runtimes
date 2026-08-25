# Third-party notices

The Let's Infer integration is AGPL-3.0-only. Incorporated or referenced
upstream components retain their original terms:

- `sgl-project/sglang`, commit
  `0e5e40d8f1460976cd7190ae479c210f0642c120`: Apache-2.0.
- `inclusionAI/ling-cookbook`, commit
  `7873c82251e9994289789b3e3cde6e2c3e45db06`: official Ling SGLang DGX
  Spark recipe reference. No cookbook source bytes are copied into this
  candidate; the repository does not declare a license.
- `MiaAI-Lab/Ling-3.0-Flash-SGLang-DSpark-DGX-Spark`, commit
  `ca840cb8d032353e24648aeee06312b0938348f6`: DGX Spark DSpark recipe
  reference. No repository source bytes are copied into this candidate; the
  repository does not declare a license.

The target checkpoint is published under MIT terms. The separately acquired
DSpark checkpoint declares its license as `other`; operators remain
responsible for its upstream terms. Exact image and checkpoint identities are
recorded in `engine/PROVENANCE.json` and `runtime.json`.
