# Third-party notices

This candidate derives its DGX Spark recipe and the two complete B12X overlay
files from:

- `MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark` commit
  `9fd7fe7896230550fad1c773e85ad765292cfde5` (MIT, Victor Cruz);
- `shawnmarck/sparkbench` commit
  `253c5e44d83c79db18daf0c0e40d73c27940e682` (MIT); and
- the immutable `eugr/spark-vllm-b12x` arm64 image manifest
  `sha256:25fe41c2e85993b4e0534b3c72f68bf327c5d2726fbe1640cf6b220715d3b0e3`
  (EUGR integration under MIT).

The base image records:

- `local-inference-lab/vllm` commit
  `ad848fc4141f201489db18d5453c50b312245a0a` (Apache-2.0);
- `lukealonso/b12x` commit
  `a63f07e90fd449b693cafbe6aef1a73309595bf7` (Apache-2.0); and
- FlashInfer commit `8044d94bf9acc5369857baf88d28906bb32bf264`.

The root `LICENSE` covers the Let's Infer adapter, persistent-cache connector,
PrefixStore integration, and candidate metadata. The adjacent license files
retain the terms of the incorporated upstream work. Model weights remain in
their Hugging Face repository and are not distributed by this candidate.
