FROM docker.io/lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6

ARG SOURCE_DATE_EPOCH=0

COPY engine/base-files.sha256 /tmp/letsinfer-dflash2-base.sha256
RUN cd /sgl-workspace/sglang && sha256sum -c /tmp/letsinfer-dflash2-base.sha256

COPY engine/sglang/python /tmp/letsinfer-dflash2/python
COPY engine/sglang/patches /tmp/letsinfer-dflash2/patches
COPY engine/patched-files.sha256 /tmp/letsinfer-dflash2-patched.sha256
RUN cp -a /tmp/letsinfer-dflash2/python/. /sgl-workspace/sglang/python/ && \
    cd /sgl-workspace/sglang && \
    patch -p1 --fuzz=0 < /tmp/letsinfer-dflash2/patches/draft-model-override.patch && \
    cd /sgl-workspace/sglang && \
    sha256sum -c /tmp/letsinfer-dflash2-patched.sha256 && \
    python3 -m compileall -q \
      python/sglang/srt/configs/model_config.py \
      python/sglang/kernels/ops/speculative/dflash.py \
      python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py \
      python/sglang/srt/models/dflash.py \
      python/sglang/srt/speculative/dflash_utils.py \
      python/sglang/srt/speculative/dflash_worker_v2.py && \
    rm -rf /tmp/letsinfer-dflash2 \
      /tmp/letsinfer-dflash2-base.sha256 \
      /tmp/letsinfer-dflash2-patched.sha256

LABEL org.opencontainers.image.source="https://github.com/letsinferlabs/runtimes" \
      org.opencontainers.image.revision="c14312a66420b75ca9a11bf1817c4db1fa26b097" \
      ai.letsinfer.runtime="qwen3.8-27b/sglang/dgx-spark" \
      ai.letsinfer.feature="dflash2"
