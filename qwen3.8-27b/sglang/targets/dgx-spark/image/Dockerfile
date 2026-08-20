FROM docker.io/lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6

ARG SOURCE_DATE_EPOCH=0

COPY engine/base-files.sha256 /tmp/letsinfer-dflash2-base.sha256
RUN cd /sgl-workspace/sglang && sha256sum -c /tmp/letsinfer-dflash2-base.sha256

COPY engine/sglang/python /tmp/letsinfer-sglang/python
RUN cp -a /tmp/letsinfer-sglang/python/. /sgl-workspace/sglang/python/ && \
    chmod 0644 \
      /sgl-workspace/sglang/python/sglang/srt/model_executor/runner/flashinfer_profiles/0.6.18-sm121.json && \
    python3 -m compileall -q \
      python/sglang/srt/configs/model_config.py \
      python/sglang/kernels/ops/speculative/dflash.py \
      python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py \
      python/sglang/srt/model_executor/runner/flashinfer_autotune.py \
      python/sglang/srt/models/dflash.py \
      python/sglang/srt/speculative/dflash_utils.py \
      python/sglang/srt/speculative/dflash_worker_v2.py && \
    rm -rf /tmp/letsinfer-sglang /tmp/letsinfer-dflash2-base.sha256

LABEL org.opencontainers.image.source="https://github.com/letsinferlabs/runtimes" \
      org.opencontainers.image.revision="c14312a66420b75ca9a11bf1817c4db1fa26b097" \
      ai.letsinfer.runtime="qwen3.8-27b/sglang/dgx-spark" \
      ai.letsinfer.feature="dflash2" \
      ai.letsinfer.flashinfer.profile="0.6.18-sm121:18f9e07ecaac1d725347d1e01d9b8adff68f02084e2e790a9cde89a29e3e8bc0"
