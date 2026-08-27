FROM docker.io/lmsysorg/sglang@sha256:7d81a16a2ebbe104be37263b49f1bfc0a6a95c1d57e500875dbe7f7c41301e45 AS letsinfer-engine-inventory-build

COPY --from=letsinfer-tools tools/engine_sbom.py /tmp/letsinfer-engine-sbom.py
RUN python3 /tmp/letsinfer-engine-sbom.py snapshot \
      --output /tmp/engine-inventory.json

FROM scratch AS letsinfer-engine-inventory
COPY --from=letsinfer-engine-inventory-build \
  /tmp/engine-inventory.json /engine-inventory.json

FROM ghcr.io/pyo3/maturin@sha256:b6c8b59a0170b77eb31a35b56034abd39972483ad0ebfff344deaa42a85f3bd3 AS letsinfer-prefix-store-build

WORKDIR /build/letsinfer-prefix-store
COPY adapter/letsinfer-prefix-store/ ./
RUN cargo build --release --locked --features python \
    && strip --strip-debug target/release/libletsinfer_prefix_store.so \
    && install -D -m 0555 target/release/libletsinfer_prefix_store.so \
      /opt/letsinfer/lib/letsinfer_prefix_store.so

FROM docker.io/lmsysorg/sglang@sha256:7d81a16a2ebbe104be37263b49f1bfc0a6a95c1d57e500875dbe7f7c41301e45

ARG SOURCE_DATE_EPOCH=0

ADD --checksum=sha256:74caf5e34b66d9f3a429af689c1c7128daba1d8208df60e81106b115c00d6fca \
  https://files.pythonhosted.org/packages/ab/a8/cdf8b3e4c98132f965f88c2313a4b493266832ad47fb52f23d14d4f86bb5/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
  /tmp/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
RUN python3 -m pip install --no-cache-dir --no-deps \
      /tmp/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl && \
    python3 -c 'import triton; assert triton.__version__ == "3.6.0"' && \
    rm -f /tmp/triton-3.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

COPY --from=letsinfer-prefix-store-build \
  /opt/letsinfer/lib/letsinfer_prefix_store.so \
  /tmp/letsinfer_prefix_store.so
RUN PYTHON_PLATLIB="$(python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')" && \
    test -n "${PYTHON_PLATLIB}" && \
    install -D -m 0555 /tmp/letsinfer_prefix_store.so \
      "${PYTHON_PLATLIB}/letsinfer_prefix_store.so" && \
    rm -f /tmp/letsinfer_prefix_store.so
COPY engine/sglang/python /tmp/letsinfer-sglang/python
RUN cp -a /tmp/letsinfer-sglang/python/. /sgl-workspace/sglang/python/ && \
    python3 -c 'import letsinfer_prefix_store' && \
    python3 -m compileall --invalidation-mode checked-hash -q \
      python/sglang/kernels/ops/speculative/dflash.py \
      python/sglang/srt/models/dflash.py \
      python/sglang/srt/speculative/dflash_worker_v2.py && \
    rm -rf /tmp/letsinfer-sglang

COPY adapter/engine_frontend.py /opt/letsinfer/lib/engine_frontend.py
COPY adapter/engine-adapter /opt/letsinfer/bin/engine-adapter
RUN chmod 0555 /opt/letsinfer/bin/engine-adapter && \
    chmod 0444 /opt/letsinfer/lib/engine_frontend.py

LABEL org.opencontainers.image.source="https://github.com/letsinferlabs/runtimes" \
      ai.letsinfer.runtime="sglang--radixark--qwen3.8-27b-nvfp4--rtx-5090" \
      ai.letsinfer.feature="dflash2-prefix-value,letsinfer-nvme-hicache" \
      ai.letsinfer.cuda="12.9.2" \
      ai.letsinfer.triton="3.6.0" \
      ai.letsinfer.build.reproducibility="source-date-epoch-v1" \
      ai.letsinfer.sglang.source="1cf2b8c54d81802abc15dcf23a29b9cc687bc01e" \
      ai.letsinfer.flashinfer.profile="runtime-autotune-sm120-v1" \
      ai.letsinfer.dflash.selector="prefix-value:eda3392ea66597f113c76cf1d9113e9bba290590e31a75ceee537e4269df2dae"
