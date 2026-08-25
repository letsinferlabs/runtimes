FROM docker.io/lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1 AS letsinfer-engine-inventory-build

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

FROM docker.io/lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1

ARG SOURCE_DATE_EPOCH=0

COPY --from=letsinfer-prefix-store-build \
  /opt/letsinfer/lib/letsinfer_prefix_store.so \
  /usr/local/lib/python3.12/dist-packages/letsinfer_prefix_store.so
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
      org.opencontainers.image.revision="ae209b30d7a86f16ff4adf5765abbdf8a38c4d51" \
      ai.letsinfer.runtime="sglang--radixark--qwen3.8-27b-nvfp4--dgx-spark" \
      ai.letsinfer.feature="dflash2-prefix-value,letsinfer-nvme-hicache" \
      ai.letsinfer.build.reproducibility="source-date-epoch-v1" \
      ai.letsinfer.sglang.source="1cf2b8c54d81802abc15dcf23a29b9cc687bc01e" \
      ai.letsinfer.flashinfer.profile="0.6.18-sm121:ea30460e95be5d3c14186e58048f8dc14b79a3416f73c073b6501c36ae8eae98" \
      ai.letsinfer.dflash.selector="prefix-value:eda3392ea66597f113c76cf1d9113e9bba290590e31a75ceee537e4269df2dae"
