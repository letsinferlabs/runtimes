FROM docker.io/lmsysorg/sglang@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4 AS letsinfer-engine-inventory-build

COPY --from=letsinfer-tools tools/engine_sbom.py /tmp/letsinfer-engine-sbom.py
RUN python3 /tmp/letsinfer-engine-sbom.py snapshot \
      --output /tmp/engine-inventory.json

FROM scratch AS letsinfer-engine-inventory
COPY --from=letsinfer-engine-inventory-build \
  /tmp/engine-inventory.json /engine-inventory.json

FROM docker.io/lmsysorg/sglang@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4

ARG SOURCE_DATE_EPOCH=0

COPY adapter/engine_frontend.py /opt/letsinfer/lib/engine_frontend.py
COPY adapter/engine-adapter /opt/letsinfer/bin/engine-adapter
RUN chmod 0555 /opt/letsinfer/bin/engine-adapter && \
    chmod 0444 /opt/letsinfer/lib/engine_frontend.py && \
    python3 -m compileall --invalidation-mode checked-hash -q \
      /opt/letsinfer/lib/engine_frontend.py \
      /opt/letsinfer/bin/engine-adapter

LABEL org.opencontainers.image.source="https://github.com/letsinferlabs/runtimes" \
      ai.letsinfer.runtime="sglang--radixark--qwen3.8-flash-next-nvfp4--dgx-spark-connectx-2" \
      ai.letsinfer.feature="qwen3.8-flash-next,tp2,connectx-rdma,engine-protocol-v2" \
      ai.letsinfer.build.reproducibility="source-date-epoch-v1" \
      ai.letsinfer.sglang.source="d91c3682b0b429e4c70df63cd57f819588ce29b0" \
      ai.letsinfer.sglang.overlay="3ea3a37a1,12070370f"
