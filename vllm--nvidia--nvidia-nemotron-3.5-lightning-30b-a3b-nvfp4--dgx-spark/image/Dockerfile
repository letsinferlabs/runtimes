FROM docker.io/vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967 AS letsinfer-engine-inventory-build

COPY --from=letsinfer-tools tools/engine_sbom.py /tmp/letsinfer-engine-sbom.py
RUN python3 /tmp/letsinfer-engine-sbom.py snapshot \
      --output /tmp/engine-inventory.json

FROM scratch AS letsinfer-engine-inventory
COPY --from=letsinfer-engine-inventory-build \
  /tmp/engine-inventory.json /engine-inventory.json

FROM docker.io/vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967

ARG SOURCE_DATE_EPOCH=0

COPY adapter/engine_frontend.py /opt/letsinfer/lib/engine_frontend.py
COPY adapter/engine-adapter /opt/letsinfer/bin/engine-adapter
RUN chmod 0555 /opt/letsinfer/bin/engine-adapter && \
    chmod 0444 /opt/letsinfer/lib/engine_frontend.py && \
    python3 -m compileall --invalidation-mode checked-hash -q \
      /opt/letsinfer/lib/engine_frontend.py

LABEL org.opencontainers.image.source="https://github.com/letsinferlabs/runtimes" \
      ai.letsinfer.runtime="vllm--nvidia--nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4--dgx-spark" \
      ai.letsinfer.feature="nemotron-3.5-lightning,dspark,engine-protocol-v2" \
      ai.letsinfer.build.reproducibility="source-date-epoch-v1" \
      ai.letsinfer.vllm.source="6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"
