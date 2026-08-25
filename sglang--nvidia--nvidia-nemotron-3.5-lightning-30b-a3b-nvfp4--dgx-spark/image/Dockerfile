FROM docker.io/lmsysorg/sglang@sha256:ee63304703432429d55cf2e4579b70c5a334ed4cc49194fb5290c077fcbbd4a0 AS letsinfer-engine-inventory-build

COPY --from=letsinfer-tools tools/engine_sbom.py /tmp/letsinfer-engine-sbom.py
RUN python3 /tmp/letsinfer-engine-sbom.py snapshot \
      --output /tmp/engine-inventory.json

FROM scratch AS letsinfer-engine-inventory
COPY --from=letsinfer-engine-inventory-build \
  /tmp/engine-inventory.json /engine-inventory.json

FROM docker.io/lmsysorg/sglang@sha256:ee63304703432429d55cf2e4579b70c5a334ed4cc49194fb5290c077fcbbd4a0

ARG SOURCE_DATE_EPOCH=0

COPY adapter/engine_frontend.py /opt/letsinfer/lib/engine_frontend.py
COPY adapter/engine-adapter /opt/letsinfer/bin/engine-adapter
RUN chmod 0555 /opt/letsinfer/bin/engine-adapter && \
    chmod 0444 /opt/letsinfer/lib/engine_frontend.py && \
    python3 -m compileall --invalidation-mode checked-hash -q \
      /opt/letsinfer/lib/engine_frontend.py

LABEL org.opencontainers.image.source="https://github.com/letsinferlabs/runtimes" \
      ai.letsinfer.runtime="sglang--nvidia--nvidia-nemotron-3.5-lightning-30b-a3b-nvfp4--dgx-spark" \
      ai.letsinfer.feature="nemotron-3.5-lightning,dspark,engine-protocol-v2" \
      ai.letsinfer.build.reproducibility="source-date-epoch-v1" \
      ai.letsinfer.sglang.source="d59c1ddf70ee17fcc41c053ed38bd60bc6cc28cc"
