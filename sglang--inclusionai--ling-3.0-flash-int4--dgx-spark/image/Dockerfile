FROM docker.io/lmsysorg/sglang@sha256:fc960102f1d2581b7dfa44885f6afcdfb107e84782e318cf33bbb2f219d0bd16 AS letsinfer-engine-inventory-build

COPY --from=letsinfer-tools tools/engine_sbom.py /tmp/letsinfer-engine-sbom.py
RUN python3 /tmp/letsinfer-engine-sbom.py snapshot \
      --output /tmp/engine-inventory.json

FROM scratch AS letsinfer-engine-inventory
COPY --from=letsinfer-engine-inventory-build \
  /tmp/engine-inventory.json /engine-inventory.json

FROM docker.io/lmsysorg/sglang@sha256:fc960102f1d2581b7dfa44885f6afcdfb107e84782e318cf33bbb2f219d0bd16

ARG SOURCE_DATE_EPOCH=0

COPY engine/patches/0001-bound-tokenize-response-context.patch /tmp/0001-bound-tokenize-response-context.patch
RUN test "$(sha256sum /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_tokenize.py | cut -d ' ' -f 1)" = \
      "f1791dbe89245cf80f2ae3c3f052e944c0b602deaf45909a31d2e4aabfc95711" && \
    patch --batch --fuzz=0 -p1 -d /sgl-workspace/sglang \
      < /tmp/0001-bound-tokenize-response-context.patch && \
    test "$(sha256sum /sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_tokenize.py | cut -d ' ' -f 1)" = \
      "e36cc8f7e196cea4e5616309c7c320a02b33bd0144c16a91f01e628df16a2c06" && \
    rm /tmp/0001-bound-tokenize-response-context.patch

COPY adapter/engine_frontend.py /opt/letsinfer/lib/engine_frontend.py
COPY adapter/engine-adapter /opt/letsinfer/bin/engine-adapter
RUN chmod 0555 /opt/letsinfer/bin/engine-adapter && \
    chmod 0444 /opt/letsinfer/lib/engine_frontend.py && \
    python3 -m compileall --invalidation-mode checked-hash -q \
      /opt/letsinfer/lib/engine_frontend.py

LABEL org.opencontainers.image.source="https://github.com/letsinferlabs/runtimes" \
      ai.letsinfer.runtime="sglang--inclusionai--ling-3.0-flash-int4--dgx-spark" \
      ai.letsinfer.feature="ling-3.0-flash,dspark,replayssm,exact-token-count,engine-protocol-v2" \
      ai.letsinfer.build.reproducibility="source-date-epoch-v1" \
      ai.letsinfer.sglang.source="0e5e40d8f1460976cd7190ae479c210f0642c120"
