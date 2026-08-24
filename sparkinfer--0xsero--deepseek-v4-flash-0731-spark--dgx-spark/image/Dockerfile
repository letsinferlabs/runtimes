FROM ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4 AS letsinfer-engine

ARG SOURCE_DATE_EPOCH=0

COPY engine/base-files.sha256 /tmp/letsinfer-base-files.sha256
RUN cd / && sha256sum -c /tmp/letsinfer-base-files.sha256

COPY engine/patches /tmp/letsinfer-patches
RUN cd /opt/sparkinfer && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/letsinfer-patches/0001-native-nvfp4-prefill-dispatch.patch && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/letsinfer-patches/0002-native-nvfp4-dual-cache-prefill.patch && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/letsinfer-patches/0003-preallocate-w4a16-route-histograms.patch && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/letsinfer-patches/0004-keep-graphed-inactive-routes-in-range.patch && \
    cd / && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/letsinfer-patches/0005-native-nvfp4-dspark-context-writer.patch && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/letsinfer-patches/0006-default-thinking-template-kwargs.patch && \
    patch --batch --forward --fuzz=0 -p1 < /tmp/letsinfer-patches/0007-cross-device-coalescer.patch && \
    /opt/runtime-venv/bin/python -m compileall --invalidation-mode checked-hash -q \
      /opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill.py \
      /opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill_mg.py \
      /opt/sparkinfer/sparkinfer/moe/fused_moe/_impl.py \
      /opt/sparkinfer/sparkinfer/moe/_shared/kernels/tiny_decode.py \
      /opt/vllm/vllm/models/deepseek_v4/nvidia/dspark.py \
      /opt/recipe/scripts/coalesce_rank_sliced_exl3.py && \
    bash -n /opt/vllm/serve-ds4-flash.sh && \
    rm -rf /tmp/letsinfer-patches /tmp/letsinfer-base-files.sha256

ADD --checksum=sha256:b1d2bef13aec8122dfa6b96e3ece2cc0a42a732685f1d44f6248c305274e8eae \
  https://files.pythonhosted.org/packages/18/10/baaac6c9ab3633bc5870d7cad893e6353b8345338610491679d1b61b5cfe/xgrammar-0.2.4-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl \
  /tmp/xgrammar-0.2.4-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl
RUN /opt/runtime-venv/bin/pip install --no-cache-dir --no-deps \
      /tmp/xgrammar-0.2.4-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl && \
    /opt/runtime-venv/bin/python -c \
      'import xgrammar; assert callable(xgrammar.normalize_tool_choice)' && \
    test -f /opt/runtime-venv/lib/python3.12/site-packages/transformers-5.13.1.dist-info/METADATA && \
    rm -f /tmp/xgrammar-0.2.4-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl

COPY scripts/prepare-and-serve.sh /opt/letsinfer/bin/sparkinfer-serve
COPY adapter/engine_frontend.py /opt/letsinfer/lib/engine_frontend.py
COPY adapter/engine-adapter /opt/letsinfer/bin/engine-adapter
RUN chmod 0555 /opt/letsinfer/bin/engine-adapter /opt/letsinfer/bin/sparkinfer-serve && \
    chmod 0444 /opt/letsinfer/lib/engine_frontend.py

LABEL org.opencontainers.image.source="https://github.com/letsinferlabs/runtimes" \
      ai.letsinfer.runtime="sparkinfer--0xsero--deepseek-v4-flash-0731-spark--dgx-spark" \
      ai.letsinfer.upstream.recipe.revision="76c51c7defffd01025c75daaf241ea323dda4734" \
      ai.letsinfer.upstream.sparkinfer.revision="272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f" \
      ai.letsinfer.build.reproducibility="source-date-epoch-v1"

FROM letsinfer-engine AS letsinfer-engine-inventory-build
COPY --from=letsinfer-tools tools/engine_sbom.py /tmp/letsinfer-engine-sbom.py
RUN python3 /tmp/letsinfer-engine-sbom.py snapshot \
      --output /tmp/engine-inventory.json

FROM scratch AS letsinfer-engine-inventory
COPY --from=letsinfer-engine-inventory-build \
  /tmp/engine-inventory.json /engine-inventory.json

FROM letsinfer-engine
