ARG SOURCE_DATE_EPOCH=0

FROM ghcr.io/pyo3/maturin@sha256:b6c8b59a0170b77eb31a35b56034abd39972483ad0ebfff344deaa42a85f3bd3 AS letsinfer-prefix-store-build

ARG SOURCE_DATE_EPOCH
WORKDIR /build/letsinfer-prefix-store
COPY adapter/letsinfer_prefix_store/ ./
RUN SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
    cargo test --release --locked && \
    cargo build --release --locked --features python && \
    strip --strip-debug target/release/libletsinfer_prefix_store.so && \
    install -D -m 0555 target/release/libletsinfer_prefix_store.so \
      /opt/letsinfer/python/letsinfer_prefix_store.so

FROM docker.io/eugr/spark-vllm-b12x@sha256:25fe41c2e85993b4e0534b3c72f68bf327c5d2726fbe1640cf6b220715d3b0e3 AS letsinfer-engine

ARG SOURCE_DATE_EPOCH

COPY engine/base-files.sha256 /tmp/letsinfer-base-files.sha256
RUN cd / && sha256sum -c /tmp/letsinfer-base-files.sha256

COPY engine/overlays/kernel.py \
  /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/kernel.py
COPY engine/overlays/route_pack.py \
  /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/route_pack.py
RUN cd / && printf '%s  %s\n' \
      c7c0437bc5337021578d5842b656cef436ada6c5f95635ad68382bd34a4486fa \
      usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/kernel.py \
      3d444809990e7cb559089b664406f8c1b95cf43bde62caa57d7d7441287cef7a \
      usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/route_pack.py \
      | sha256sum -c - && \
    python3 -c 'import ast,pathlib; [ast.parse(pathlib.Path(path).read_text(encoding="utf-8"), filename=path) for path in ("/usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/kernel.py", "/usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/route_pack.py")]'

COPY adapter/engine_frontend.py /opt/letsinfer/lib/engine_frontend.py
COPY adapter/engine-adapter /opt/letsinfer/bin/engine-adapter
COPY --from=letsinfer-prefix-store-build \
  /opt/letsinfer/python/letsinfer_prefix_store.so \
  /usr/local/lib/python3.12/dist-packages/letsinfer_prefix_store.so
COPY adapter/letsinfer_prefix_connector/__init__.py \
  /usr/local/lib/python3.12/dist-packages/letsinfer_prefix_connector/__init__.py
COPY adapter/letsinfer_prefix_connector/connector.py \
  /usr/local/lib/python3.12/dist-packages/letsinfer_prefix_connector/connector.py
RUN --mount=type=bind,source=adapter/letsinfer_prefix_connector/tests,target=/tmp/connector-tests,readonly \
    PYTHONDONTWRITEBYTECODE=1 python3 /tmp/connector-tests/run.py
RUN chmod 0555 /opt/letsinfer/bin/engine-adapter && \
    chmod 0444 \
      /opt/letsinfer/lib/engine_frontend.py \
      /usr/local/lib/python3.12/dist-packages/letsinfer_prefix_connector/__init__.py \
      /usr/local/lib/python3.12/dist-packages/letsinfer_prefix_connector/connector.py && \
    /opt/letsinfer/bin/engine-adapter verify --protocol 2 && \
    python3 -B -c 'from letsinfer_prefix_store import PrefixStore; from letsinfer_prefix_connector import LetsInferPrefixConnector; store = PrefixStore("/tmp/letsinfer-prefix-store-selftest", 1048576, 60, 4, 0, True); assert store is not None; assert LetsInferPrefixConnector.__name__ == "LetsInferPrefixConnector"' && \
    rm -rf /tmp/letsinfer-prefix-store-selftest

LABEL org.opencontainers.image.source="https://github.com/letsinferlabs/runtimes" \
      ai.letsinfer.runtime="vllm-b12x--ornith-ai--ornith-1.5-35b-a3b-nvfp4--dgx-spark" \
      ai.letsinfer.upstream.sparkbench.revision="253c5e44d83c79db18daf0c0e40d73c27940e682" \
      ai.letsinfer.upstream.mia.revision="9fd7fe7896230550fad1c773e85ad765292cfde5" \
      ai.letsinfer.upstream.vllm.revision="ad848fc4141f201489db18d5453c50b312245a0a" \
      ai.letsinfer.upstream.b12x.revision="a63f07e90fd449b693cafbe6aef1a73309595bf7" \
      ai.letsinfer.kv-connector.address-stability="host-copy-no-device-registration-v1" \
      ai.letsinfer.build.reproducibility="source-date-epoch-v1"

FROM letsinfer-engine AS letsinfer-engine-inventory-build
COPY --from=letsinfer-tools tools/engine_sbom.py /tmp/letsinfer-engine-sbom.py
RUN python3 /tmp/letsinfer-engine-sbom.py snapshot \
      --output /tmp/engine-inventory.json

FROM scratch AS letsinfer-engine-inventory
COPY --from=letsinfer-engine-inventory-build \
  /tmp/engine-inventory.json /engine-inventory.json

FROM letsinfer-engine
