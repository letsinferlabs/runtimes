ARG SOURCE_DATE_EPOCH=1785594180

FROM ghcr.io/pyo3/maturin@sha256:b6c8b59a0170b77eb31a35b56034abd39972483ad0ebfff344deaa42a85f3bd3 AS bridge

WORKDIR /build/letsinfer-bridge
COPY adapter/letsinfer-bridge/ ./
RUN SOURCE_DATE_EPOCH=1785594180 cargo build --release --locked \
    && strip --strip-debug target/release/libletsinfer_prefix_capi.so \
    && install -D -m 0555 target/release/libletsinfer_prefix_capi.so \
      /opt/letsinfer/lib/libletsinfer_prefix_capi.so

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS build

WORKDIR /build/dwarfstar
COPY engine/ ./

# Fail the image build if the qualified scheduler, memory, cache, and exact
# token-count contracts are absent from the packed source.
RUN grep -Fq 'if (slot >= 0 && !g_stop_requested)' ds4_server.c \
    && grep -Fq '#define DS4_COALESCE_HARD_MAX 128' ds4_server.c \
    && grep -Fq 'awaiting host-memory headroom' ds4_server.c \
    && grep -Fq 'DS4_BATCH_IDLE_HOST_FLOOR_MB' ds4.c \
    && grep -Fq 'DS4_BATCH_PENDING_HOST_RESERVE_MB' ds4.c \
    && grep -Fq 'ds4_batch_pending_host_reserve_for_tokens' ds4.c \
    && grep -Fq 'busy_host_reserve' ds4.c \
    && grep -Fq 'overlap_host_reserve' ds4.c \
    && grep -Fq 'mneed != 0 || pending_host_reserve != 0' ds4.c \
    && ! grep -Fq '1u + busy_rows + (other_busy ? 1u : 0u)' ds4.c \
    && grep -Fq 'const uint64_t idle_floor = other_busy ? 0' ds4.c \
    && grep -Fq 'const bool reuse_retry_plan = memory_retry && j->cont_retry_plan_valid' ds4_server.c \
    && grep -Fq 'j->cont_memory_waits > 0 || j->cont_retry_target_paid' ds4_server.c \
    && grep -Fq 'req->place_bank = j->cont_retry_place_bank' ds4_server.c \
    && grep -Fq 'l == plcp && b2 == restored_bank' ds4_server.c \
    && grep -Fq '!live_covers_prompt && s->letsinfer' ds4_server.c \
    && grep -Fq 'pb == restored_bank || plcp == plen' ds4_server.c \
    && ! grep -Fq 'DS4_CONT_IDLE_MEMORY_RETRY_MAX' ds4_server.c \
    && grep -Fq 'request-owned raw/comp/index cache slabs' ds4.c \
    && grep -Fq 'ds4_batch_bank_map_ensure(ctx, b, flen)' ds4.c \
    && grep -Fq 'mneed == 0u && candidate_host_reserve == 0u' ds4.c \
    && grep -Fq '!bank_mapping_paid && !ds4_batch_bank_map_ensure' ds4.c \
    && grep -Fq 'ds4_dspark_slabs_prepare_bank(&ctx->dsl, &ctx->g, b)' ds4.c \
    && grep -Fq 'ds4_gpu_tensor_trim(sl->multi_raw[il]' ds4.c \
    && grep -Fq 'ds4_gpu_tensor_trim(sl->multi_comp_scale[il]' ds4.c \
    && grep -Fq 'ds4_gpu_tensor_trim(sl->multi_index_scale[il]' ds4.c \
    && grep -Fq 'ds4_batch_ctx_trim_idle_bank' ds4.c \
    && grep -Fq 'DS4_SERVER_SHALLOW_IDLE_MAX' ds4_server.c \
    && grep -Fq 'warm_idle_reclaim(s)' ds4_server.c \
    && grep -Fq 'parse_generated_legacy_tool_call' ds4_server.c \
    && grep -Fq 'DS4_DSPARK_AUTO_BATCH_MAX_EXTENT 24576u' ds4_batch_policy.h \
    && grep -Fq 'batch_dspark_policy_id' ds4.c \
    && grep -Fq 'const ds4_dspark_batch_policy prefill_auto_batch' ds4.c \
    && grep -Fq 'ds4_gpu_set_dspark_batch_policy(prefill_sorted_wide' ds4.c \
    && grep -Fq 'segmented prefill cohort fenced above 32768' ds4.c \
    && grep -Fq 'ds4_gpu_tensor_resident_total(sl->multi_raw[il])' ds4.c \
    && grep -Fq 't->vmm_reservation = base->vmm_reservation' ds4_cuda.cu \
    && grep -Fq 'ds4_gpu_set_dspark_batch_policy' ds4_cuda.cu \
    && grep -Fq 'handle_token_count' ds4_server.c \
    && grep -Fq '/v1/token-count' ds4_server.c \
    && grep -Fq 'ds4_tokenize_rendered_chat' ds4_server.c

# The CUDA base exports packaging-only CFLAGS. Remove them so DwarfStar's
# retained -O3/-ffast-math/-march=native release flags stay authoritative.
RUN env -u CFLAGS -u CXXFLAGS \
      SOURCE_DATE_EPOCH=1785594180 ZERO_AR_DATE=1 \
      make -j"$(nproc)" cuda CUDA_ARCH=sm_121 CUDA_EXTRA_BINS= \
        DEBUG_FLAGS="-g -frandom-seed=ds4-overdrive-6cd1146" \
        NVCC_EXTRA_FLAGS="--frandom-seed=1785594180" \
    && strip --strip-debug ds4-server \
    && objcopy --remove-section .note.gnu.build-id ds4-server \
    && install -D -m 0555 ds4-server /opt/dwarfstar/ds4-server \
    && find /opt/dwarfstar -exec touch --date=@1785594180 {} +

FROM ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x@sha256:19627342e1da2607f4db50745dca30e57d7dd0ebff06062f03fd69b43a252931

ARG LETSINFER_RUNTIME_REVISION=letsinfer-ds4-0.11.0-rc.1
LABEL org.opencontainers.image.title="DS4 Overdrive for Let's Infer" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND MIT" \
      org.opencontainers.image.revision="${LETSINFER_RUNTIME_REVISION}" \
      ai.letsinfer.build.reproducibility="source-date-epoch-v1"

RUN --mount=from=build,source=/opt/dwarfstar,target=/build-output,ro \
    --mount=from=bridge,source=/opt/letsinfer/lib,target=/bridge-output,ro \
    install -D -m 0555 /build-output/ds4-server /opt/dwarfstar/ds4-server \
    && install -D -m 0555 /bridge-output/libletsinfer_prefix_capi.so \
      /opt/letsinfer/lib/libletsinfer_prefix_capi.so \
    && find /opt/dwarfstar -exec touch --date=@1785594180 {} + \
    && touch --date=@1785594180 /opt /

COPY adapter/engine_frontend.py /opt/letsinfer/lib/engine_frontend.py
COPY adapter/engine-adapter /opt/letsinfer/bin/engine-adapter
RUN chmod 0555 /opt/letsinfer/bin/engine-adapter \
    && chmod 0444 /opt/letsinfer/lib/engine_frontend.py \
    && find /opt/letsinfer -exec touch --date=@1785594180 {} +

WORKDIR /opt/dwarfstar
