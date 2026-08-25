#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -Eeuo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: sparkinfer-serve MODEL_SOURCE SERVED_MODEL HOST PORT [VLLM_ARG ...]" >&2
  exit 2
fi

source_dir=$1
served_model=$2
host=$3
port=$4
shift 4

model_dir=${MODEL_PATH:-/root/models/tp1}
draft_dir=${SPEC_MODEL_PATH:-/root/models/dspark-draft-k64}
manifest=${model_dir}/rank-sliced-tp1-manifest.json

if [[ ! -d "${source_dir}" ]]; then
  echo "model snapshot is missing: ${source_dir}" >&2
  exit 1
fi

if [[ ! -f "${manifest}" ]]; then
  mkdir -p "${model_dir}"
  /opt/runtime-venv/bin/python /opt/recipe/scripts/coalesce_rank_sliced_exl3.py \
    --input-dir "${source_dir}" \
    --output-dir "${model_dir}" \
    --link-carried \
    --reuse-complete \
    --workers "${COALESCE_WORKERS:-1}"
fi

verify_args=()
if [[ "${VERIFY_MODEL_CHECKSUMS:-1}" != "1" ]]; then
  verify_args+=(--skip-checksums)
fi
/opt/runtime-venv/bin/python /opt/recipe/scripts/verify_tp1_manifest.py \
  "${model_dir}" "${verify_args[@]}"

if [[ ! -f "${draft_dir}/model.safetensors.index.json" ]]; then
  /opt/runtime-venv/bin/python /opt/recipe/scripts/build_dspark_draft.py \
    --source "${model_dir}" \
    --output "${draft_dir}" \
    --experts "${DSPARK_DRAFT_EXPERTS:-64}" \
    --structured-per-category "${DSPARK_STRUCTURED_EXPERTS_PER_CATEGORY:-32}"
fi

/opt/runtime-venv/bin/python /opt/recipe/scripts/selftest.py

export VLLM_PYTHON=/opt/runtime-venv/bin/python
export MODE=dspark
export BACKEND=b12x-a8
export INDEXER_BACKEND=b12x
export ALLREDUCE_MODE=nccl
export TP_SIZE=1
export DCP_SIZE=1
export MODEL_PATH=${model_dir}
export SPEC_MODEL_PATH=${draft_dir}
export SERVED_MODEL_NAME=${served_model}
export HOST=${host}
export PORT=${port}
export KV_CACHE_DTYPE=nvfp4_ds_mla
export LOAD_FORMAT=instanttensor
export PREFIX_CACHE=1
export ENABLE_FLASHINFER_AUTOTUNE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_B12X_WO_PROJECTION=1
export XDG_CACHE_HOME=/root/.cache

cache_args=()
if [[ -n "${LETSINFER_KV_TRANSFER_CONFIG:-}" ]]; then
  cache_args+=(--kv-transfer-config "${LETSINFER_KV_TRANSFER_CONFIG}")
fi

exec /opt/vllm/serve-ds4-flash.sh "${cache_args[@]}" "$@"
