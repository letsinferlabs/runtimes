#!/bin/sh
set -eu

candidate_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output=${1:?usage: build-ios-engine.sh OUTPUT_DIRECTORY}
source_root=${MLC_LLM_SOURCE_DIR:?set MLC_LLM_SOURCE_DIR to the pinned MLC LLM checkout}
model_root=${MLC_MODEL_SOURCE_DIR:?set MLC_MODEL_SOURCE_DIR to the pinned MLC model checkout}
expected_revision=9fa644f54b04983adea4d0168f49fc6af4a893ba
expected_model_revision=8c14ce481d4c692769976ad52afea453a102df19
actual_revision=$(git -C "$source_root" rev-parse HEAD)
actual_model_revision=$(git -C "$model_root" rev-parse HEAD)

if [ "$actual_revision" != "$expected_revision" ]; then
  echo "MLC LLM source revision differs" >&2
  exit 1
fi
if [ "$actual_model_revision" != "$expected_model_revision" ]; then
  echo "MLC model source revision differs" >&2
  exit 1
fi
if ! git -C "$source_root" diff --quiet HEAD -- \
  || ! git -C "$model_root" diff --quiet HEAD --
then
  echo "MLC source or model checkout contains tracked changes" >&2
  exit 1
fi
if git -C "$source_root" submodule status --recursive | grep -Eq '^[+-U]'; then
  echo "MLC source submodules are not at their pinned revisions" >&2
  exit 1
fi
(
  cd "$model_root"
  shasum -a 256 -c "$candidate_dir/engine/model-files.sha256"
)

mkdir -p "$output"
cp "$candidate_dir/engine/mlc-package-config.json" "$output/mlc-package-config.json"
cache="$output/.model-cache"
cache_model="$cache/hf/mlc-ai/Qwen3-0.6B-q4f16_1-MLC"
mkdir -p "$(dirname "$cache_model")"
ln -s "$model_root" "$cache_model"
cleanup() {
  if [ -L "$cache_model" ]; then unlink "$cache_model"; fi
  rmdir "$cache/hf/mlc-ai" "$cache/hf" "$cache" 2>/dev/null || true
}
trap cleanup EXIT
(
  cd "$output"
  MLC_DOWNLOAD_CACHE_POLICY=READONLY \
  MLC_LLM_READONLY_WEIGHT_CACHE="$cache" \
  MLC_LLM_SOURCE_DIR="$source_root" \
  PYTHONPATH="$source_root/python${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m mlc_llm package
)

for artifact in \
  dist/lib/libmlc_llm.a \
  dist/lib/libmodel_iphone.a \
  dist/lib/libsentencepiece.a \
  dist/lib/libtokenizers_cpp.a \
  dist/lib/libtokenizers_c.a \
  dist/lib/libtvm_ffi_static.a \
  dist/lib/libtvm_runtime.a \
  dist/bundle/mlc-app-config.json
do
  if [ ! -f "$output/$artifact" ]; then
    echo "MLC Engine output is incomplete: $artifact" >&2
    exit 1
  fi
done
