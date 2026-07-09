#!/usr/bin/env bash
set -euo pipefail

SCENE_ID=${1:?scene id, e.g. 026}
DATA_ROOT=${2:?processed Waymo root, e.g. /path/to/waymo/new_processed}
OUTPUT_ROOT=${3:-outputs/waymo/paper}
GPU=${GPU:-0}
PORT=${PORT:-6020}
CONFIG=${CONFIG:-arguments/waymo_default.py}

# Bind the target GPU before Python imports torch.
export CUDA_VISIBLE_DEVICES="${GPU}"

python train.py \
  --configs "${CONFIG}" \
  -s "${DATA_ROOT}/${SCENE_ID}" \
  --resolution -1 \
  -m "${OUTPUT_ROOT}/${SCENE_ID}" \
  --gpu "${GPU}" \
  --visible_threshold 0.01 \
  --base_layer -1 \
  --port "${PORT}"
