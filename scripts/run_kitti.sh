#!/usr/bin/env bash
set -euo pipefail

SEQ=${1:?KITTI sequence suffix, e.g. 01}
DATA_ROOT=${2:?processed KITTI root}
OUTPUT_ROOT=${3:-outputs/kitti/paper}
GPU=${GPU:-0}
PORT=${PORT:-6030}
CONFIG=${CONFIG:-}
CONFIG_ARGS=()
if [[ -n "${CONFIG}" ]]; then
  CONFIG_ARGS=(--configs "${CONFIG}")
fi

# Bind the target GPU before Python imports torch.
export CUDA_VISIBLE_DEVICES="${GPU}"

python train.py \
  "${CONFIG_ARGS[@]}" \
  -s "${DATA_ROOT}/2011_09_26_drive_00${SEQ}_sync" \
  --resolution -1 \
  -m "${OUTPUT_ROOT}/${SEQ}" \
  --gpu "${GPU}" \
  --visible_threshold -1 \
  --base_layer -1 \
  --port "${PORT}"
