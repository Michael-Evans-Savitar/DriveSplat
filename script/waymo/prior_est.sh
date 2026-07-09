#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-"${REPO_ROOT}/data/waymo/new_processed"}
CONDA_ENV=${CONDA_ENV:-drivesplat}

sequences=(016)

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

for seq in "${sequences[@]}"; do
  seq_padded=$(printf "%03d" "${seq}")
  python "${SCRIPT_DIR}/generate_lidar_depth.py" \
    --datadir "${DATA_ROOT}/${seq_padded}"
done
