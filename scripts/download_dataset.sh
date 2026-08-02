#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate LISA

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DEST="$PROJECT_ROOT/data/MASC-MEG"
ICA_DEST="$PROJECT_ROOT/data/clean"

mkdir -p "$DATA_DEST" "$ICA_DEST" "$PROJECT_ROOT/logs"

# Datasets -> data/MASC-MEG (flat layout)
lisa-download-osf \
  --project-ids ag3kj hqvm3 u5327 dr4wy \
  --output-dir "$DATA_DEST" \
  --no-project-subdir \
  "$@"

# ICA project: only clean/ -> data/clean, drop clean/ from local paths
lisa-download-osf \
  --project-ids 3wrft \
  --output-dir "$PROJECT_ROOT/data" \
  --no-project-subdir \
  --remote-subdir clean/ \
  "$@"

conda deactivate
