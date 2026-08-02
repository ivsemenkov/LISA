#!/usr/bin/env bash
# Full dataset preparation: generate audio chunks and embeddings, preprocess MEG, and QC.
# This script runs all steps sequentially - useful for local development.
#
# For HPC, consider using prepare_dataset_gpu.sh and prepare_dataset_cpu.sh separately
# to avoid wasting GPU resources during MEG preprocessing.
#
# Usage:
#   bash scripts/prepare_dataset.sh [clean|raw] [n_jobs]
#
# Arguments:
#   MODE   - "clean" (default) for artifact-cleaned MEG in data/clean/,
#            "raw" for original MASC-MEG dataset in data/MASC-MEG/
#   N_JOBS - Number of parallel jobs for MEG preprocessing (default: 1).
#            Use -1 for all available CPUs.

set -euo pipefail

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate LISA

MODE="${1:-clean}"   # clean (default) or raw
N_JOBS="${2:-1}"     # number of parallel jobs (default: 1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

lisa-generate-embeddings

# Apply ICA if using clean mode
if [ "$MODE" = "clean" ]; then
  echo "Applying ICA to raw MEG data..."
  lisa-apply-ica \
    --raw-meg-dir "$PROJECT_ROOT/data/MASC-MEG" \
    --ica-dir "$PROJECT_ROOT/data/clean" \
    --output-dir "$PROJECT_ROOT/data/clean" \
    --n-jobs "$N_JOBS"
  echo ""
fi

if [ "$MODE" = "raw" ]; then
  lisa-prepare-meg \
    --meg-dir "$PROJECT_ROOT/data/MASC-MEG" \
    --meg-format "bids" \
    --n-jobs "$N_JOBS"
elif [ "$MODE" = "clean" ]; then
  lisa-prepare-meg --n-jobs "$N_JOBS"
else
  echo "Usage: $0 [clean|raw] [n_jobs]" >&2
  exit 1
fi

lisa-meg-qc

conda deactivate
