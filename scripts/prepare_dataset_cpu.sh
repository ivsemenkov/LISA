#!/usr/bin/env bash
# CPU phase of dataset preparation: preprocess MEG data and run QC.
# This script should be run on a CPU-only node after prepare_dataset_gpu.sh completes.
#
# Usage:
#   bash scripts/prepare_dataset_cpu.sh [clean|raw] [n_jobs]
#
# Arguments:
#   MODE   - "clean" (default) for artifact-cleaned MEG in data/clean/,
#            "raw" for original MASC-MEG dataset in data/MASC-MEG/
#   N_JOBS - Number of parallel jobs for MEG preprocessing (default: 1).
#            Use -1 for all available CPUs.
#

set -euo pipefail

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate LISA

MODE="${1:-clean}"   # clean (default) or raw
N_JOBS="${2:-1}"     # number of parallel jobs (default: 1)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== CPU Phase: MEG Preprocessing and QC ==="
echo "Mode: $MODE, N_JOBS: $N_JOBS"
echo "Starting at $(date)"

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

# Preprocess MEG data (CPU-only, supports parallelization)
echo "Running lisa-prepare-meg with $N_JOBS jobs..."
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

# Run MEG quality check
echo "Running lisa-meg-qc..."
lisa-meg-qc

echo "CPU phase completed at $(date)"

conda deactivate
