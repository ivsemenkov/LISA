#!/usr/bin/env bash
# GPU phase of dataset preparation: generate audio chunks and embeddings.
# This script should be run on a GPU node as generate-embeddings uses foundation models.
#
# Usage:
#   bash scripts/prepare_dataset_gpu.sh

set -euo pipefail

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate LISA

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== GPU Phase: Generate Audio Chunks and Embeddings ==="
echo "Starting at $(date)"

# Generate audio chunks and embeddings using a foundation model (GPU-accelerated)
echo "Running lisa-generate-embeddings..."
lisa-generate-embeddings

echo "GPU phase completed at $(date)"

conda deactivate
