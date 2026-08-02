#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "MFAligner"; then
  if [ "${FORCE_RECREATE:-0}" = "1" ]; then
    conda env remove -y -n MFAligner
  else
    echo "ERROR: conda env 'MFAligner' already exists. Set FORCE_RECREATE=1 to recreate." >&2
    exit 1
  fi
fi

conda create -y -n MFAligner --override-channels -c conda-forge --strict-channel-priority \
  python=3.12 montreal-forced-aligner=3.3.8 num2words=0.5.14 pip

conda activate MFAligner

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$PROJECT_ROOT/.mfa_root}"
mkdir -p "$MFA_ROOT_DIR"

mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
mfa model download g2p english_us_mfa

python -m pip install -e . --no-deps
conda deactivate
