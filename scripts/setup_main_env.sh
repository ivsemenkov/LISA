#!/usr/bin/env bash
set -euo pipefail

FLAVOR="${1:-cu118}"  # cpu|cu118
if [ "$FLAVOR" != "cpu" ] && [ "$FLAVOR" != "cu118" ]; then
  echo "Usage: $0 [cpu|cu118]" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "LISA"; then
  if [ "${FORCE_RECREATE:-0}" = "1" ]; then
    conda env remove -y -n LISA
  else
    echo "ERROR: conda env 'LISA' already exists. Set FORCE_RECREATE=1 to recreate." >&2
    exit 1
  fi
fi

conda create -y -n LISA python=3.11 pip
conda activate LISA
python -m pip install -U pip

if [ "$FLAVOR" = "cpu" ]; then
  python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
fi

# Install package with ClearML extra so --logger clearml works out of the box.
# Training defaults to --logger local (disk only); pass --logger clearml when needed.
python -m pip install -e ".[clearml]"
python -m pip check

conda deactivate
