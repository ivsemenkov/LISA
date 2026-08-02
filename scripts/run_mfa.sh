#!/usr/bin/env bash
set -euo pipefail

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export MFA_ROOT_DIR="${MFA_ROOT_DIR:-$PROJECT_ROOT/.mfa_root}"
mkdir -p "$MFA_ROOT_DIR"

conda activate MFAligner

MFA_DICT_DIR="$MFA_ROOT_DIR/pretrained_models/dictionary"

STIMULI_DIR="$PROJECT_ROOT/data/MASC-MEG/stimuli"
cd "$PROJECT_ROOT"

# Preprocess stimuli for MFA alignment
python src/lisa/mfa/move_stimuli.py
python src/lisa/mfa/process_numbers.py
python src/lisa/mfa/find_oov_words.py --mfa-dict "$MFA_DICT_DIR/english_mfa.dict"

# Create full dict
mfa g2p "$STIMULI_DIR/oov_words.txt" english_us_mfa "$STIMULI_DIR/custom.dict"

cp "$STIMULI_DIR/custom.dict" "$MFA_DICT_DIR/custom.dict"
cd "$MFA_DICT_DIR"
cat english_mfa.dict custom.dict > full.dict

cd "$PROJECT_ROOT"

# Align files
mfa align "$STIMULI_DIR/combined" full english_mfa "$STIMULI_DIR/combined_mfa" --clean

conda deactivate
