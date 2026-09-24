# Reproducing the LISA experiments and figures

This document describes the complete workflow used for the MASC-MEG experiments
in this repository: data preparation, the full training matrix, aggregation of
retrieval results, and the interpretation analyses applied to the selected
2-block, 25-branch model.

If you mainly want to run your own configurations or reuse the encoder, start
with the README and [`ADAPTING.md`](ADAPTING.md) instead.

For model equations and tensor flow, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
For the effective behavior of individual `lisa-train` options, see
[`HYPERPARAMETERS.md`](HYPERPARAMETERS.md).

> [!IMPORTANT]
> Run all commands from the repository root. The commands below target the
> current codebase and the cleaned 208-channel MASC-MEG pipeline used in the
> experiments.

## Sources of truth for completed runs

Every completed run directory contains three complementary provenance files:

- `cli.txt` — the exact command line recorded from `sys.argv` when the run was
  created;
- `config.json` — the normalized effective configuration after parsing and
  derived-value resolution;
- `metrics.csv` — per-epoch train, validation, auxiliary-test, and final-test
  metrics.

`cli.txt` is the canonical audit trail for how a particular run was launched.
It may contain an absolute path to the `lisa-train` executable from the original
environment; replacing that prefix with `lisa-train` does not change the
command. `config.json` is the canonical source for the effective values used by
the model, including values derived after parsing.

The final experimental design contains 320 training runs. Every run generated
by this workflow contains `cli.txt`, `config.json`, and `metrics.csv`. The
compact shell loops below reproduce the design without listing 320 nearly
identical commands individually.

# 1. Environment and dataset

## 1.1 Create the environments

The main environment uses Python 3.11 and PyTorch 2.6.0. The MFA environment is
separate because Montreal Forced Aligner has its own dependency stack.

```bash
bash scripts/setup_mfa_env.sh
bash scripts/setup_main_env.sh          # CUDA 11.8 PyTorch
# or
bash scripts/setup_main_env.sh cpu      # CPU-only PyTorch
```

Activate the main environment for all subsequent `lisa-*` commands unless a
step explicitly uses MFA:

```bash
conda activate LISA
```

The setup scripts install the repository in editable mode. Therefore the CLI
entry points call the current source tree rather than a copied package.

## 1.2 Download the source data and ICA files

```bash
bash scripts/download_dataset.sh
```

The script downloads:

- the original MASC-MEG projects into `data/MASC-MEG/`;
- the precomputed ICA solutions from [OSF project 3wrft](https://osf.io/3wrft/)
  into `data/clean/`.

Additional options accepted by `lisa-download-osf` may be appended to the shell
script command.

## 1.3 Run forced alignment

```bash
conda activate MFAligner
bash scripts/run_mfa.sh
```

For this step use `MFAligner` environment. 
This prepares the audio/text inputs, generates pronunciations for out-of-
vocabulary items, and writes TextGrid alignments under:

```text
data/MASC-MEG/stimuli/combined_mfa/
```

## 1.4 Prepare the cleaned dataset

The paper experiments use the artifact-cleaned MEG recordings:

```bash
bash scripts/prepare_dataset.sh clean 8
```

The second argument is the number of parallel CPU jobs. Use `-1` for all
available CPUs.

On a cluster, the GPU and CPU phases can be separated:

```bash
# GPU node: audio chunks and Wav2Vec2 embeddings
bash scripts/prepare_dataset_gpu.sh

# CPU node: apply ICA, preprocess MEG, and run QC
bash scripts/prepare_dataset_cpu.sh clean 8
```

The canonical preprocessing defaults are:

- audio window: 3.0 s;
- audio overlap: 2.0 s, corresponding to a 1.0 s stride;
- Wav2Vec2 checkpoint: `facebook/wav2vec2-base-960h`;
- audio representation: mean of the last four hidden layers;
- MEG sampling rate: 100 Hz;
- MEG offset relative to audio: +150 ms during training;
- validation pair: story 3, sound 4.

The core generated files are:

```text
data/preprocessed/audio/extract_features_train4.npy
data/preprocessed/audio/extract_features_test4.npy

data/preprocessed/dataframe/chunks_info_train.csv
data/preprocessed/dataframe/chunks_info_test.csv
data/preprocessed/dataframe/df_train27.csv
data/preprocessed/dataframe/df_test27.csv

data/preprocessed/meg/meg27_sr100.npz
data/preprocessed/meg/meg27_sr100_scalers.npz

data/preprocessed/coords/sensor_xyz.npy
data/preprocessed/coords/coords208_xy_scaled.npy
```

`meg27_sr100.npz` stores the normalized MEG used for training. The sibling
sidecar `meg27_sr100_scalers.npz` records the channel preprocessing transform.
Physiological and source interpretation invert that transform before
constructing physical-unit patterns.

`sensor_xyz.npy` and `coords208_xy_scaled.npy` are the sensor-layout assets
used by spatial attention. Source analysis reconstructs participant- and
session-specific KIT geometry from the BIDS ELP, HSP, and MRK files, fits a
participant-specific head-to-`fsaverage` transform and forward model, and
reports the group result in the shared `fsaverage` frame. See Section 6.7.

For the canonical dataframe generated by the commands above:

| Split | Candidate audio windows | Subject/session rows |
|---|---:|---:|
| Train, before validation holdout | 2,698 | 132,202 |
| Train-fit after holding out `3:4` | 2,590 | 126,910 |
| Validation `3:4` | 108 | 5,292 |
| Test | 1,005 | 49,245 |

The raw-data control can be prepared with `raw` instead of `clean`, but this is
not the dataset variant used by the reported experiments.

# 2. Additional precomputed inputs

The baseline, linear-reduction, and architecture runs use the canonical files
from Section 1 directly. PCA reductions, alternative segment lengths, and the
occlusion-feature analysis require additional preprocessing.

## 2.1 PCA-reduced audio embeddings

Feature-axis PCA was evaluated at 16, 32, 64, 128, and 256 components:

```bash
for dim in 16 32 64 128 256; do
  lisa-precompute-pca \
    --embedding-layers 4 \
    --axis feature \
    --n-components "$dim"
done
```

Time-axis PCA was evaluated at 32 and 64 components:

```bash
for dim in 32 64; do
  lisa-precompute-pca \
    --embedding-layers 4 \
    --axis time \
    --n-components "$dim"
done
```

PCA is fitted on the complete non-test training audio array and then applied to
the test array. The utility writes reduced train/test arrays and an NPZ file
containing the fitted PCA parameters under `data/preprocessed/audio/`.

## 2.2 Segment-duration datasets

The duration experiment uses the same anchors for every duration. All anchors
are gated by feasibility at 5 s, so the train rows, test rows, and retrieval
candidate bank are identical across the complete sweep.

First verify that the current Wav2Vec2 setup reproduces the canonical 3 s
embeddings:

```bash
lisa-regen-windows --verify-baseline
```

Generate all five common-anchor datasets:

```bash
lisa-regen-windows \
  --window-sizes 1.5 2.25 3 4 5 \
  --anchor-window 5
```

Because `--window-tags` is omitted, the utility assigns the neutral tags `w1p5`,
`w2p25`, `w3`, `w4`, and `w5`. The command uses the available device.

The resulting common-anchor datasets have:

| Split | Candidate audio windows | Subject/session rows |
|---|---:|---:|
| Train, before validation holdout | 2,652 | 129,948 |
| Train-fit after holding out `3:4` | 2,546 | 124,754 |
| Validation `3:4` | 106 | 5,194 |
| Test | 991 | 48,559 |

The tagged `w3` dataset is a regenerated 3 s dataset restricted to the same
5 s-feasible anchors as the other duration conditions. It is therefore not
identical in sample count to the unrestricted canonical 3 s baseline.

## 2.3 Linguistic features for occlusion analysis

Occlusion-feature analysis uses MFA words and phonemes, GPT-2 surprisal and
entropy, word frequency, insertion labels, loudness, and acoustic onset
features. Precompute the per-word table once:

```bash
lisa-precompute-word-features --device cuda
```

Use `--device cpu` when CUDA is unavailable. The default output is:

```text
data/preprocessed/linguistic/word_features.csv
```

# 3. Common training protocol

## 3.1 Logging backend

Training always writes the same local run directory layout. The logger only
controls whether metrics and figures are also sent to a ClearML server.

Use local disk logging (the CLI default):

```bash
RUNTIME_ARGS=(
  --device cuda
)
```

Equivalent explicit form:

```bash
RUNTIME_ARGS=(
  --device cuda
  --logger local
)
```

Optional ClearML logging (requires a configured ClearML account):

```bash
RUNTIME_ARGS=(
  --device cuda
  --logger clearml
)
```

ClearML changes run identifiers and remote tracking, but not the model
architecture, optimizer, split, or loss. ClearML is installed by
`scripts/setup_main_env.sh`. In ClearML offline mode, run directories contain
`_offline-<run-id>`; local runs use `_<run-id>`.

## 3.2 Common experiment arguments

The following arguments were shared by all 320 final paper runs:

```bash
EXPERIMENTS_ROOT="outputs/experiments"

PAPER_COMMON_ARGS=(
  --use-spatial-attention 3D
  --head-pool single_conv
  --dl-n-workers 8
  --nepoch 50
  --weight-decay 0
  --upload-metrics-csv
  --torch-deterministic
  --early-stopping-patience 7
)
```

The helper below is used by the compact commands in the following sections:

```bash
run_paper_model() {
  local run_group="$1"
  local depth="$2"
  local branches="$3"
  local seed="$4"
  shift 4

  lisa-train \
    "${RUNTIME_ARGS[@]}" \
    "${PAPER_COMMON_ARGS[@]}" \
    --experiments-root "$EXPERIMENTS_ROOT" \
    --run-group "$run_group" \
    --seed "$seed" \
    --n-temporal-module-blocks "$depth" \
    --n-channels-unmix "$branches" \
    --n-channels-block "$branches" \
    "$@"
}
```

The model is checkpointed only when validation loss strictly improves. The
current training loop also records auxiliary test metrics after every epoch,
but they do not enter optimization, early stopping, or checkpoint selection.
After training, the best-validation model, audio adapter, and criterion are
reloaded and the final test metrics are computed again.

## 3.3 Run-directory contents

A completed baseline run normally contains:

```text
outputs/experiments/<run-group>/<run-name>_<run-id>/
├── cli.txt
├── config.json
├── metrics.csv
├── lisa_checkpoint.pt
├── criterion_best.pt
├── audio_model_best.pt             # only when the audio adapter has parameters
├── final_test_per_window.csv
├── final_test_per_window.npz
└── filters_<run-name>.npz
```

The exact set of checkpoints depends on the reduction mode. An identity audio
adapter does not require `audio_model_best.pt`.

# 4. Complete training matrix

The final paper experiment matrix contains the following families:

| Family | Runs |
|---|---:|
| Branch-count × temporal-depth sweep | 144 |
| Additional seeds for the selected 2-block K=25 model | 5 |
| Architecture ablations | 54 |
| Temporal-filter-support additional runs | 39 |
| Feature-axis reductions | 48 |
| Time-axis reductions and pooling | 15 |
| Segment-duration runs | 15 |
| **Total** | **320** |

The loops below reproduce every retained family. A condition that reuses a run
from an earlier family is left out of the later family's count and is not
trained again.

## 4.1 Branch-count and temporal-depth sweep: 144 runs

The branch grid is:

```bash
K_VALUES=(
  1 2 3 4 5 6 7 8 9 10
  15 20 25 30 35 40 45 50
  75 100 150 200 250 270
)
```

Run all six temporal depths with seed 42:

```bash
for depth in 0 1 2 3 4 5; do
  for k in "${K_VALUES[@]}"; do
    run_paper_model \
      "${depth}conv-seed42-paper" \
      "$depth" "$k" 42
  done
done
```

This produces six run groups, each containing one run for every value of
`n_channels_unmix`. `n_channels_block` is set to the same value throughout the
reported sweep.

## 4.2 Additional seeds for the selected main model: 5 runs

The selected model for the main analyses has two temporal blocks and `K=25`.
Its seed-42 run is already included in the main sweep. Add seeds 43, 44, 45,
46, and 47 as follows:

```bash
for seed in 43 44 45 46 47; do
  run_paper_model \
    "2conv-seed${seed}-paper" \
    2 25 "$seed"
done
```

Together with the seed-42 sweep run, this gives the six main seeds 42–47 for
the selected 2-block, K=25 model.

## 4.3 Architecture ablations: 54 runs

Final Figure 13 evaluates nine architecture variants of the selected 2-block,
K=25 model at each of the six main seeds 42, 43, 44, 45, 46, and 47
(9 × 6 = 54 runs). The corresponding full-model baselines are already included
in Sections 4.1 and 4.2 and are not repeated or counted here. This family is
the 2-block ablation set only.

Some variants remove one component of the spatial front-end, whereas the
`only-*` variants retain one spatial transformation path while disabling the
others.

```bash
for seed in 42 43 44 45 46 47; do
  # Replace 3D spherical-harmonic attention with 2D Fourier attention.
  run_paper_model \
    "ablate-2dfull-2conv-seed${seed}-paper" \
    2 25 "$seed" \
    --use-spatial-attention 2D

  # Disable spatial attention while retaining shared unmixing and the
  # subject-specific mapping.
  run_paper_model \
    "ablate-no3d-2conv-seed${seed}-paper" \
    2 25 "$seed" \
    --use-spatial-attention none

  # Remove only the subject-specific mapping.
  run_paper_model \
    "ablate-nosubject-2conv-seed${seed}-paper" \
    2 25 "$seed" \
    --no-use-subject-layer

  # Remove only the shared unmixing layer.
  run_paper_model \
    "ablate-nounmix-2conv-seed${seed}-paper" \
    2 25 "$seed" \
    --no-use-unmixing-layer

  # Retain only 2D spatial attention before the temporal filter.
  run_paper_model \
    "ablate-only2d-2conv-seed${seed}-paper" \
    2 25 "$seed" \
    --use-spatial-attention 2D \
    --no-use-unmixing-layer \
    --no-use-subject-layer \
    --n-channels-attention 25

  # Retain only 3D spatial attention before the temporal filter.
  run_paper_model \
    "ablate-only3d-2conv-seed${seed}-paper" \
    2 25 "$seed" \
    --no-use-unmixing-layer \
    --no-use-subject-layer \
    --n-channels-attention 25

  # Retain only the subject-specific spatial mapping.
  run_paper_model \
    "ablate-onlysubject-2conv-seed${seed}-paper" \
    2 25 "$seed" \
    --use-spatial-attention none \
    --no-use-unmixing-layer

  # Retain only the shared unmixing layer.
  run_paper_model \
    "ablate-onlyunmix-2conv-seed${seed}-paper" \
    2 25 "$seed" \
    --use-spatial-attention none \
    --no-use-subject-layer

  # Replace the temporal filter with a one-tap depthwise transformation.
  run_paper_model \
    "ablate-tfkernel1-2conv-seed${seed}-paper" \
    2 25 "$seed" \
    --temporal-filter-kernel-size 1
done
```

The one-tap condition retains a learned channel-wise depthwise weight but has
no temporal receptive field. It therefore removes temporal mixing from the
interpretable filter while preserving the required channel-wise
transformation. Seeds 42–44 of this one-tap run are reused by the
temporal-filter-support sweep in Section 4.4 and are not trained again there.

## 4.4 Temporal-filter-support sweep: 39 additional runs

This sweep holds the selected architecture fixed at two temporal blocks and
25 branches. It covers 15 supports at seeds 42, 43, and 44. At 100 Hz those
supports run from 10 ms through 2.99 s:

| Taps | Nominal length at 100 Hz | Run group | Already counted | Additional |
|---:|---:|---|---|---|
| 1 | 10 ms | `ablate-tfkernel1-2conv-seed{seed}-paper` | seeds 42–44 (Section 4.3) | — |
| 5 | 50 ms | `tfk5-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 9 | 90 ms | `tfk9-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 15 | 150 ms | `2conv-seed{seed}-paper` | seed 42 (Section 4.1), seeds 43, 44 (Section 4.2) | — |
| 19 | 190 ms | `tfk19-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 25 | 250 ms | `tfk25-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 29 | 290 ms | `tfk29-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 49 | 490 ms | `tfk49-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 65 | 650 ms | `tfk65-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 81 | 810 ms | `tfk81-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 101 | 1010 ms | `tfk101-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 151 | 1510 ms | `tfk151-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 201 | 2010 ms | `tfk201-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 251 | 2510 ms | `tfk251-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |
| 299 | 2990 ms | `tfk299-2conv-seed{seed}-paper` | — | seeds 42, 43, 44 |

Train the 13 supports that are not already counted:

```bash
for seed in 42 43 44; do
  for taps in 5 9 19 25 29 49 65 81 101 151 201 251 299; do
    run_paper_model \
      "tfk${taps}-2conv-seed${seed}-paper" \
      2 25 "$seed" \
      --temporal-filter-kernel-size "$taps"
  done
done
```

The 1-tap condition is the no-temporal-context control already trained in
Section 4.3 for seeds 42–44. The 15-tap, 150 ms condition is the already-trained
main model, from Section 4.1 for seed 42 and Section 4.2 for seeds 43 and 44.
The sweep therefore contains 15 supports × 3 seeds = 45 conditions and adds
13 × 3 = 39 runs. The paper reports a broad long-support regime: mean Top-1 is
highest at 2.51 s, mean Top-10 is highest at 1.51 s, and both are lower again
at 2.99 s.

## 4.5 Feature-axis reduction: 48 runs

Final Figure 11 uses the selected 2-block, 25-branch model at matched seeds
42, 43, and 44. There are 16 feature-reduction conditions (5 PCA dimensions and
11 linear dimensions), so this family contains 16 × 3 = 48 runs. The unreduced
baseline for each seed is the corresponding 2-block, K=25 run from Section 4.1
(seed 42) or Section 4.2 (seeds 43 and 44) and is not repeated here.

### PCA reductions

The PCA files from Section 2.1 must exist before these runs are launched.
PCA dimensions are 16, 32, 64, 128, and 256.

```bash
for seed in 42 43 44; do
  for dim in 16 32 64 128 256; do
    run_paper_model \
      "pca${dim}-2conv-seed${seed}-paper" \
      2 25 "$seed" \
      --feature-reduction pca \
      --feature-reduction-dim "$dim"
  done
done
```

### Trainable linear reductions

LinearDR dimensions are 1, 2, 4, 8, 12, 16, 32, 64, 128, 256, and 768.

```bash
for seed in 42 43 44; do
  for dim in 1 2 4 8 12 16 32 64 128 256 768; do
    run_paper_model \
      "lineardr${dim}-2conv-seed${seed}-paper" \
      2 25 "$seed" \
      --feature-reduction linear \
      --feature-reduction-dim "$dim"
  done
done
```

The MEG and audio linear feature mappings are separate trainable modules; they
do not share weights.

## 4.6 Time-axis reduction and pooling: 15 runs

All time-axis reduction and pooling experiments use the selected 2-block,
25-branch model with seed 42. These 15 runs are the complete time-axis family
in the final matrix; they are not repeated for seeds 43–47. The corresponding
baseline is the seed-42 2-block, 25-branch run from Section 4.1 and is not
repeated here.

### PCA and trainable linear time reductions

```bash
for dim in 32 64; do
  run_paper_model \
    "timepca${dim}-2conv-seed42-paper" \
    2 25 42 \
    --time-reduction pca \
    --time-reduction-dim "$dim"
done

for dim in 32 64; do
  run_paper_model \
    "timelineardr${dim}-2conv-seed42-paper" \
    2 25 42 \
    --time-reduction linear \
    --time-reduction-dim "$dim"
done
```

### Pooling and attention reductions

```bash
for dim in 8 32 64; do
  run_paper_model \
    "timepool-adaptiveavg-k${dim}-2conv-seed42-paper" \
    2 25 42 \
    --time-reduction adaptive_avg \
    --time-reduction-dim "$dim"
done

run_paper_model \
  "timepool-adaptivemax-k8-2conv-seed42-paper" \
  2 25 42 \
  --time-reduction adaptive_max \
  --time-reduction-dim 8

run_paper_model \
  "timepool-attention-h128-2conv-seed42-paper" \
  2 25 42 \
  --time-reduction attn \
  --time-reduction-hidden-dim 128

run_paper_model \
  "timepool-gatedattention-h128-2conv-seed42-paper" \
  2 25 42 \
  --time-reduction gated_attn \
  --time-reduction-hidden-dim 128

run_paper_model \
  "timepool-max-2conv-seed42-paper" \
  2 25 42 \
  --time-reduction max

run_paper_model \
  "timepool-mean-2conv-seed42-paper" \
  2 25 42 \
  --time-reduction mean

for dim in 8 32 64; do
  run_paper_model \
    "timepool-queryattention-k${dim}-heads4-2conv-seed42-paper" \
    2 25 42 \
    --time-reduction query_attn \
    --time-reduction-dim "$dim" \
    --time-reduction-num-heads 4
done
```

The training code automatically disables strict deterministic-algorithm mode
for adaptive average and adaptive max pooling because their CUDA backward paths
are incompatible with the requested deterministic setting. This normalized
value is written to `config.json`.

## 4.7 Segment-duration runs: 15 runs

The duration figure compares five segment lengths at temporal depths 0, 2, and
5, all with seed 42 and K=25. These 15 runs are the complete segment-duration
family. The matrix counts that single reproduction path once. The run-group
suffix and `--window-tag` use the neutral duration tags created in Section 2.2:

```bash
WINDOW_TAGS=(w1p5 w2p25 w3 w4 w5)

for depth in 0 2 5; do
  for tag in "${WINDOW_TAGS[@]}"; do
    run_paper_model \
      "${depth}conv-seed42-paper-${tag}" \
      "$depth" 25 42 \
      --window-tag "$tag"
  done
done
```

# 5. Audit and sanity checks

## 5.1 Build a run registry

```bash
lisa-scan-runs \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --out outputs/experiment_registry.csv
```

The final paper matrix should contain 320 rows when all families above were run
in an initially empty experiment root. Reused conditions are present once.

A direct provenance check is:

```bash
printf 'cli.txt:     '
find "$EXPERIMENTS_ROOT" -mindepth 3 -maxdepth 3 -name cli.txt | wc -l
printf 'config.json: '
find "$EXPERIMENTS_ROOT" -mindepth 3 -maxdepth 3 -name config.json | wc -l
printf 'metrics.csv: '
find "$EXPERIMENTS_ROOT" -mindepth 3 -maxdepth 3 -name metrics.csv | wc -l
```

Each count should be 320 for the complete paper reproduction.

## 5.2 Export model parameter counts

Build a per-run capacity-and-performance table from the normalized
`config.json` and `metrics.csv` files:

```bash
lisa-export-parameter-counts \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --out outputs/model_parameter_counts.csv \
  --strict
```

The command reconstructs the configured MEG and audio modules on CPU without
loading their checkpoints. It writes one row per discovered `config.json`;
`status`/`error` cover parameter reconstruction and
`metrics_status`/`metrics_error` independently cover metric extraction.
`--strict` returns a non-zero exit status after writing the CSV if either part
is unavailable for any row.

The main parameter columns are:

| Columns | Meaning |
|---|---|
| `trainable_params`, `total_params` | Complete inference network: MEG model plus audio adapter; excludes the loss criterion. |
| `meg_model_*` | Complete `LISA` module, including the projection head and optional MEG time reducer. |
| `meg_core_*` | MEG model after excluding `feature_projection` and the explicit `time_reducer`. |
| `meg_projection_resampler_*` | The `ConvHead` projection/resampling head. Its convolution jointly changes feature channels and, when strided, temporal resolution, so its weights are not split between those two operations. |
| `meg_time_reducer_*` | Explicit MEG-side `TimeReduction` applied after the projection head. |
| `audio_adapter_*` | Complete trainable/non-trainable audio-side adapter. |
| `audio_feature_reducer_*`, `audio_time_reducer_*`, `audio_other_*` | Non-overlapping partition of the audio adapter. |
| `criterion_*` | Learnable CLIP-temperature parameter, reported separately from the inference network. |
| `optimizer_trainable_params` | Network trainable parameters plus the learnable criterion parameters. |
| `best_epoch` | Epoch selected by minimum validation loss and used for the final test evaluation. |
| `loss_final_test`, `top1s_final_test`, `top10s_final_test` | Raw final-test metrics logged after restoring the best-validation MEG, audio, and criterion checkpoints. Accuracies are fractions in `[0, 1]`. |
| `top1_final_test_pct`, `top10_final_test_pct` | The same final-test accuracies expressed as percentages for tables and plots. |
| `metrics_status`, `metrics_error`, `metrics_path` | Metric extraction audit fields; auxiliary per-epoch test values are never substituted for a missing final-test row. |

Every `*` component has separate `*_trainable_params` and `*_total_params`
columns. The component partitions are non-overlapping and sum to their
corresponding MEG, audio, and full-network totals.

Use `model_family`, `model_subfamily`, `ablation`, `architecture_id`,
`architecture_label`, and the copied architecture fields such as
`n_temporal_module_blocks`, `n_channels_unmix`, `feature_reduction`, and
`time_reduction` to filter or deduplicate rows for paper tables.
`architecture_id` intentionally ignores run identity, seed, and window tag.
For a direct capacity comparison, plot `trainable_params` or `total_params`
against `top1_final_test_pct`/`top10_final_test_pct`, filtering by
`model_family` or `model_subfamily` as needed.

Precomputed feature/time PCA is an offline data transform rather than a
PyTorch module in the trained inference network. It therefore contributes zero
module parameters; affected rows are marked with
`audio_precomputed_reduction=true`.

## 5.3 Selected-model sanity values

The six reported runs of the selected 2-block, 25-branch model produced the
following final-test values. Small floating-point variation is possible across
PyTorch versions, drivers, hardware, and numerical backends; these numbers are
intended as sanity checks rather than byte-level acceptance tests.

| Seed | Best validation epoch | Top-1 (%) | Top-10 (%) |
|---:|---:|---:|---:|
| 42 | 9 | 40.008 | 70.598 |
| 43 | 8 | 39.890 | 70.387 |
| 44 | 5 | 39.192 | 70.003 |
| 45 | 7 | 39.959 | 70.399 |
| 46 | 7 | 39.986 | 70.858 |
| 47 | 7 | 39.464 | 70.151 |

Across the six seeds, Top-1 was **39.750 ± 0.340%** and Top-10 was
**70.399 ± 0.306%** (mean ± sample SD).

## 5.4 Aggregate the six selected-model seeds

The following script reads the final-test row associated with each run's
validation-selected checkpoint:

```bash
python - <<'PY'
from pathlib import Path
import json
import pandas as pd

root = Path('outputs/experiments')
rows = []
for seed in range(42, 48):
    group = root / f'2conv-seed{seed}-paper'
    matches = []
    for config_path in group.glob('*/config.json'):
        config = json.loads(config_path.read_text())
        if int(config['n_channels_unmix']) == 25:
            matches.append(config_path.parent)
    if len(matches) != 1:
        raise RuntimeError(f'Expected one K=25 run in {group}, found {len(matches)}')
    metrics = pd.read_csv(matches[0] / 'metrics.csv')
    final = metrics.dropna(subset=['top1s_final_test', 'top10s_final_test']).iloc[-1]
    rows.append({
        'seed': seed,
        'best_epoch': int(final['epoch']),
        'top1_pct': 100 * final['top1s_final_test'],
        'top10_pct': 100 * final['top10s_final_test'],
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print('\nMean and sample SD across seeds:')
print(df[['top1_pct', 'top10_pct']].agg(['mean', 'std']))
PY
```

# 6. Reproduce tables and figures

Several analysis commands require the selected model checkpoint. The main
interpretation model is the seed-42, 2-block, 25-branch run.

## 6.1 Resolve the main run identifier

The helper below works for local, online ClearML, and ClearML offline directory
names:

```bash
MAIN_RUN_GROUP="2conv-seed42-paper"
MAIN_RUN_DIR=$(find \
  "$EXPERIMENTS_ROOT/$MAIN_RUN_GROUP" \
  -mindepth 1 -maxdepth 1 -type d \
  -name '*-25branches_*' \
  -print -quit)

if [[ -z "$MAIN_RUN_DIR" ]]; then
  echo "Could not find the main K=25 run" >&2
  exit 1
fi

MAIN_BASENAME=$(basename "$MAIN_RUN_DIR")
if [[ "$MAIN_BASENAME" == *_offline-* ]]; then
  MAIN_RUN_ID="${MAIN_BASENAME##*_offline-}"
else
  MAIN_RUN_ID="${MAIN_BASENAME##*_}"
fi

printf 'Main run directory: %s\n' "$MAIN_RUN_DIR"
printf 'Main run ID:        %s\n' "$MAIN_RUN_ID"
```

If multiple matching K=25 runs exist in the group, select the intended one
explicitly instead of relying on `find -print -quit`.

## 6.2 Branch-count × temporal-depth figure

The branch-count × temporal-depth figure uses all six seed-42 run groups:

```bash
lisa-plot-metrics \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --run-group \
    0conv-seed42-paper \
    1conv-seed42-paper \
    2conv-seed42-paper \
    3conv-seed42-paper \
    4conv-seed42-paper \
    5conv-seed42-paper
```

The default output is written under:

```text
outputs/plots/unmix_k_analysis/
```

With `--epochs best`, the plotted values are final-test metrics from the
validation-selected checkpoint. Numeric epoch values plot validation metrics.

## 6.3 Retrieval architecture and reduction tables

`lisa-plot-retrieval-ablations` pairs each condition with the same-seed
2-block, 25-branch model. Pass `--seeds` to expand `{seed}` in the run-group
templates. The three families use different seed sets, so they are plotted
separately. Each command writes `all_results.csv`, `config_diffs.csv`, and,
when `--seeds` is set, `summary_results.csv` with the mean, minimum, and
maximum across those seeds.

Architecture ablations, seeds 42–47:

```bash
lisa-plot-retrieval-ablations \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --baseline-run-group '2conv-seed{seed}-paper' \
  --n-branches 25 \
  --seeds 42 43 44 45 46 47 \
  --out-dir outputs/plots/retrieval_reductions_ablations/architecture-seeds42-47-25branches \
  --ablation-run-groups \
    'ablate-2dfull-2conv-seed{seed}-paper' \
    'ablate-no3d-2conv-seed{seed}-paper' \
    'ablate-nosubject-2conv-seed{seed}-paper' \
    'ablate-nounmix-2conv-seed{seed}-paper' \
    'ablate-only2d-2conv-seed{seed}-paper' \
    'ablate-only3d-2conv-seed{seed}-paper' \
    'ablate-onlysubject-2conv-seed{seed}-paper' \
    'ablate-onlyunmix-2conv-seed{seed}-paper' \
    'ablate-tfkernel1-2conv-seed{seed}-paper'
```

Feature-axis reductions, seeds 42–44:

```bash
lisa-plot-retrieval-ablations \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --baseline-run-group '2conv-seed{seed}-paper' \
  --n-branches 25 \
  --seeds 42 43 44 \
  --out-dir outputs/plots/retrieval_reductions_ablations/feature-seeds42-44-25branches \
  --feature-run-groups \
    'pca16-2conv-seed{seed}-paper' \
    'pca32-2conv-seed{seed}-paper' \
    'pca64-2conv-seed{seed}-paper' \
    'pca128-2conv-seed{seed}-paper' \
    'pca256-2conv-seed{seed}-paper' \
    'lineardr1-2conv-seed{seed}-paper' \
    'lineardr2-2conv-seed{seed}-paper' \
    'lineardr4-2conv-seed{seed}-paper' \
    'lineardr8-2conv-seed{seed}-paper' \
    'lineardr12-2conv-seed{seed}-paper' \
    'lineardr16-2conv-seed{seed}-paper' \
    'lineardr32-2conv-seed{seed}-paper' \
    'lineardr64-2conv-seed{seed}-paper' \
    'lineardr128-2conv-seed{seed}-paper' \
    'lineardr256-2conv-seed{seed}-paper' \
    'lineardr768-2conv-seed{seed}-paper'
```

Time-axis reductions stay at the 15 seed-42 runs from Section 4.6:

```bash
TIME_REDUCTIONS_2CONV=(
  timepca32-2conv-seed42-paper
  timepca64-2conv-seed42-paper
  timelineardr32-2conv-seed42-paper
  timelineardr64-2conv-seed42-paper
  timepool-mean-2conv-seed42-paper
  timepool-max-2conv-seed42-paper
  timepool-adaptiveavg-k8-2conv-seed42-paper
  timepool-adaptiveavg-k32-2conv-seed42-paper
  timepool-adaptiveavg-k64-2conv-seed42-paper
  timepool-adaptivemax-k8-2conv-seed42-paper
  timepool-queryattention-k8-heads4-2conv-seed42-paper
  timepool-queryattention-k32-heads4-2conv-seed42-paper
  timepool-queryattention-k64-heads4-2conv-seed42-paper
  timepool-attention-h128-2conv-seed42-paper
  timepool-gatedattention-h128-2conv-seed42-paper
)

lisa-plot-retrieval-ablations \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --baseline-run-group 2conv-seed42-paper \
  --n-branches 25 \
  --time-run-groups "${TIME_REDUCTIONS_2CONV[@]}"
```

The seed-42 time-reduction command writes under:

```text
outputs/plots/retrieval_reductions_ablations/2conv-seed42-paper-25branches/
```

Each command validates that runs in its family differ from the paired baseline
only in the configuration keys permitted for that family. It requires
`final_test_per_window.csv` in every selected run. Use `--overwrite` only when
intentionally replacing an existing non-empty output directory.

## 6.4 Segment-duration figure

The paper figure uses:

- depths: 0, 2, and 5 blocks;
- tags: `w1p5`, `w2p25`, `w3`, `w4`, and `w5`;
- durations: 1.5, 2.25, 3, 4, and 5 s;
- branches: 25;
- Acc@n curve: 2-block model through n=50.

The plotting CLI defaults to the same neutral tags. An equivalent explicit
invocation is:

```bash
lisa-plot-segment-duration-ablation \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --window-tags w1p5 w2p25 w3 w4 w5 \
  --window-seconds 1.5 2.25 3 4 5
```

Outputs are written under a parameter-specific directory in:

```text
outputs/plots/segment_duration_ablation/
```

This command also requires `final_test_per_window.csv` for all 15 selected
runs because the Acc@n curves are reconstructed from the true-candidate ranks.

## 6.5 Temporal-filter length ablation

The plotting command uses all fifteen kernel sizes listed in Section 4.4, each
at seeds 42, 43, and 44, for 45 seed-by-support conditions. That set includes
the reused 1-tap and 15-tap runs as well as the 39 additional training runs.
It validates the expected seed, architecture, sampling rate, test-window
identities, and temporal-filter shapes of every run before writing any results.

Start from an empty output directory so renamed or removed artifacts cannot
remain from an earlier version:

```bash
lisa-plot-temporal-filter-length-ablation \
  --experiments-root "$EXPERIMENTS_ROOT"
```

The current artifact set is:

```text
outputs/plots/temporal_filter_length_ablation/
├── temporal_filter_ablation_table.csv
├── temporal_filter_ablation_summary.csv
├── temporal_filter_ablation_table.txt
├── temporal_filter_subject_deltas.csv
├── temporal_filter_performance_delta.{pdf,png}
├── temporal_filter_subject_spread.{pdf,png}
├── temporal_filter_frequency_medians_raw.{pdf,png}
├── temporal_filter_frequency_medians_demeaned.{pdf,png}
├── temporal_filter_frequency_heatmaps_raw.{pdf,png}
└── temporal_filter_frequency_heatmaps_demeaned.{pdf,png}
```

The performance curve reports all-window Top-1 and Top-10 changes relative to
the 15-tap (150 ms) model of the same seed, on a logarithmic support axis. The
line is the mean over seeds 42–44 and the shaded band spans their minimum and
maximum. The paper reports a broad long-support regime: mean Top-1 is highest
at 2.51 s (251 taps), mean Top-10 is highest at 1.51 s (151 taps), and both
are lower again at 2.99 s (299 taps). The table CSV holds one row per run and
the summary CSV one row per kernel size, and both LaTeX tables list the
individual seed accuracies.

The subject-spread and frequency-response artifacts describe one trained model
each and use seed 42. The subject-spread panels show
per-subject changes, median/IQR summaries, and the all-window aggregate curve
on a shared vertical scale; these are descriptive summaries rather than
confidence intervals.

The frequency-response figures show peak-normalized kernel amplitude
`|H(f)|` from 0 to the 50-Hz Nyquist frequency, both for the learned kernels
and after subtracting each kernel's temporal mean. Those mean-subtracted
spectra are descriptive summaries written by this plotter. Canonical
physiological interpretation in Sections 6.7–6.10 uses the learned temporal
filters as trained. The 1-tap raw response is validated as
frequency-independent and its mean-subtracted kernel as identically zero, then
omitted from the spectral figures. Heatmap branches are sorted independently
within each model by spectral centroid and are not matched across filter
lengths.

## 6.6 Spatial-attention overview

The attention overview uses branch counts 5, 10, and 25 across all six temporal
depths:

```bash
lisa-plot-spatial-attention-maps \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --group-template '{blocks}conv-seed42-paper' \
  --blocks 0 1 2 3 4 5 \
  --branches 5 10 25
```

The command loads one clean MEG recording only to obtain sensor positions. It
requires the corresponding checkpoints for all 18 selected runs.

## 6.7 Spatial-filter and spatial-pattern SVD

The final-paper command uses all 27 participants, session 0, and story 1. It
interprets temporal filters as trained and sets the Figure 6b RAP-MUSIC
subspace threshold to 0.98. The SVD cumulative-energy threshold stays at the
CLI default of 0.95. The CLI default for `--dipole-thr-music` is 0.98, and
the paper command sets that threshold explicitly.

```bash
lisa-plot-spatial-topography-svd \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --run-group "$MAIN_RUN_GROUP" \
  --run-id "$MAIN_RUN_ID" \
  --no-demean-temporal-filters \
  --dipole-thr-music 0.98 \
  --raw-bids-root data/MASC-MEG \
  --geometry-cache-dir outputs/source_geometry
```

The command loads normalized MEG from `data/preprocessed/meg/meg27_sr100.npz`
and restores physical sensor scaling with the sibling sidecar
`data/preprocessed/meg/meg27_sr100_scalers.npz` before constructing Haufe
patterns. Participant/session-specific KIT geometry is rebuilt from the raw
BIDS digitization files under `--raw-bids-root`. For participant `sub-XX`,
session `ses-Y`, and task `Z` those files are:

```text
data/MASC-MEG/sub-XX/ses-Y/meg/sub-XX_ses-Y_acq-ELP_headshape.pos
data/MASC-MEG/sub-XX/ses-Y/meg/sub-XX_ses-Y_acq-HSP_headshape.pos
data/MASC-MEG/sub-XX/ses-Y/meg/sub-XX_ses-Y_task-Z_markers.mrk
```

Each participant gets a coregistration and forward solution in the shared MNE
`fsaverage` anatomy. Cached head-to-`fsaverage` transforms are written under
`--geometry-cache-dir`. Group dipole coordinates use that common `fsaverage`
frame. The first source-space command may download the MNE `fsaverage`
dataset.

Outputs include raw and row-L2-normalized SVD topographies, singular values,
energy tables, component arrays, and RAP-MUSIC diagnostics. With
`--no-demean-temporal-filters`, the output directory name ends in `_raw_tf`.

## 6.8 Branch clustering and compact interpretation figures

Clustering and plotting are separate steps. The clustering command writes
assignments and diagnostics under `outputs/motif_clusters/`; the plotting
command consumes those assignments and writes summaries under
`outputs/plots/branch_interpretations/`.

Generate assignments for each training story. The final-paper commands pass
`--no-demean-temporal-filters` so clustering uses the learned temporal filters
as trained. Geometry flags match Section 6.7. The scaler sidecar
`meg27_sr100_scalers.npz` is required beside the preprocessed MEG array.

```bash
for story in 0 1 2 3; do
  lisa-combined-cluster-branch-interpretations \
    --experiments-root "$EXPERIMENTS_ROOT" \
    --run-group "$MAIN_RUN_GROUP" \
    --run-id "$MAIN_RUN_ID" \
    --session 0 \
    --story-id "$story" \
    --no-demean-temporal-filters \
    --raw-bids-root data/MASC-MEG \
    --geometry-cache-dir outputs/source_geometry
done
```

The combined clustering defaults to a minimum raw source-and-temporal
correlation of 0.25.

Render all main and residual clusters for an audit view:

```bash
for story in 0 1 2 3; do
  lisa-plot-branch-interpretations \
    --experiments-root "$EXPERIMENTS_ROOT" \
    --run-group "$MAIN_RUN_GROUP" \
    --run-id "$MAIN_RUN_ID" \
    --session 0 \
    --story-id "$story" \
    --no-demean-temporal-filters
done
```

For the compact paper visualization, render the 12 largest main clusters. Main
cluster identifiers are assigned in descending cluster-size order, so `C1`
through `C12` are selected by a deterministic, non-manual criterion:

```bash
lisa-plot-branch-interpretations \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --run-group "$MAIN_RUN_GROUP" \
  --run-id "$MAIN_RUN_ID" \
  --session 0 \
  --story-id 1 \
  --clusters C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 \
  --no-demean-temporal-filters
```

Plotting must use the same temporal-filter setting as clustering so it reads
the `_raw_tf` assignments. The optional demeaning flag remains available; these
paper commands leave the learned filters unchanged.

## 6.9 Occlusion-feature analysis and seed robustness

Paired MEG occlusion requires the linguistic feature table from Section 2.3 and
the six selected-model checkpoints from Sections 4.1–4.2 (2-block, 25-branch,
seeds 42–47). Resolve each run the same way as in Section 6.1, then run the
analysis once per trained model. Keep the occlusion donor-selection and
permutation seed fixed at 42 for every model so only the checkpoint changes:

```bash
OCCLUSION_SEED=42

for model_seed in 42 43 44 45 46 47; do
  RUN_GROUP="2conv-seed${model_seed}-paper"
  RUN_DIR=$(find \
    "$EXPERIMENTS_ROOT/$RUN_GROUP" \
    -mindepth 1 -maxdepth 1 -type d \
    -name '*-25branches_*' \
    -print -quit)

  if [[ -z "$RUN_DIR" ]]; then
    echo "Could not find the K=25 run in $RUN_GROUP" >&2
    exit 1
  fi

  BASENAME=$(basename "$RUN_DIR")
  if [[ "$BASENAME" == *_offline-* ]]; then
    RUN_ID="${BASENAME##*_offline-}"
  else
    RUN_ID="${BASENAME##*_}"
  fi

  lisa-occlusion-feature-analysis \
    --experiments-root "$EXPERIMENTS_ROOT" \
    --run-group "$RUN_GROUP" \
    --run-id "$RUN_ID" \
    --seed "$OCCLUSION_SEED"
done
```

Without `--seed`, the CLI would reuse each run's training seed. Pass
`--seed 42` explicitly so donor pairs and permutations stay identical across
model seeds.

Each model seed writes under:

```text
outputs/plots/occlusion_feature_analysis/<run-group>/<run-name>-<run-id>/
├── feature_effects.pdf
├── feature_summary.csv
├── combo_metrics.csv
├── participant_metrics.csv
├── window_eligibility.csv
├── stats.json
└── COMPLETED.json
```

Use CUDA when available. Lower `--inference-batch-size` if the default MEG
forward batch does not fit in memory. The analysis is expensive because it
evaluates many masked donor interventions with a large permutation count
(`--n-permutations`, default 100000).

Aggregate across the six completed seed directories:

```bash
lisa-plot-occlusion-seed-robustness \
  '2conv-seed4[2-7]-paper'
```

Main robustness outputs:

```text
outputs/plots/occlusion_seed_robustness/2conv-seed4[2-7]-paper/
├── large_effects_robustness.pdf
└── smaller_effects_robustness.pdf
```

The headline single-seed `feature_effects` figure used in the manuscript is the
`feature_effects.pdf` from one completed model-seed directory above; the
robustness panels summarize the same feature battery across model seeds 42–47
with a shared occlusion seed of 42.

## 6.10 Interpretation reproducibility

Appendix D uses two separate raw/as-trained comparisons. Both read item
statistics written with `--no-demean-temporal-filters` (the `_raw_tf` suffix).
The default run-group template is `2conv-seed{seed}-paper`. These analyses are
not a Cartesian product of every seed with every story.

Story and repeated-session stability uses seed 42, stories 0–3, and sessions 0
and 1. Section 6.8 already clusters seed 42, session 0, stories 0–3. Add
session 1 for those stories:

```bash
for story in 0 1 2 3; do
  lisa-combined-cluster-branch-interpretations \
    --experiments-root "$EXPERIMENTS_ROOT" \
    --run-group "$MAIN_RUN_GROUP" \
    --run-id "$MAIN_RUN_ID" \
    --session 1 \
    --story-id "$story" \
    --no-demean-temporal-filters \
    --raw-bids-root data/MASC-MEG \
    --geometry-cache-dir outputs/source_geometry
done
```

```bash
lisa-analyze-interpretation-reproducibility \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --seeds 42 \
  --story-ids 0 1 2 3 \
  --sessions 0 1 \
  --compare stories sessions \
  --no-demean-temporal-filters
```

Cross-initialization stability uses seeds 42–47, story 1, and session 0. Seed
42, story 1, session 0 is already clustered in Section 6.8. Resolve seeds
43–47 the same way as in Section 6.9 and cluster story 1, session 0:

```bash
for seed in 43 44 45 46 47; do
  RUN_GROUP="2conv-seed${seed}-paper"
  RUN_DIR=$(find \
    "$EXPERIMENTS_ROOT/$RUN_GROUP" \
    -mindepth 1 -maxdepth 1 -type d \
    -name '*-25branches_*' \
    -print -quit)

  if [[ -z "$RUN_DIR" ]]; then
    echo "Could not find the K=25 run in $RUN_GROUP" >&2
    exit 1
  fi

  BASENAME=$(basename "$RUN_DIR")
  if [[ "$BASENAME" == *_offline-* ]]; then
    RUN_ID="${BASENAME##*_offline-}"
  else
    RUN_ID="${BASENAME##*_}"
  fi

  lisa-combined-cluster-branch-interpretations \
    --experiments-root "$EXPERIMENTS_ROOT" \
    --run-group "$RUN_GROUP" \
    --run-id "$RUN_ID" \
    --session 0 \
    --story-id 1 \
    --no-demean-temporal-filters \
    --raw-bids-root data/MASC-MEG \
    --geometry-cache-dir outputs/source_geometry
done
```

```bash
lisa-analyze-interpretation-reproducibility \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --seeds 42 43 44 45 46 47 \
  --story-ids 1 \
  --sessions 0 \
  --compare seeds \
  --no-demean-temporal-filters
```

Outputs go to `outputs/plots/analyze_interpretation_reproducibility/`.
Optional `--haufe-geometry` adds spatial roughness of the physical-unit Haufe
patterns.

# 7. Reproducibility notes

## Validation and test use

The validation split is the complete `(story_id=3, sound_id=4)` pair from the
non-test dataframe. The best checkpoint is selected only by validation loss.
The test bank consists of all 1,005 canonical test candidates, or all 991
common-anchor candidates in the duration experiment.

Auxiliary test values are visible in `metrics.csv` at each epoch but do not
change model weights, early stopping, or the saved checkpoint. Final test
metrics are recalculated after reloading the best-validation model.

## Seeds

The branch-count × temporal-depth sweep uses seed 42. The selected 2-block,
K=25 model adds seeds 43–47, giving the six main seeds 42–47. Architecture
ablations use those same six seeds. Feature-axis reductions and the
temporal-filter-support sweep use seeds 42–44. Time-axis reductions and
segment-duration runs stay at seed 42. Every run resets Python, NumPy, and
PyTorch RNG state independently, so the order in which the shell loops are
executed does not affect their configured seed.

## Determinism

`--torch-deterministic` requests deterministic PyTorch algorithms and is part of
the paper commands above. Exact byte-level equality still depends on the software
and hardware stack. Adaptive pooling runs explicitly normalize this option to
`false` because the corresponding CUDA backward operation is unsupported in
strict deterministic mode.

## Clean MEG preprocessing

The reported runs use the released per-recording ICA solutions and cleaned FIF
files. To reproduce the same input variant, use `clean` in the preparation
scripts and retain the same ICA files. Training itself loads normalized MEG
tensors from `data/preprocessed/meg/meg27_sr100.npz`. Physiological and source
interpretation restore physical sensor scaling with
`data/preprocessed/meg/meg27_sr100_scalers.npz`. `--meg-files-dir` is used by
training-time and post-hoc spatial visualizations to obtain sensor metadata.

## Source-space visualizations

Source estimates and RAP-MUSIC plots use participant- and session-specific
KIT sensor geometry from the raw BIDS ELP, HSP, and MRK files. Each
participant has a coregistration and forward solution. Group coordinates are
reported in the shared MNE `fsaverage` anatomy. Patterns are built after
restoring physical sensor scaling from `meg27_sr100_scalers.npz`. The
final-paper RAP-MUSIC acceptance threshold is 0.98, and the canonical
physiological commands pass `--no-demean-temporal-filters`. See Sections 6.7
and 6.8.

## Existing output directories

Several plotting commands fail rather than silently overwrite a non-empty
output directory. Use a fresh output root, delete the previous generated
artifacts, or pass an explicit supported overwrite flag when replacement is
intentional.

# 8. Minimal reproduction paths

A complete reproduction of the final paper matrix requires all 320 runs. For
narrower checks:

### Main numerical result

```bash
run_paper_model "2conv-seed42-paper" 2 25 42
```

### Six-seed robustness of the main model

Run the main seed-42 command above and the five additional 2-block commands from
Section 4.2.

### Main interpretation figures

Only the main 2-block, 25-branch checkpoint is required for Sections 6.6–6.8,
except that the spatial-attention overview additionally loads 18 checkpoints
from the branch/depth sweep. Sections 6.7 and 6.8 also need the BIDS
digitization files and `meg27_sr100_scalers.npz`. Section 6.10 also needs
raw/as-trained item statistics for seed 42, stories 0–3, sessions 0 and 1,
and for seeds 43–47, story 1, session 0.

### Occlusion-feature figures

Precompute word features (Section 2.3), then run Section 6.9 on the six
selected-model seeds. Seed robustness needs all six occlusion directories.

### Main retrieval ablation figure

Run the 2-block baselines reused by Sections 4.3, 4.5, and 4.6, plus the 54
architecture ablations, 48 feature-axis reductions, and 15 time-axis
reductions those sections train. PCA preprocessing must be completed first.
