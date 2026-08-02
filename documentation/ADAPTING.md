# Adapting LISA for your own experiments

This repository is built around the MASC-MEG retrieval pipeline, but the LISA
model itself is a reusable MEG/EEG encoder. Use this page when you want to run
your own experiments first, and only consult
[`REPRODUCE.md`](REPRODUCE.md) when you need the paper matrix.

## Two ways to use the code

### 1. Own runs on the prepared MASC-MEG pipeline

If you follow the README quickstart through data preparation, you already have
the canonical 3 s inputs. Train any configuration you like by choosing a new
`--run-group` and the options documented in
[`HYPERPARAMETERS.md`](HYPERPARAMETERS.md):

```bash
conda activate LISA

lisa-train \
  --run-group my-2conv-k25-seed7 \
  --seed 7 \
  --n-temporal-module-blocks 2 \
  --n-channels-unmix 25 \
  --n-channels-block 25
```

Useful patterns:

- Keep `--run-group` short and path-safe; it becomes the experiment folder name.
- Change one factor at a time (branches, depth, spatial attention, temporal
  filter length, seed) so run directories stay comparable.
- Inspect results with `lisa-scan-runs` and the plotting CLIs listed in the
  README.
- Every completed run stores `cli.txt`, `config.json`, `metrics.csv`, and
  checkpoints under `outputs/experiments/<run-group>/…`.

The paper command loops in [`REPRODUCE.md`](REPRODUCE.md) are optional recipes,
not required entry points.

### 2. Reuse LISA as an encoder on a new task

The `LISA` module in `src/lisa/model/nn_modules.py` maps

```text
(meg, subject_idx) → (interpretable_repr, embedding)
```

You can attach your own head (classification, regression, sequence prediction,
or a different retrieval objective). The synthetic walkthrough is:

[`notebooks/demo.ipynb`](../notebooks/demo.ipynb)

That notebook does not use MASC-MEG, Wav2Vec2, or the contrastive trainer. It
builds a small labeled dataset, trains spatial/temporal filters with a separate
classification head, and plots filter frequency responses and spatial patterns.

For tensor shapes, front-end factorization, and interpretation helpers, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## What is reusable vs MASC-specific

| Reusable | Tied to this repository’s MASC-MEG defaults |
|---|---|
| `LISA` spatial/temporal front-end and temporal tail | 208-channel layout and tracked coords under `data/preprocessed/coords/` |
| Spatial-attention / unmixing / subject-layer options | `n_subjects=27` and subject indices in `[0, 26]` |
| Temporal-filter and block configuration | 100 Hz MEG windows and the default 3 s / 1 s-stride chunking |
| Interpretation utilities that take a checkpoint + MEG | OSF download, MFA, ICA application, Wav2Vec2 embedding scripts |
| CLI training loop and local run layout | Contrastive audio-bank retrieval against Wav2Vec2 targets |

When porting to another dataset you typically must supply:

1. sensor coordinates compatible with the chosen spatial-attention mode;
2. a subject-index tensor if you keep the subject layer;
3. your own targets and loss (the bundled trainer assumes retrieval against an
   audio embedding bank).

Changing channel count, sampling rate, or subject count is supported by the
module API, but the end-to-end shell scripts and default paths still assume the
MASC-MEG layout described in the README.

## Suggested first experiments

1. Run the quickstart `lisa-train --run-group …` on the prepared data with a
   small change (for example a different seed or branch count).
2. Open `notebooks/demo.ipynb` to see filter readouts without the full dataset.
3. Only then scale to the families in [`REPRODUCE.md`](REPRODUCE.md) if you want
   the paper tables and figures.
