# LISA test suite

The tests in this directory protect the implementation details that are most
important for training, retrieval evaluation, model interpretation, and
reproduction of the reported experiments.

The suite is designed to run without the MASC-MEG dataset, released
checkpoints, OSF access, or network downloads. Tests use synthetic arrays,
small toy models, and temporary files. Model-forward tests run on CPU and use
real sensor coordinates when they are available, otherwise deterministic
synthetic coordinates are generated.

## Setup

Run the tests from the repository root in the main LISA environment.
PyTorch must be installed separately, as described in the root
[`README.md`](../README.md). Then install the package and development
dependencies in editable mode:

```bash
pip install -e ".[dev]"
```

No GPU is required for the test suite. The tests do not validate a complete
CUDA training run.

For headless systems, Matplotlib can be forced to use a non-interactive
backend:

```bash
export MPLBACKEND=Agg
```

## Running the tests

Run the complete suite:

```bash
pytest -q tests/
```

Show individual test names and reasons for skipped tests:

```bash
pytest -v -rs tests/
```

Stop after the first failure and use shorter tracebacks:

```bash
pytest -q -x --tb=short tests/
```

Report the slowest tests:

```bash
pytest -q --durations=20 tests/
```

Run one file, class, or individual test:

```bash
pytest -q tests/test_training_criteria.py

pytest -q \
  tests/test_lisa_variations.py::TestTemporalModuleBlocks

pytest -q \
  tests/test_training_data_validation.py::test_validation_split_holds_out_requested_story_sound
```

Use `-k` for a name-based subset:

```bash
pytest -q tests/ -k "retrieval or validation_split"
```

## Recommended workflows

### During routine development

The following subset covers the central training, retrieval, and reduction
invariants without running the large architecture grid:

```bash
pytest -q \
  tests/test_training_criteria.py \
  tests/test_trainer_outputs.py \
  tests/test_training_cli.py \
  tests/test_training_data_validation.py \
  tests/test_audio_reduction.py
```

This subset is useful for iteration, but it is not a replacement for the full
suite.

### Architecture regression checks

`test_lisa_variations.py` instantiates many representative model
configurations and performs real CPU forward passes:

```bash
pytest -q tests/test_lisa_variations.py
```

This file is substantially slower than the other tests because it contains a
large parametrized grid over spatial attention, FIR initialization, projection
heads, temporal depth, temporal-filter type, gating, and related combinations.

### Full check

Run the entire suite from a clean editable installation:

```bash
pytest -q tests/
```

Unexpected skips should be inspected with `pytest -rs`. A skipped test is not
equivalent to a successful check of the corresponding functionality.

## Test-suite map

| File | Main responsibility |
|---|---|
| `test_training_criteria.py` | Retrieval loss and metrics, candidate-ID handling, and fail-fast candidate-bank checks. |
| `test_trainer_outputs.py` | Tensor-to-NumPy ownership and per-window final-test output serialization. |
| `test_training_cli.py` | Removed CLI options, argument validation, semantic values, and paper-baseline defaults. |
| `test_training_data_validation.py` | Validation splitting and fail-fast audio-index checks. |
| `test_lisa_variations.py` | Model construction, forward passes, output shapes, interpretable/tail APIs, spatial and temporal filter extraction, and representative architecture combinations. |
| `test_audio_reduction.py` | Feature- and time-axis reduction parsing, PCA artifacts, audio-side reducers, invalid combinations, output-length checks, and determinism normalization for adaptive pooling. |
| `test_export_parameter_counts.py` | Run-family classification, parameter partitioning across model components, and stable architecture identifiers. |
| `test_cluster_branch_interpretations.py` | Combined branch-clustering validation, minimum-similarity enforcement, rough-pattern handling, and deterministic cluster ordering. |
| `test_plot_spatial_topography_svd.py` | SVD variants, temporal-filter demeaning, channel alignment, source-index bookkeeping, and expected publication-figure panels. |
| `test_plot_style.py` | Figure sizing, readable grid geometry, validation of overly dense layouts, and isolation of Matplotlib style changes. |
| `test_precompute_word_features.py` | Boundary handling for word-level language-model surprisal and entropy computation using a toy tokenizer and model. |

## What the tests verify

### Training and retrieval correctness

The suite checks that:

- duplicate positive audio IDs are collapsed into unique retrieval candidates;
- batches without at least two unique candidates fail explicitly;
- candidate IDs, rather than candidate-array positions, determine labels;
- Top-*k* metrics and saved rankings agree;
- invalid, missing, duplicated, or undersized candidate banks fail fast;
- final per-window outputs preserve dataset-row identity even when multiple
  rows share the same `wav_index`;
- the requested validation story/sound is held out and has no overlapping
  `wav_index` values with the fit subset;
- out-of-range audio indices are detected before training;
- current CLI defaults match the paper baseline.

### Model architecture

The architecture tests cover representative combinations of:

- no spatial attention, 2D Fourier attention, and 3D spherical-harmonic
  attention;
- random and FIR-initialized temporal filters;
- `single_conv`, `conv`, and internal `max` projection heads;
- zero to four temporal convolutional blocks;
- causal convolution and forward-backward (`filtfilt`) temporal filtering;
- global and branch-wise temporal-filter gates;
- optional GELU after the interpretable temporal filter;
- frozen and trainable FIR-initialized filters;
- separate `forward_interpretable` and `forward_tail` execution;
- spatial-filter and temporal-filter extraction;
- trainable time-axis reduction after the MEG projection head.

The large parametrized test is a representative regression grid, not an
exhaustive Cartesian product of every CLI parameter.

### Reduction pipelines

Reduction tests verify both configuration rules and tensor behavior:

- feature-axis and time-axis reductions cannot be enabled simultaneously;
- required dimensions and attention-specific options are validated;
- PCA preprocessing writes the expected transformed embeddings and artifacts;
- feature-PCA and time-PCA modes load the correct audio representation;
- trainable audio-side reductions produce the required output shapes;
- reducers reject requested output lengths larger than the input sequence;
- single-token reducers retain an explicit singleton time axis;
- adaptive pooling normalizes incompatible deterministic-algorithm settings
  with an explicit warning.

### Interpretation and statistical analysis

The interpretation tests lock down several paper-critical numerical details:

- clustering produces a complete, plot-compatible assignment of every item;
- source-space scans retain original source indices after invalid locations
  are removed.

## Test data and side effects

Tests should not:

- access the network;
- download pretrained models;
- require MASC-MEG data;
- modify files outside pytest-managed temporary directories;
- write into the experiment root;
- depend on a particular GPU.

`test_lisa_variations.py` reads
`data/preprocessed/coords/sensor_xyz.npy` and
`data/preprocessed/coords/coords208_xy_scaled.npy` when those files are
present. If they are absent, it generates deterministic synthetic coordinates,
so the test remains self-contained.

## What the suite does not establish

Passing the unit and regression tests does **not** by itself establish that:

- a full model can be trained end to end on the complete dataset;
- a CUDA run is numerically identical across hardware or software versions;
- preprocessing reproduces the released MEG files from raw data;
- reported paper metrics are reproduced exactly;
- every possible combination of all CLI arguments is valid;
- scientific conclusions follow solely from implementation correctness.

Exact reproduction of the experiment matrix and reported outputs is described
in [`documentation/REPRODUCE.md`](../documentation/REPRODUCE.md). The architecture and complete
CLI semantics are documented in
[`documentation/ARCHITECTURE.md`](../documentation/ARCHITECTURE.md) and
[`documentation/HYPERPARAMETERS.md`](../documentation/HYPERPARAMETERS.md).

## Adding or changing tests

When modifying correctness-critical code:

1. Add a regression test that fails on the previous implementation.
2. Prefer small deterministic arrays with an independently calculable expected
   result.
3. Use `tmp_path` for generated files and `monkeypatch` for external state.
4. Reuse production parser defaults when testing the default model rather than
   copying values into the test.
5. Test both valid behavior and the relevant fail-fast path.
6. Avoid network access, full datasets, and permanent files.
7. Keep test names specific enough to identify the invariant that failed.

Tests that change the public CLI, paper baseline, retrieval semantics, dataset
splits, checkpoint selection, or inferential statistics should be treated as
publication-critical and reviewed together with the corresponding
documentation.
