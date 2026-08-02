# LISA Training Hyperparameters

This document is the reference for the `lisa-train` command in the current
repository. It describes the effective behavior of every training CLI option,
including derived values, interactions between flags, and validation rules.

For the model equations and tensor flow, see [`ARCHITECTURE.md`](ARCHITECTURE.md).
Commands used to reproduce complete experiment families and figures belong in
[`REPRODUCE.md`](REPRODUCE.md).

> [!IMPORTANT]
> Defaults below are the defaults of the current CLI, not a promise that every
> default is appropriate for a new dataset. The current data pipeline is tied to
> 208-channel MASC-MEG data from 27 participants and 768-dimensional
> Wav2Vec2-base embeddings.

## CLI conventions

- `--run-group` is the only required argument.
- Argument abbreviation is disabled. Use complete option names.
- Boolean options implemented with `BooleanOptionalAction` accept both forms,
  for example `--tf-gated` and `--no-tf-gated`.
- `--seed none` and `--early-stopping-patience none` are accepted as special
  string values and normalized to `null` in `config.json`.
- `--use-spatial-attention none` is normalized to `null` in `config.json`.
- Several values are derived after parsing and saved in the run configuration;
  they are listed in [Derived configuration fields](#9-derived-configuration-fields).

## Reference configuration

The parser defaults correspond to the architecture of the main 2-block,
25-branch model:

| Component | Default |
|---|---:|
| Spatial attention | 3D spherical harmonics |
| Attention channels | 270 |
| Interpretable branches, \(K\) | 25 |
| Spatial harmonic order parameter | 24 |
| Shared unmixing | enabled |
| Subject-specific mapping | enabled |
| Temporal-filter kernel | 15 samples |
| Temporal-filter type | `conv` |
| Temporal blocks | 2 |
| Temporal-block width | `auto` → 25 |
| Projection head | `single_conv`, stride 2 |
| Output feature size | 768 |
| Feature/time reduction | disabled |

Runtime and logging choices such as deterministic execution, DataLoader worker
count, logger backend, and artifact upload are not part of the architecture and
may differ between runs.

# 1. Data, alignment, and split configuration

## Data-loading options

| Option | Default | Meaning and effective behavior |
|---|---:|---|
| `--dirprocess` | `data/preprocessed` | Root containing `audio/`, `dataframe/`, `meg/`, and `coords/`. Training MEG arrays, audio embeddings, metadata tables, and coordinate files are loaded from here. |
| `--meg-files-dir` | `data/clean` | Root used to load sensor information when plotting spatial filters during training. It is **not** the source of the MEG tensors used for optimization. |
| `--meg-format` | `fif` | Sensor-information format used by spatial-filter plotting: `fif` for cleaned FIF files or `bids` for raw MASC-MEG. |
| `--meg-sr` | `100` | MEG sampling rate in Hz. Selects `meg/meg27_sr<meg_sr>.npz` and the dataframe columns `meg<meg_sr>_start/stop`; it does not resample data at training time. |
| `--meg-offset` | `0.15` | Delay, in seconds, added to both MEG window boundaries relative to the audio window. It is converted once as `int(meg_offset * meg_sr)`, so fractional samples are truncated. |
| `--embedding-layers` | `4` | Selects precomputed files named `extract_features_<split><layers>...npy`. It does not recompute or average Wav2Vec2 layers during training. |
| `--window-tag` | empty | Selects alternative window-length files produced by `lisa-regen-windows`, for example `w5`. The tag must be alphanumeric. |

With the canonical settings, the main files are:

```text
<dirprocess>/audio/extract_features_train4.npy
<dirprocess>/audio/extract_features_test4.npy
<dirprocess>/dataframe/df_train27.csv
<dirprocess>/dataframe/df_test27.csv
<dirprocess>/meg/meg27_sr100.npz
<dirprocess>/coords/sensor_xyz.npy
<dirprocess>/coords/coords208_xy_scaled.npy
```

For `--window-tag w5`, the selected audio and dataframe files become:

```text
extract_features_train4_w5.npy
extract_features_test4_w5.npy
df_train27_w5.csv
df_test27_w5.csv
```

The same MEG NPZ is reused; the tagged dataframe supplies the alternative
window boundaries.

### Window tags and PCA

Tagged window datasets cannot currently be combined with
`--feature-reduction pca` or `--time-reduction pca`. The parser rejects this
combination because window-tagged PCA filenames are not generated or resolved
by the current pipeline.

## Validation split

| Option | Default | Meaning |
|---|---:|---|
| `--val-story-sounds` | `3:4` | Comma-separated `story_id:sound_id` pairs held out from the training dataframe as validation data, for example `3:4,2:7`. |

The split is performed by complete story/sound pairs. The code verifies that:

- every requested pair exists;
- train-fit and validation are both non-empty;
- their sets of `wav_index` values do not overlap.

The best checkpoint is selected by strictly lower validation loss. The test
split is not used in checkpoint selection, although auxiliary test metrics are
logged after each epoch by the current training loop.

## Batch construction

| Option | Default | Meaning |
|---|---:|---|
| `--batch-size` | `100` | Training batch size. Training is shuffled and uses `drop_last=True`. |
| `--test-batch-size` | `20` | Validation and test batch size. Evaluation is not shuffled and does not drop samples. |
| `--dl-n-workers` | `0` | Number of worker processes for every DataLoader. |

Runtime constraints:

- the fit-training set must contain at least `batch_size` rows;
- every CLIP-loss batch must contain at least two distinct `wav_index` values;
- validation and test must each contain at least two rows;
- validation and test sizes must not leave a final batch of size one;
- every evaluation candidate bank must contain at least 10 candidates because
  Top-10 is always computed.

# 2. Audio feature and time reduction

The unreduced audio arrays have shape `(N, T_audio, 768)` on disk and are
transposed to `(B, 768, T_audio)` before entering the audio-side adapter.
Feature-axis and time-axis reduction are mutually exclusive in a single run.

## Feature-axis reduction

| Option | Default | Allowed values |
|---|---:|---|
| `--feature-reduction` | `none` | `none`, `pca`, `linear` |
| `--feature-reduction-dim` | `null` | Positive integer required for `pca` or `linear`; must not exceed 768 |

### `none`

The MEG model outputs 768 channels and the audio adapter is an identity unless
a time reduction is active.

### `pca`

PCA-reduced audio arrays must first be generated with `lisa-precompute-pca`.
Training loads files with the suffix `_PCA<D>`, for example:

```text
extract_features_train4_PCA256.npy
extract_features_test4_PCA256.npy
```

The audio arrays already have `D` feature channels, so the audio adapter is an
identity. The MEG projection head is constructed to output the same `D`
channels. PCA parameters are fitted on the complete non-test `train` audio
array by the preprocessing utility and then applied to the test array.

### `linear`

The raw 768-dimensional audio arrays are loaded. Two separate trainable
mappings are learned:

- the MEG projection head outputs `D` channels;
- the audio adapter applies an independent learned `1×1` convolution from
  768 to `D` channels.

The two mappings do not share parameters.

## Time-axis reduction

| Option | Default | Allowed values |
|---|---:|---|
| `--time-reduction` | `none` | `none`, `pca`, `linear`, `mean`, `max`, `adaptive_avg`, `adaptive_max`, `attn`, `gated_attn`, `query_attn` |
| `--time-reduction-dim` | `null` | Required for `pca`, `linear`, `adaptive_avg`, `adaptive_max`, and `query_attn` |
| `--time-reduction-hidden-dim` | `null` | Optional only for `attn` and `gated_attn`; defaults internally to the embedding feature size |
| `--time-reduction-num-heads` | `1` | Used only by `query_attn`; must be positive |

The reducer is applied after the MEG projection head and, except for PCA, by a
separate audio-side module of the same type. Corresponding MEG and audio
reducers do **not** share parameters.

| Mode | Output time length | MEG side | Audio side |
|---|---:|---|---|
| `none` | unchanged | no reducer | no reducer |
| `pca` | `time_reduction_dim` | learned linear map over the MEG time axis | precomputed PCA-reduced audio; no trainable reducer |
| `linear` | `time_reduction_dim` | learned linear map over time | separate learned linear map over time |
| `mean` | 1 | arithmetic mean | arithmetic mean |
| `max` | 1 | maximum | maximum |
| `adaptive_avg` | `time_reduction_dim` | adaptive average pooling | separate adaptive average pooling |
| `adaptive_max` | `time_reduction_dim` | adaptive max pooling | separate adaptive max pooling |
| `attn` | 1 | additive attention pooling | separate additive attention pooling |
| `gated_attn` | 1 | gated attention pooling | separate gated attention pooling |
| `query_attn` | `time_reduction_dim` | learnable-query multi-head attention | separate learnable-query multi-head attention |

Additional rules:

- `time_reduction_dim` must not exceed the loaded audio sequence length for
  `linear`, adaptive pooling, or `query_attn`;
- `query_attn` requires the embedding feature size to be divisible by
  `time_reduction_num_heads`;
- `time_reduction_hidden_dim` is rejected for modes other than `attn` and
  `gated_attn`;
- a non-default `time_reduction_num_heads` is rejected unless the mode is
  `query_attn`;
- deterministic PyTorch execution is automatically disabled, with a warning,
  for `adaptive_avg` and `adaptive_max` because the corresponding backward
  operation is incompatible with the requested deterministic mode.

## Parameter-count export

The reduction options above are copied into the filterable output of
`lisa-export-parameter-counts`. The same row also contains the final test loss,
Top-1, and Top-10 measured after restoring the validation-selected checkpoint.
The command reports the MEG core,
projection/resampling head, explicit MEG time reducer, audio feature reducer,
and audio time reducer separately. These CSV fields are derived audit metadata,
not additional training hyperparameters. See
[`REPRODUCE.md`](REPRODUCE.md#52-export-model-parameter-counts) for the command
and column reference, and
[`ARCHITECTURE.md`](ARCHITECTURE.md#parameter-count-partition) for the exact
module boundaries.

# 3. Spatial front-end

## Spatial dimensions and optional layers

| Option | Default | Meaning |
|---|---:|---|
| `--n-channels-attention` | `270` | Number of intermediate attention channels, \(A\). When attention is disabled but both unmixing and subject layers remain enabled, this also remains the shared unmixing output width. |
| `--n-channels-unmix` | `25` | Number of interpretable branches, \(K\), and the required channel count before the depthwise temporal filter. This is the branch count used in experiment names and sweeps. |
| `--use-spatial-attention` | `3D` | `3D`, `2D`, or explicit `none`. Omitting this option uses `3D`; omission does not disable attention. |
| `--use-unmixing-layer` / `--no-use-unmixing-layer` | enabled | Enables or disables the shared biased `1×1` unmixing convolution. |
| `--use-subject-layer` / `--no-use-subject-layer` | enabled | Enables or disables the subject-specific bias-free linear mapping. |

The spatial stage must output exactly `n_channels_unmix` channels before the
temporal filter. The supported mappings are:

| Unmixing | Subject layer | Effective channel path |
|---|---|---|
| enabled | enabled | active input → `n_channels_attention` → `n_channels_unmix` |
| enabled | disabled | active input → `n_channels_unmix` |
| disabled | enabled | active input → `n_channels_unmix` |
| disabled | disabled | identity; active input channels must already equal `n_channels_unmix` |

Here, the active input has `n_channels_attention` channels when spatial
attention is enabled and 208 channels when it is disabled. Consequently, when
both learned linear layers are disabled:

- with attention enabled, `n_channels_attention` must equal
  `n_channels_unmix`;
- with attention disabled, `n_channels_unmix` must equal 208.

## Coordinate basis

| Option | Default | Meaning |
|---|---:|---|
| `--n-spatial-harmonics` | `24` | Spatial basis order/size parameter, \(H\). It is **not** the branch count \(K\). |

For `3D`, the implementation uses real spherical-harmonic terms up to degree
`H - 1`, producing `H²` coefficient positions per attention channel. For `2D`,
it uses sine and cosine terms over an `H × H` spatial-frequency grid, producing
`2H²` coefficients per attention channel.

## Spatial dropout

| Option | Default | Meaning |
|---|---:|---|
| `--spatial-dropout-number` | `0` | Number of independently sampled dropout centers per training forward pass. Zero disables spatial dropout. |
| `--spatial-dropout-radius` | `0.1` | Radius around each center in the normalized 2D sensor-coordinate space. Must be non-negative. |
| `--dropout-center-low` | `0.1` | Lower bound used for both coordinates of every sampled center. |
| `--dropout-center-high` | `0.9` | Upper bound; must be greater than the lower bound. |

Spatial dropout exists only inside the coordinate-based attention layer. Its
parameters have no effect when `--use-spatial-attention none` is used. During
training, a sensor mask sampled in the 2D layout is applied to all attention
channels before their sensor-wise softmax. It is disabled in evaluation mode.

# 4. Branch-wise temporal filter

| Option | Default | Meaning |
|---|---:|---|
| `--temporal-filter-type` | `conv` | `conv` applies one depthwise convolution; `filtfilt` applies the same convolution forward and backward. |
| `--temporal-filter-kernel-size` | `15` | Kernel length in MEG samples. At 100 Hz, 15 samples correspond to 150 ms. |
| `--temporal-filter-padding-mode` | `zeros` | `zeros`, `reflect`, `replicate`, or `circular`. |
| `--temporal-filter-bands` | `null` | Optional FIR band-pass initialization, formatted as `low-high,low-high` in Hz. |
| `--temporal-filter-freeze` / `--no-temporal-filter-freeze` | disabled | Freezes the temporal-filter weights **only when FIR bands are supplied and FIR initialization is executed**. |

The temporal filter is always present and depthwise: there is one bias-free
kernel per branch and no cross-branch mixing.

### FIR initialization

For example:

```bash
--temporal-filter-bands 1-4,4-8,8-12
```

uses SciPy `firwin` to initialize band-pass filters. If fewer bands than
branches are supplied, bands are cycled across branches. Frequencies must form
valid band-pass intervals for the configured `meg_sr` and kernel length;
invalid values are rejected by `firwin` at model construction.

`--temporal-filter-freeze` has no effect when
`--temporal-filter-bands` is omitted, because random temporal-filter
initialization does not invoke the FIR-freezing path.

### Kernel-size note

Use a positive odd kernel size, especially with `filtfilt`. The standard `conv`
path uses PyTorch `padding="same"`. The custom `filtfilt` implementation uses
symmetric padding `(kernel_size - 1) // 2`; with an even kernel this does not
preserve the original temporal length.

## Optional temporal gate and activation

| Option | Default | Meaning |
|---|---:|---|
| `--tf-gated` / `--no-tf-gated` | disabled | Enables residual interpolation between the unfiltered and filtered branch signals. |
| `--tf-gate-per-channel` / `--no-tf-gate-per-channel` | disabled | When gating is enabled, selects one gate per branch instead of one scalar shared by all branches. |
| `--tf-gate-init-alpha` | `0.05` | Initial interpolation coefficient before logit parameterization. Values are clamped internally to `[1e-6, 1 - 1e-6]`. |
| `--tf-gelu` / `--no-tf-gelu` | disabled | Applies GELU after temporal filtering or gated interpolation. |

With gating enabled,

```text
output = input + sigmoid(gamma) * (filtered - input)
```

`tf_gate_per_channel` and `tf_gate_init_alpha` have no effect when gating is
disabled. Gate parameters are optimized in the same special optimizer group as
the temporal-filter weights.

# 5. Temporal tail and projection head

## Temporal convolutional blocks

| Option | Default | Meaning |
|---|---:|---|
| `--n-temporal-module-blocks` | `2` | Number of shared nonlinear temporal blocks. Must be a non-negative integer. |
| `--n-channels-block` | `auto` | Width of every temporal block. `auto` is resolved to `n_channels_unmix`; otherwise provide a positive integer. |

When the block count is zero, the temporal module becomes an identity and the
projection head receives the interpretable front-end output directly. The
projection head still remains part of the model, so this setting is not a
literal removal of all post-front-end processing. If a nonmatching
`n_channels_block` is supplied with zero blocks, it is ignored with a warning.

## Projection head

| Option | Default | Allowed values and behavior |
|---|---:|---|
| `--head-pool` | `single_conv` | `single_conv`: one 3-tap strided projection to the embedding channels. `conv`: a 3-tap strided convolution to twice the tail width, GELU, then a `1×1` feature projection. |
| `--head-stride` | `2` | Positive temporal stride used by the downsampling layer in the projection head. |

Both head variants finish with batch normalization over the output feature
channels. The internal model class also implements a `max` head, but the
current `lisa-train` CLI intentionally exposes only `single_conv` and `conv`.

For input length `L`, the downsampling length is computed with the standard
1-D convolution/pooling formula. The implementation uses padding 0 when
`head_stride == 2` and padding 1 for every other stride. With the canonical
3-second MEG window (`L = 300`), kernel size 3, and stride 2, the output length
is 149.

# 6. Optimization and regularization

## Main optimizer

| Option | Default | Meaning |
|---|---:|---|
| `--optim` | `AdamW` | `Adam` or `AdamW`, using the PyTorch defaults for optimizer parameters not exposed by this CLI. |
| `--lr-fe` | `3e-4` | Base learning rate for the MEG model excluding the temporal-filter group, plus all trainable audio-adapter parameters. |
| `--weight-decay` | `0.0` | Base optimizer weight decay. It is coupled L2 regularization with `Adam` and decoupled weight decay with `AdamW`. |
| `--nepoch` | `50` | Maximum number of epochs. There is no learning-rate scheduler. |

## Learnable CLIP temperature

| Option | Default | Meaning |
|---|---:|---|
| `--clip-temperature` | `1.0` | Positive initial temperature. The criterion stores `log(temperature)` as a learnable scalar and divides cosine similarity by its exponential. |
| `--clip-temperature-lr` | `1e-3` | Learning rate for the temperature parameter. Its optimizer group always uses zero weight decay. |

`clip_temperature` must be strictly positive. The parser does not enforce this
before the logarithm is taken, so zero or negative values are invalid.

The training loss is one-directional MEG-to-audio cross-entropy. Repeated
`wav_index` values inside a batch share one deduplicated audio candidate. Each
batch must therefore contain at least two unique audio IDs.

## Temporal-filter optimizer group

| Option | Default | Meaning |
|---|---:|---|
| `--lr-temporal-mult` | `1.0` | Multiplies `lr_fe` for temporal-filter weights and, when present, temporal-gate parameters. |
| `--wd-temporal` | `null` | Weight decay for the same temporal group. `null` inherits `weight_decay`. |
| `--tf-smooth` | `0.0` | Coefficient of the mean squared second-difference penalty on temporal-filter kernels. |
| `--tf-unfreeze-epoch` | `null` | If set, freezes temporal-filter and gate parameters before training and enables them at this zero-based epoch index. |

The smoothness term is:

```text
mean((w[..., 2:] - 2*w[..., 1:-1] + w[..., :-2]) ** 2)
```

and is added to the optimization objective when `tf_smooth > 0` and the kernel
has at least three samples. The logged `loss_train` remains the contrastive loss
without the smoothness term, even though the penalty affects gradients.

### Freeze/unfreeze interactions

- `temporal_filter_freeze=true` freezes FIR-initialized filter weights at model
  construction; it does not by itself freeze a temporal gate.
- setting `tf_unfreeze_epoch` freezes both the temporal filter and gate before
  epoch 0, regardless of their initial state, then enables both when
  `epoch >= tf_unfreeze_epoch`;
- `tf_unfreeze_epoch=0` enables them at the beginning of the first epoch and is
  effectively equivalent to no staged freeze;
- a value greater than or equal to `nepoch` keeps them frozen for the complete
  run;
- when both FIR freezing and staged unfreezing are used, staged unfreezing takes
  precedence at the requested epoch.

## Early stopping

| Option | Default | Meaning |
|---|---:|---|
| `--early-stopping-patience` | `7` | Stop after this many consecutive epochs without a strictly lower validation loss. Use `none` to train all `nepoch` epochs. |

Epoch 0 is eligible to become the best checkpoint. The saved MEG model,
trainable audio adapter, and learnable temperature are all restored from the
same best-validation epoch before final test evaluation.

# 7. Randomness, determinism, and device

| Option | Default | Meaning |
|---|---:|---|
| `--seed` | `42` | Integer seed for Python, NumPy, PyTorch, CUDA, DataLoader shuffling, and coordinate-attention initialization. Use `none` to leave RNGs unseeded. |
| `--torch-deterministic` / `--no-torch-deterministic` | disabled | Enables deterministic PyTorch algorithms, deterministic cuDNN, disables cuDNN benchmarking, and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. |
| `--device` | `cuda` | `cpu` or a string beginning with `cuda`, such as `cuda:1`. CUDA availability is checked before training. |

A fixed seed controls stochastic initialization and sample order, but strict
reproduction also depends on the software stack and hardware. Deterministic
mode can be substantially slower. Using `--seed none --torch-deterministic`
keeps supported low-level operations deterministic but leaves RNG-driven
behavior unseeded and triggers a warning.

As noted above, adaptive average/max time reduction automatically disables
deterministic mode because its training backward path is incompatible with the
requested PyTorch setting.

# 8. Logging and artifacts

## Run identity and output location

| Option | Default | Meaning |
|---|---:|---|
| `--run-group` | required | Group identifier used as one directory level. It must start with a letter or digit and contain only letters, digits, `.`, `_`, or `-`; path separators and `..` are forbidden. |
| `--experiments-root` | `outputs/experiments` | Root directory for all training runs. |

The run name is derived automatically as:

```text
<run_group>-<n_channels_unmix>branches
```

and the local output directory is:

```text
<experiments_root>/<run_group>/<run_name>_<run_id>/
```

The run ID is an eight-character UUID for local logging or the ClearML task ID
for ClearML logging.

## Logger backend

| Option | Default | Meaning |
|---|---:|---|
| `--logger` | `local` | `local` writes only to disk; `clearml` enables local plus remote logging. |
| `--logging-project` | `LISA-MEG` | ClearML project name. Ignored by the local backend. |
| `--upload-metrics-csv` / `--no-upload-metrics-csv` | disabled | Uploads the final local `metrics.csv` as a ClearML artifact. |
| `--log-images-to-server` / `--no-log-images-to-server` | disabled | Reports generated figures to ClearML. Local figure files are written regardless. |

ClearML support is included by `scripts/setup_main_env.sh` via the optional
dependency. If you installed without extras, add it with:

```bash
pip install -e ".[clearml]"
```

Then pass `--logger clearml`.

## Figure and retrieval artifacts

| Option | Default | Meaning |
|---|---:|---|
| `--image-formats` | `png pdf` | One or more formats passed to Matplotlib `savefig`. When server image logging is enabled, PNG is added automatically if absent. |
| `--max-branches-to-plot` | `20` | Maximum number of branches included in per-epoch temporal and spatial filter figures. It does not limit the branches saved in filter arrays. |
| `--save-test-top-k` | `20` | Number of highest-ranked candidate IDs and similarities retained per test window in `final_test_per_window.npz`. Must not exceed the test candidate-bank size. |

`rank_of_true` is computed against the complete test candidate bank and is
saved independently of `save_test_top_k`, allowing any post-hoc Top-K accuracy
to be recovered. The CSV contains metadata, true rank/similarity, and the Top-1
prediction; the NPZ contains the requested Top-K arrays and candidate IDs.

Per-epoch temporal- and spatial-filter figures are disabled by default. When
`--plot-filter-graphs` is enabled, training generates them after every epoch;
spatial-filter figures are generated for all subjects. `max_branches_to_plot`
limits the number of branches in each figure, not the number of subject figures.

# 9. Derived configuration fields

The following keys appear in `config.json` but are not independent CLI
hyperparameters:

| Field | Derivation |
|---|---|
| `n_features` | 768 unless feature reduction is active; then equals `feature_reduction_dim`. |
| `n_channels_block` | `n_channels_unmix` when the CLI value is `auto`; otherwise the parsed positive integer. |
| `time_reduction_input_length` | For MEG-side `linear` or PCA matching, inferred from the actual MEG window length after the projection head. Otherwise `null`. |
| `audio_time_reduction_input_length` | For trainable audio-side `linear` time reduction, inferred from the loaded audio sequence length. Otherwise `null`. |
| `checkpoint` | Hardcoded to `lisa_checkpoint`. |
| `run_name` | `<run_group>-<n_channels_unmix>branches`. |
| `use_spatial_attention` | CLI string `none` is stored as `null`; `2D` and `3D` remain strings. |
| `device` | Converted to a `torch.device` during execution and serialized as a string by the logger. |

These derived values are required when reloading a model from `config.json` and
should not be edited independently of their source options.

# 10. Compatibility and validation summary

The current CLI rejects or fails fast on the following combinations:

- feature and time reduction both active;
- reduction dimension missing for a reduction that requires it;
- reduction dimension supplied to a mode that does not use it;
- attention hidden dimension supplied outside `attn`/`gated_attn`;
- non-default attention head count outside `query_attn`;
- feature-reduction dimension greater than 768;
- window tags containing separators or non-alphanumeric characters;
- window-tagged datasets combined with PCA reduction;
- both unmixing and subject layers disabled when the active spatial channel
  count differs from `n_channels_unmix`;
- negative spatial-dropout count or radius;
- invalid dropout center interval;
- non-positive `save_test_top_k` or early-stopping patience;
- unavailable CUDA device;
- validation pairs absent from the dataframe or overlapping train/validation
  audio candidate IDs;
- evaluation splits producing a singleton final batch;
- fewer than 10 retrieval candidates;
- `save_test_top_k` larger than the test candidate bank;
- query-attention feature size not divisible by the number of heads;
- time-reduction output length larger than the loaded audio sequence for modes
  that explicitly preserve multiple output tokens.

Some basic positive-range assumptions, such as positive channel counts,
positive kernel length, positive head stride, and positive initial CLIP
temperature, are enforced by downstream PyTorch/SciPy construction rather than
uniform parser-level checks. Supplying non-positive values is unsupported.
