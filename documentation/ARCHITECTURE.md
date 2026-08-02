# LISA Architecture

<p align="center">
  <a href="assets/LISA.png">
    <img src="assets/LISA.png" alt="LISA architecture" width="1100">
  </a>
</p>

<p align="center">
  <em>
    Reference architecture used for the main 2-block, 25-branch model. Optional
    architectural variants are described below.
  </em>
</p>

LISA is a multi-subject MEG encoder for audio-segment retrieval. It maps a
sensor-space MEG window to the same representation as a corresponding
Wav2Vec2 audio embedding. The model is deliberately split into:

1. an **interpretable front-end**, consisting of a subject-conditioned spatial
   transform followed by one depthwise temporal filter per branch;
2. a **shared temporal tail**, which captures additional nonlinear temporal
   structure; and
3. a **projection head**, which produces the final MEG embedding used by the
   contrastive retrieval objective.

The current repository targets the 208-channel MASC-MEG data and uses a
separate audio-side adapter when feature- or time-reduction ablations require
one.

## Notation and tensor layout

Throughout this document:

| Symbol | Meaning |
|---|---|
| \(B\) | batch size |
| \(M\) | number of MEG sensors; fixed to 208 in the current data pipeline |
| \(T\) | number of MEG samples in a window |
| \(A\) | number of spatial-attention channels (`n_channels_attention`) |
| \(K\) | number of interpretable branches (`n_channels_unmix`) |
| \(C\) | width of the temporal convolutional tail (`n_channels_block`) |
| \(F\) | final embedding feature dimension (`n_features`) |
| \(T_y\) | final embedding time dimension |
| \(S\) | number of subjects; fixed to 27 in the current data pipeline |

MEG tensors use the PyTorch channel-first layout
`(batch, channels, time)`. A model input is a tuple:

```python
(meg, subject_idx)
```

where:

- `meg` has shape `(B, M, T)`;
- `subject_idx` has shape `(B,)`, dtype `torch.long`, and contains zero-based
  subject indices in `[0, S - 1]`.

Audio embeddings are stored on disk as `(N, T_audio, F_audio)` and are
transposed to `(N, F_audio, T_audio)` before entering the audio adapter or the
retrieval objective.

## End-to-end data flow

The complete MEG pathway is:

```text
MEG (B, M, T) + subject index (B,)
    │
    ▼
coordinate-parameterized spatial attention (optional)
    │                  (B, A, T), or unchanged (B, M, T)
    ▼
shared 1×1 unmixing layer (optional)
    │
    ▼
subject-specific linear layer (optional)
    │
    ▼
K branch signals (B, K, T)
    │
    ▼
depthwise temporal filter: one kernel per branch
    │
    ▼
interpretable representation z (B, K, T)
    │
    ▼
0 or more dilated convolutional blocks
    │                  (B, C, T), or (B, K, T) when disabled
    ▼
projection head
    │
    ▼
optional time reduction
    │
    ▼
MEG embedding y (B, F, T_y)
```

The audio pathway produces a tensor with the same non-batch shape
`(F, T_y)`. The two representations are compared with a CLIP-style
contrastive objective.

# 1. Interpretable front-end

The interpretable front-end is implemented by `SpatialModule`. Despite its
historical name, it contains both the spatial transform and the branch-wise
temporal filters.

## 1.1 Coordinate-parameterized spatial attention

The optional spatial-attention layer maps the original sensors to \(A\)
intermediate spatial channels. It is important that this is **not
input-conditioned self-attention**: the sensor weights are learned model
parameters and are the same for every window. Sensor coordinates constrain the
weights to lie in a low-dimensional smooth spatial basis.

For attention channel \(a\), the implementation first constructs sensor logits
\(L_{a,m}\) from a coordinate basis and learned coefficients. It then applies a
softmax over sensors:

\[
C_{a,m} = \frac{\exp(L_{a,m})}
               {\sum_{j=1}^{M}\exp(L_{a,j})}.
\]

The attention output is

\[
u_a(t) = \sum_{m=1}^{M} C_{a,m} x_m(t).
\]

Consequently, each attention row is non-negative and sums to one. Signed
sensor-space filters become possible in the following learned linear layers.

Three modes are supported:

### 3D spherical-harmonic attention

With `use_spatial_attention="3D"`, sensor Cartesian coordinates are converted
to spherical coordinates. Let
\(H = \texttt{n_spatial_harmonics}\). The code constructs the real
spherical-harmonic basis for degrees \(0,\ldots,H-1\), giving \(H^2\) basis
functions, and learns one coefficient tensor of shape `(H, H)` for every
attention channel.

The resulting attention-logit matrix has shape `(A, M)`.

### 2D Fourier attention

With `use_spatial_attention="2D"`, the layer evaluates sine and cosine terms
on the normalized 2D sensor layout. For each pair of spatial frequencies
\((k,l)\), where \(k,l=1,\ldots,H\), it uses both

\[
\cos\bigl(2\pi(kx_m + ly_m)\bigr)
\quad\text{and}\quad
\sin\bigl(2\pi(kx_m + ly_m)\bigr).
\]

This gives `2 × H × H` basis coefficients per attention channel before the
sensor-wise softmax.

### No spatial attention

With `use_spatial_attention=None` (CLI value `none`), this stage is an identity
mapping and the next enabled spatial layer receives the original \(M\) sensor
channels.

### Spatial dropout

When enabled, spatial dropout samples circular regions in the normalized 2D
sensor layout during training. Sensors falling within any sampled region are
masked at the attention-logit level before softmax. The same sampled sensor
mask is broadcast across all attention channels for that forward pass.
Spatial dropout is inactive in evaluation mode.

## 1.2 Shared unmixing and subject-specific mapping

After spatial attention, the model can apply two optional linear transforms:

1. a shared biased `1×1` convolution (`unmixing_layer`);
2. a subject-specific bias-free linear map (`SubjectPlusLayer`).

The exact dimensions depend on which layers are enabled.

| Unmixing | Subject layer | Active mapping before temporal filtering |
|---|---|---|
| enabled | enabled | current channels → \(A\) by shared unmixing, then \(A\) → \(K\) by the subject matrix |
| enabled | disabled | current channels → \(K\) by shared unmixing |
| disabled | enabled | current channels → \(K\) directly by the subject matrix |
| disabled | disabled | identity; the active channel count must already equal \(K\) |

For the standard configuration, let:

- \(C\in\mathbb{R}^{A\times M}\) be the softmax-normalized spatial-attention
  matrix;
- \(W_u\in\mathbb{R}^{A\times A}\) and
  \(b_u\in\mathbb{R}^{A}\) be the shared unmixing parameters;
- \(W_s\in\mathbb{R}^{K\times A}\) be the matrix for subject \(s\).

The branch signals before temporal filtering are

\[
v_s(t) = W_s\bigl(W_u Cx(t) + b_u\bigr).
\]

The effective subject-specific sensor-space affine transform is therefore

\[
W^{\mathrm{eff}}_s = W_s W_u C,
\qquad
b^{\mathrm{eff}}_s = W_s b_u.
\]

Only the subject-specific matrix differs between participants. Spatial
attention, shared unmixing, temporal filters, temporal tail, and projection
head are shared.

## 1.3 Branch-wise temporal filtering

The spatial stage always outputs exactly \(K\) branch signals before temporal
filtering. The temporal filter is a depthwise `Conv1d` with:

- `in_channels = out_channels = K`;
- `groups = K`;
- one bias-free kernel per branch;
- same-length padding.

Thus branch \(k\) is filtered only by its own temporal kernel \(h_k\); the
filter does not mix branches.

Two implementations are available:

- `conv`: one depthwise 1-D convolution;
- `filtfilt`: the same padded depthwise convolution is applied once in the
  forward direction and once after reversing time, after which the output is
  reversed back.

### Initialization

By default, temporal kernels use the normal PyTorch convolution
initialization. When `temporal_filter_bands` is provided, each branch is
initialized with a SciPy `firwin` band-pass filter. If fewer bands than branches
are supplied, the bands are cycled across branches. The initialized kernels can
remain trainable or be frozen.

### Optional residual gate

With temporal gating enabled, the filtered output is interpolated with the
unfiltered branch signal:

\[
z = v + \alpha\,(h * v - v),
\qquad
\alpha = \sigma(\gamma).
\]

`gamma` is either:

- one scalar shared across all branches; or
- a length-\(K\) vector with one value per branch.

`tf_gate_init_alpha` specifies the initial value of \(\alpha\) before its logit
parameterization. If gating is disabled, the front-end uses the filtered signal
directly.

An optional GELU can be applied after filtering or gated interpolation. It is
disabled in the reference model, so its interpretable front-end remains affine
in the MEG input followed by a linear depthwise temporal filter.

## 1.4 Interpretable representation

The output of the front-end is

```text
z: (B, K, T)
```

and is returned by `forward_interpretable`. Each channel of `z` corresponds to
one branch defined by:

- a subject-specific effective spatial filter over the original sensors; and
- one shared temporal kernel.

This branch identity is preserved up to the beginning of the temporal tail.

# 2. Shared temporal tail

The temporal tail is implemented by `TemporalModule` as a sequence of
`ConvBlock` modules. It is optional: `n_temporal_module_blocks=0` replaces it
with an identity mapping.

When at least one block is enabled:

- the first block maps \(K\) input branches to \(C\) channels;
- every later block keeps \(C\) channels;
- all convolutions use kernel size 3 and `padding="same"`, so temporal length
  is preserved.

For block index \(b\), the dilation factors are

\[
d_1(b)=2^{(2b\bmod 5)},
\qquad
d_2(b)=2^{((2b+1)\bmod 5)},
\qquad
d_3=2.
\]

The block performs:

1. `Conv1d(..., dilation=d1)`;
2. an input residual connection for blocks after the first;
3. batch normalization and GELU;
4. `Conv1d(..., dilation=d2)` with a residual connection;
5. batch normalization and GELU;
6. `Conv1d(C, 2C, dilation=2)` followed by a channel-wise GLU, returning
   \(C\) channels.

There is no additional residual connection around the final GLU stage.

For the first five blocks, the `(d1, d2, d3)` schedules are:

| Block | Dilations |
|---:|---|
| 0 | `(1, 2, 2)` |
| 1 | `(4, 8, 2)` |
| 2 | `(16, 1, 2)` |
| 3 | `(2, 4, 2)` |
| 4 | `(8, 16, 2)` |

The schedule then repeats every five blocks.

# 3. Projection head

`ConvHead` maps the temporal-tail channels to the target feature dimension
\(F\) and optionally downsamples time.

The training CLI exposes two modes.

## `single_conv` — default

```text
Identity → GELU → Conv1d(C_in, F, kernel=3, stride=s) → BatchNorm1d(F)
```

## `conv`

```text
Conv1d(C_in, 2C_in, kernel=3, stride=s)
    → GELU
    → Conv1d(2C_in, F, kernel=1)
    → BatchNorm1d(F)
```

The model class also implements a `max` variant for internal experiments and
tests, but it is not exposed by the current training CLI.

For the layer that changes temporal length, the output size is

\[
T_y =
\left\lfloor
\frac{T + 2p - d(k-1) - 1}{s}
\right\rfloor + 1.
\]

The current heads use `k=3`, `d=1`, and:

- `p=0` when `head_stride=2`;
- `p=1` otherwise.

If the temporal tail is disabled, the head receives \(K\) channels directly;
`n_channels_block` is then ignored.

# 4. Optional time reduction

After the MEG projection head, `TimeReduction` can further reduce the temporal
axis of an embedding shaped `(B, F, T_y)`.

| Mode | Operation | Output time length |
|---|---|---:|
| `none` | identity | `T_y` |
| `mean` | mean over time | 1 |
| `max` | maximum over time | 1 |
| `linear` | learned linear map from input time positions to output positions | configured |
| `adaptive_avg` | adaptive average pooling | configured |
| `adaptive_max` | adaptive max pooling | configured |
| `attn` | additive attention pooling | 1 |
| `gated_attn` | gated attention pooling | 1 |
| `query_attn` | learned queries with multi-head attention | configured |

The reduction acts independently within each embedding feature channel, except
for the attention-based reducers, whose attention scores or queries are
computed from the full feature vector at each time point.

# 5. Audio pathway

The retrieval target is a sequence of Wav2Vec2 features. The preprocessing code
uses `facebook/wav2vec2-base-960h` by default and averages the last
`embedding_layers` hidden states at every Wav2Vec2 time step. The paper
configuration uses the last four states, producing 768-dimensional features.

The canonical 3-second audio windows contain 149 Wav2Vec2 time steps, so their
stored shape is:

```text
(N_candidates, 149, 768)
```

and their training-time shape is:

```text
(N_candidates, 768, 149)
```

With no reduction, the audio model is an identity mapping. Reduction ablations
modify the MEG and audio pathways as follows.

## Feature-axis reduction

| Mode | Audio pathway | MEG pathway |
|---|---|---|
| `none` | identity at 768 features | projection head outputs 768 features |
| `pca` | PCA-reduced audio is precomputed and loaded | projection head directly outputs the PCA dimension |
| `linear` | learned `1×1` convolution from 768 to the requested dimension | projection head outputs the same requested dimension |

## Time-axis reduction

| Mode | Audio pathway | MEG pathway |
|---|---|---|
| `pca` | time-axis PCA is precomputed and loaded | learned linear time reduction to the same number of components |
| all trainable/non-PCA modes | a separate `TimeReduction` instance | another, independently parameterized `TimeReduction` instance |

The MEG and audio reducers do **not** share trainable parameters. They are only
required to produce the same final shape. Feature-axis and time-axis reductions
are mutually exclusive in the current training CLI.

## Parameter-count partition

`lisa-export-parameter-counts` reconstructs both pathways from each saved
`config.json` and counts PyTorch parameters without loading checkpoint weights.
Trainable and total counts use the same non-overlapping partition:

```text
full inference network = complete MEG model + complete audio adapter
complete MEG model     = MEG core + projection/resampler + MEG time reducer
complete audio adapter = audio feature reducer + audio time reducer + other
```

Here `MEG core` is every `LISA` parameter except those owned by
`feature_projection` and the optional explicit `time_reducer`.
`feature_projection` is a `ConvHead`: its convolution performs feature
projection and can simultaneously perform temporal resampling through its
stride. Those shared weights cannot be meaningfully divided into separate
feature-projection and resampling parameter sets, so the CSV reports the whole
module as `meg_projection_resampler_*`.

The headline `trainable_params` and `total_params` columns contain the complete
MEG model plus audio adapter. The learnable CLIP temperature is not part of the
inference network and is reported separately under `criterion_*`;
`optimizer_trainable_params` includes it.

Precomputed PCA is applied to audio arrays before training and is not an
instantiated PyTorch module in the saved inference network. It therefore adds
no module parameters to these counts; the CSV records this case with
`audio_precomputed_reduction=true`.

# 6. Contrastive retrieval objective

Before comparison, the MEG and audio embeddings must have exactly the same
non-batch shape. For three-dimensional embeddings, both tensors are normalized
over the joint feature and time dimensions. Their similarity is therefore the
cosine similarity between flattened `(F, T_y)` representations:

\[
\operatorname{sim}(y_i,a_j)
= \frac{\langle y_i,a_j\rangle_F}
       {\lVert y_i\rVert_F\,\lVert a_j\rVert_F}.
\]

Training uses a one-directional CLIP-style cross-entropy loss from MEG windows
to audio candidates. A batch can contain the same audio window for multiple
subjects or sessions. `positive_ids` are therefore deduplicated, one audio
embedding is retained for each unique audio-window ID, and every MEG sample is
assigned to the corresponding unique candidate.

The similarities are divided by a positive learned temperature
`exp(log_temperature)` before cross-entropy. The implementation does not add a
separate symmetric audio-to-MEG loss.

At evaluation time, retrieval metrics compare each MEG output against the
complete candidate bank provided for that split rather than only against the
current mini-batch.

# 7. Forward and extraction APIs

## Forward pass

```python
z, y = model((meg, subject_idx))
```

is equivalent to:

```python
z = model.forward_interpretable((meg, subject_idx))
y = model.forward_tail(z)
```

where:

- `z` has shape `(B, K, T)`;
- `y` has shape `(B, F, T_y)`.

`forward_tail` includes the temporal convolutional blocks, projection head, and
optional MEG-side time reduction.

## Spatial-filter extraction

```python
weights, bias = model.extract_spatial_filters()
```

returns the composed effective spatial transform before temporal filtering:

- `weights`: `(S, K, M)`;
- `bias`: `(S, K, 1)` or `None`.

The method composes spatial attention, unmixing, and the subject-specific
matrix. If the subject layer is disabled, the shared transform is repeated for
all subjects to preserve the same output interface.

These arrays are **spatial filters**, not cortical source locations. Sensor
patterns and template-space source projections are separate post-hoc analyses
that additionally depend on data covariance and the chosen forward/inverse
model.

## Temporal-filter extraction

```python
kernels = model.extract_temporal_filters()
```

returns the learned depthwise kernels with shape `(K, kernel_size)`.

# 8. Reference 2-block, 25-branch configuration

The architecture figure at the top of this document depicts the main reference
configuration currently used for detailed interpretation:

| Component | Setting | Output shape for a 3-second window |
|---|---|---|
| Input | 208 channels, 100 Hz | `(B, 208, 300)` |
| 3D spatial attention | `A=270`, `H=24` | `(B, 270, 300)` |
| Shared unmixing | `270 → 270` | `(B, 270, 300)` |
| Subject layer | `270 → K`, `K=25` | `(B, 25, 300)` |
| Temporal filters | depthwise, kernel 15 samples | `(B, 25, 300)` |
| Temporal block 0 | width 25, dilations `(1, 2, 2)` | `(B, 25, 300)` |
| Temporal block 1 | width 25, dilations `(4, 8, 2)` | `(B, 25, 300)` |
| Projection head | `single_conv`, kernel 3, stride 2 | `(B, 768, 149)` |
| Time reduction | none | `(B, 768, 149)` |
| Audio adapter | identity | `(B, 768, 149)` |

For this configuration, one final embedding time point has a theoretical
receptive field of 55 MEG samples, or 550 ms at 100 Hz, before boundary-padding
effects. Consecutive embedding positions are separated by two MEG samples
(20 ms).

# 9. Implementation map

The architecture is defined primarily in:

| Component | Source |
|---|---|
| Spatial attention, subject layer, temporal filters, tail, heads, reducers | `src/lisa/model/nn_modules.py` |
| Construction from saved/CLI configuration | `src/lisa/model/load_model.py` |
| Training-time data and model assembly | `src/lisa/cli/train_model.py` |
| Model and reduction CLI options | `src/lisa/cli/parse_experiment_args.py` |
| Per-run parameter-count and final-test-metric CSV export | `src/lisa/cli/export_parameter_counts.py` |
| MEG/audio sample interface | `src/lisa/data/datasets.py` |
| Audio PCA preprocessing | `src/lisa/data/audio_reduction.py` |
| Wav2Vec2 target generation | `src/lisa/data/generate_all_embeddings.py` |
| Contrastive loss and retrieval similarity | `src/lisa/training/criteria.py` |
| Architecture, reduction, and parameter-count tests | `tests/test_lisa_variations.py`, `tests/test_audio_reduction.py`, `tests/test_export_parameter_counts.py` |

For exact CLI defaults and valid combinations, see
[`HYPERPARAMETERS.md`](HYPERPARAMETERS.md). For running your own experiments or
reusing the encoder, see [`ADAPTING.md`](ADAPTING.md). For commands reproducing
the paper experiments and figures, see [`REPRODUCE.md`](REPRODUCE.md).
