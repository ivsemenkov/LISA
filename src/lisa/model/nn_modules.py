"""Neural network modules for LISA models (spatial attention, filtering, pooling)."""

import math
import warnings
from typing import Tuple

import einops
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin


def get_spatial_attention_layer(
    attention_type,
    n_input,
    n_output,
    K,
    coords_xy,
    n_dropout,
    dropout_radius,
    coords_xyz,
    seed=None,
    dropout_center_low=0.1,
    dropout_center_high=0.9,
):
    """Factory for spatial attention layers.

    Args:
        attention_type: "2D", "3D", or None (disabled).
        n_input: Number of input channels (MEG sensors).
        n_output: Number of attention output channels.
        K: Spatial harmonics / basis size.
        coords_xy: 2D sensor coordinates (N, 2).
        n_dropout: Number of spatial dropout centers.
        dropout_radius: Radius for spatial dropout.
        coords_xyz: 3D sensor coordinates (N, 3).
        seed: Optional RNG seed for reproducible init of spatial weights (Z) and dropout centers; when None, init is non-deterministic.
        dropout_center_low: Lower bound for dropout center sampling range.
        dropout_center_high: Upper bound for dropout center sampling range.
    """
    if attention_type is None:
        return None
    elif attention_type == '2D':
        return Spatial2DAttentionLayer(
            n_input=n_input,
            n_output=n_output,
            K=K,
            coords_xy=coords_xy,
            n_dropout=n_dropout,
            dropout_radius=dropout_radius,
            seed=seed,
            dropout_center_low=dropout_center_low,
            dropout_center_high=dropout_center_high,
        )
    elif attention_type == '3D':
        return Spatial3DAttentionLayer(
            n_input=n_input,
            n_output=n_output,
            K=K,
            coords_xy=coords_xy,
            n_dropout=n_dropout,
            dropout_radius=dropout_radius,
            coords_xyz=coords_xyz,
            seed=seed,
            dropout_center_low=dropout_center_low,
            dropout_center_high=dropout_center_high,
        )
    else:
        raise NotImplementedError(
            f'attention_type should be 2D, 3D, or None. Got {attention_type}'
        )


def cart2sph(sensor_xyz):
    """Convert Cartesian sensor coordinates to spherical (r, theta, phi)."""
    x, y, z = sensor_xyz[:, 0], sensor_xyz[:, 1], sensor_xyz[:, 2]
    xy = np.linalg.norm(sensor_xyz[:, :2], axis=-1)
    r = np.linalg.norm(sensor_xyz, axis=-1)
    theta = np.arctan2(xy, z)
    phi = np.arctan2(y, x)
    return np.stack((r, theta, phi), axis=-1)


class SpatialAttentionLayerBase(nn.Module):
    """Base class for spatial attention layers.

    Provides common functionality for both 2D and 3D spatial attention layers,
    including coordinate processing, dropout logic, and forward pass.
    """

    def __init__(
        self,
        n_input,
        n_output,
        K,
        coords_xy,
        n_dropout,
        dropout_radius,
        seed=None,
        dropout_center_low=0.1,
        dropout_center_high=0.9,
    ):
        """Initialize base spatial attention layer.

        Args:
            n_input: Number of sensors.
            n_output: Number of attention outputs.
            K: Harmonic/basis order.
            coords_xy: 2D coordinates (N, 2) for dropout.
            n_dropout: Number of dropout centers.
            dropout_radius: Dropout radius in coord space.
            seed: Optional RNG seed for reproducible init of Z and dropout centers; when None, init is non-deterministic.
            dropout_center_low: Lower bound for dropout center sampling range.
            dropout_center_high: Upper bound for dropout center sampling range.
        """
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.n_dropout = n_dropout
        self.dropout_radius = dropout_radius
        self.K = K
        self._seed = seed
        self.dropout_center_low = dropout_center_low
        self.dropout_center_high = dropout_center_high

        coords_xy = torch.tensor(coords_xy, dtype=torch.float32, requires_grad=False)
        assert coords_xy.shape == (n_input, 2), (
            f'coords_xy.shape: {coords_xy.shape}, n_input: {n_input}'
        )
        if not torch.isfinite(coords_xy).all():
            raise ValueError('coords_xy contains NaN/Inf')
        self.register_buffer('_coords_xy', coords_xy)

        layout = self._create_layout_buffer()
        self.register_buffer(self._layout_buffer_name, layout)

        Z = self._create_parameter_tensor()
        self.Z = nn.Parameter(Z)

    @property
    def _layout_buffer_name(self):
        """Return the name of the layout buffer attribute.

        Subclasses must override this to specify their layout buffer name.
        """
        raise NotImplementedError(
            'Subclasses must implement _layout_buffer_name property'
        )

    def _create_layout_buffer(self):
        """Create and return the layout buffer.

        Subclasses must implement this to create their specific layout.

        Returns:
            Layout tensor to be registered as a buffer.
        """
        raise NotImplementedError('Subclasses must implement _create_layout_buffer')

    def _create_parameter_tensor(self):
        """Create and return the Z parameter tensor.

        Subclasses must implement this to create their specific parameter shape.

        Returns:
            Parameter tensor to be registered as nn.Parameter.
        """
        raise NotImplementedError('Subclasses must implement _create_parameter_tensor')

    def _get_attention_matrix(self):
        """Compute attention matrix A from Z and layout.

        Subclasses must implement this to handle their specific layout structure.

        Returns:
            Attention matrix A with shape (n_output, n_input).
        """
        raise NotImplementedError('Subclasses must implement _get_attention_matrix')

    def get_spatial_filter(self):
        """Get the spatial filter (softmax-normalized attention weights)."""
        A = self._get_attention_matrix()
        ASoftmax = F.softmax(A, dim=1)
        return ASoftmax.clone().detach()

    def forward(self, x):
        """Forward pass with optional spatial dropout."""
        A = self._get_attention_matrix()
        if self.training and self.n_dropout > 0:
            # Spatial dropout (experimental): randomly masks sensors within a radius of random centers in normalized 2D layout.
            # Spatial dropout centers are sampled uniformly in coords_xy-space from [dropout_center_low, dropout_center_high].
            # If your coords_xy use a different scale/range, adjust dropout_center_low and dropout_center_high.
            dropout_range = self.dropout_center_high - self.dropout_center_low
            dropout_location = (
                torch.rand(size=(self.n_dropout, 2), device=A.device) * dropout_range
                + self.dropout_center_low
            )
            # torch.cdist requires 2D: (n_points_1, d) and (n_points_2, d)
            distances = torch.cdist(
                dropout_location, self._coords_xy, p=2
            )  # (n_dropout, n_input)
            # Mark sensors within dropout_radius of any dropout center
            within_radius = (distances <= self.dropout_radius).any(dim=0)  # (n_input,)
            mask = torch.zeros((1, self.n_input), dtype=A.dtype, device=A.device)
            mask[0, within_radius] = -float('inf')
            A = A + mask
        ASoftmax = F.softmax(A, dim=1)
        SAx = torch.einsum('oi, bit -> bot', ASoftmax, x)
        return SAx


class Spatial3DAttentionLayer(SpatialAttentionLayerBase):
    """3D spherical-harmonics attention over sensors.

    Args:
        n_input: Number of sensors.
        n_output: Number of attention outputs.
        K: Harmonic order (basis size).
        coords_xy: 2D coordinates (for dropout).
        n_dropout: Number of dropout centers.
        dropout_radius: Dropout radius in coord space.
        coords_xyz: 3D coordinates (N, 3).
        seed: Optional RNG seed for reproducible init of Z and dropout centers; when None, init is non-deterministic.
        dropout_center_low: Lower bound for dropout center sampling range.
        dropout_center_high: Upper bound for dropout center sampling range.
    """

    def __init__(
        self,
        n_input,
        n_output,
        K,
        coords_xy,
        n_dropout,
        dropout_radius,
        coords_xyz,
        seed=None,
        dropout_center_low=0.1,
        dropout_center_high=0.9,
    ):
        # Store coords_xyz for use in _create_layout_buffer (must be before super().__init__)
        self._coords_xyz = coords_xyz

        super().__init__(
            n_input=n_input,
            n_output=n_output,
            K=K,
            coords_xy=coords_xy,
            n_dropout=n_dropout,
            dropout_radius=dropout_radius,
            seed=seed,
            dropout_center_low=dropout_center_low,
            dropout_center_high=dropout_center_high,
        )

    def _create_parameter_tensor(self):
        """Create Z parameter tensor for 3D spherical harmonics."""
        if self._seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self._seed
        generator = torch.Generator()
        generator.manual_seed(seed)

        Z = (
            torch.randn(size=((self.n_output, self.K, self.K)), generator=generator)
            * 2
            / (self.n_input + self.n_output)
        )
        Z = einops.rearrange(Z, 'j k l -> j k l 1')
        return Z

    @property
    def _layout_buffer_name(self):
        """Return the layout buffer name for 3D attention."""
        return '_layout'

    def _create_layout_buffer(self):
        """Create layout from spherical coordinates for 3D attention."""
        coords_sph = cart2sph(self._coords_xyz)
        return self._create_layout(coords_sph, self.K - 1)

    def _create_layout(self, coords_sph, L=8):
        """Create layout from numpy spherical coordinates. All work stays in numpy."""
        n_input = coords_sph.shape[0]

        coords_theta = coords_sph[:, 1].astype(np.float64)
        coords_phi = coords_sph[:, 2].astype(np.float32)

        res = scipy.special.assoc_legendre_p_all(
            L, L, np.cos(coords_theta), norm=False, diff_n=0
        )
        assert res.shape == (1, L + 1, 2 * L + 1, n_input)

        Plak_values = res[0, :, : L + 1, :].transpose(1, 0, 2)  # (m, degree, n_input)

        def get_factor(l, m):
            result = math.log(2 * l + 1)
            result -= math.log((2 if m == 0 else 1) * 2 * math.pi)
            result += scipy.special.gammaln(l - abs(m) + 1)
            result -= scipy.special.gammaln(l + abs(m) + 1)
            result /= 2
            result = math.exp(result)
            return result

        layout = np.zeros((1, L + 1, L + 1, n_input), dtype=np.float32)
        counter = -1
        for l in range(0, L + 1):
            for m in range(-l, l + 1):
                counter += 1
                i, j = counter % (L + 1), counter // (L + 1)
                mult_left = get_factor(l, m)

                if m >= 0:
                    mult = np.cos(m * coords_phi)
                elif m < 0:
                    mult = np.sin(-m * coords_phi)
                sela = mult_left * Plak_values[abs(m), l, :] * mult

                layout[:, i, j, :] = sela
        layout = torch.tensor(layout, dtype=torch.float32, requires_grad=False)
        return layout

    def _get_attention_matrix(self):
        """Compute attention matrix from Z and layout for 3D spherical harmonics."""
        A = einops.reduce(self.Z * self._layout, 'j k l i -> j i', 'sum')
        return A


class Spatial2DAttentionLayer(SpatialAttentionLayerBase):
    """2D Fourier attention over sensors.

    Args:
        Same as SpatialAttentionLayerBase; seed controls reproducible init of Z and dropout centers.
    """

    def __init__(
        self,
        n_input,
        n_output,
        K,
        coords_xy,
        n_dropout,
        dropout_radius,
        seed=None,
        dropout_center_low=0.1,
        dropout_center_high=0.9,
    ):
        super().__init__(
            n_input=n_input,
            n_output=n_output,
            K=K,
            coords_xy=coords_xy,
            n_dropout=n_dropout,
            dropout_radius=dropout_radius,
            seed=seed,
            dropout_center_low=dropout_center_low,
            dropout_center_high=dropout_center_high,
        )

    @property
    def _layout_buffer_name(self):
        """Return the layout buffer name for 2D attention."""
        return '_fourier_layout'

    def _create_parameter_tensor(self):
        """Create Z parameter tensor for 2D Fourier basis."""
        if self._seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self._seed
        generator = torch.Generator()
        generator.manual_seed(seed)

        Z = (
            torch.randn(size=((self.n_output, 2, self.K, self.K)), generator=generator)
            * 2
            / (self.n_input + self.n_output)
        )
        Z = einops.rearrange(Z, 'j a k l -> j a k l 1')
        return Z

    def _create_layout_buffer(self):
        """Create layout from 2D coordinates for 2D Fourier attention."""
        coords_x = self._coords_xy[:, 0]
        coords_y = self._coords_xy[:, 1]
        fourier_layout = torch.zeros(
            (2, self.K, self.K, self.n_input), requires_grad=False
        )
        for k in range(self.K):
            for l in range(self.K):
                coords = 2 * math.pi * ((k + 1) * coords_x + (l + 1) * coords_y)
                fourier_layout[0, k, l, :] = torch.cos(coords)
                fourier_layout[1, k, l, :] = torch.sin(coords)
        fourier_layout = einops.rearrange(fourier_layout, 'a k l i -> 1 a k l i')
        return fourier_layout

    def _get_attention_matrix(self):
        """Compute attention matrix from Z and layout for 2D Fourier basis."""
        A = einops.reduce(self.Z * self._fourier_layout, 'j a k l i -> j i', 'sum')
        return A


class SubjectPlusLayer(nn.Module):
    """Subject-specific linear mapping.

    Args:
        n_input: Input channels.
        n_output: Output channels.
        n_subjects: Number of subjects. Subject IDs should be in range [0, n_subjects-1].
    """

    def __init__(self, n_input, n_output, n_subjects):
        super().__init__()

        A = self._create_parameters(n_input, n_output, n_subjects)
        self.A = nn.Parameter(A)

    def _create_parameters(self, n_input, n_output, n_subjects):
        A = torch.zeros(size=(n_subjects, n_output, n_input))
        with torch.no_grad():
            for subjects in range(n_subjects):
                layer = nn.Conv1d(
                    in_channels=n_input, out_channels=n_output, kernel_size=1
                )
                A[subjects] = einops.rearrange(layer.weight.data, 'o i 1 -> o i')
        return A

    def forward(self, x, s):
        """Forward pass with 0-indexed subject IDs.

        Args:
            x: Input tensor (B, C, T).
            s: Subject indices (B,). Must be 0-indexed [0, n_subjects-1].
                For n_subjects=27, use indices 0, 1, 2, ..., 26.

        Returns:
            Output tensor (B, C_out, T).

        Raises:
            ValueError: If subject indices are out of valid range.
        """
        assert s.dtype == torch.long, (
            f'subject index dtype must be torch.long, got {s.dtype}'
        )

        min_s = int(s.min().item())
        max_s = int(s.max().item())
        n_subjects = self.A.size(0)

        # Validate 0-indexed range
        if min_s < 0 or max_s >= n_subjects:
            raise ValueError(
                f'Subject indices out of range: [{min_s}, {max_s}], '
                f'valid range is [0, {n_subjects - 1}] (0-indexed). '
                f'For {n_subjects} subjects, use indices 0, 1, ..., {n_subjects - 1}.'
            )

        # Direct 0-indexed access to subject parameters
        A_ = self.A[s, :, :]
        out = torch.einsum('bji, bit -> bjt', A_, x)
        return out


class ConvBlock(nn.Module):
    """Dilated temporal convolution block with residual connections."""

    def __init__(self, n_input, n_output, block_index):
        super().__init__()

        self.kernel_size = 3
        self.block_index = block_index
        dilation1 = 2 ** (2 * block_index % 5)
        dilation2 = 2 ** ((2 * block_index + 1) % 5)
        dilation3 = 2

        self.conv1 = nn.Conv1d(
            in_channels=n_input,
            out_channels=n_output,
            kernel_size=self.kernel_size,
            dilation=dilation1,
            padding='same',
        )
        self.conv2 = nn.Conv1d(
            in_channels=n_output,
            out_channels=n_output,
            kernel_size=self.kernel_size,
            dilation=dilation2,
            padding='same',
        )
        self.conv3 = nn.Conv1d(
            in_channels=n_output,
            out_channels=2 * n_output,
            kernel_size=self.kernel_size,
            dilation=dilation3,
            padding='same',
        )

        self.batchnorm1 = nn.BatchNorm1d(n_output)
        self.batchnorm2 = nn.BatchNorm1d(n_output)

        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.activation3 = nn.GLU(dim=-2)

    def forward(self, x):
        c1x = self.conv1(x)
        res1 = c1x if self.block_index == 0 else x + c1x
        res1 = self.batchnorm1(res1)
        res1 = self.activation1(res1)

        c2x = self.conv2(res1)
        res2 = res1 + c2x
        res2 = self.batchnorm2(res2)
        res2 = self.activation2(res2)

        c3x = self.conv3(res2)
        out = self.activation3(c3x)

        return out


class ConvHead(nn.Module):
    """Projection head for temporal features with optional pooling."""

    def __init__(self, n_channels, n_features, pool, head_stride):
        super().__init__()
        self._pool_type = pool

        if pool == 'single_conv':
            self.pool = nn.Identity()
            self.conv = nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_features,
                kernel_size=3,
                stride=head_stride,
                padding=0 if head_stride == 2 else 1,
            )
        else:
            if pool == 'max':
                self.pool = nn.Sequential(
                    nn.MaxPool1d(
                        kernel_size=3,
                        stride=head_stride,
                        padding=0 if head_stride == 2 else 1,
                    ),
                    nn.Conv1d(
                        in_channels=n_channels,
                        out_channels=2 * n_channels,
                        kernel_size=1,
                    ),
                )
            elif pool == 'conv':
                self.pool = nn.Conv1d(
                    in_channels=n_channels,
                    out_channels=2 * n_channels,
                    kernel_size=3,
                    stride=head_stride,
                    padding=0 if head_stride == 2 else 1,
                )
            else:
                raise NotImplementedError(f'Unknown pool type: {pool}')

            self.conv = nn.Conv1d(
                in_channels=2 * n_channels, out_channels=n_features, kernel_size=1
            )
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm1d(n_features)

    def forward(self, x):
        x = self.pool(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        return x

    def _get_time_downsampler_layer(self):
        """Return the layer that changes temporal length inside the head."""
        if self._pool_type == 'single_conv':
            return self.conv
        if self._pool_type == 'max':
            return self.pool[0]
        if self._pool_type == 'conv':
            return self.pool
        raise RuntimeError(f'Unknown ConvHead pool type: {self._pool_type!r}')

    def output_length(self, input_length: int) -> int:
        """Return the temporal length after ConvHead pooling/projection."""
        if input_length <= 0:
            raise ValueError(f'Expected positive input_length, got {input_length}.')
        layer = self._get_time_downsampler_layer()
        kernel_size = layer.kernel_size
        stride = layer.stride
        padding = layer.padding
        dilation = layer.dilation
        kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        stride = stride[0] if isinstance(stride, tuple) else stride
        padding = padding[0] if isinstance(padding, tuple) else padding
        dilation = dilation[0] if isinstance(dilation, tuple) else dilation
        output_length = (
            input_length + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        if output_length <= 0:
            raise ValueError(
                'ConvHead produced a non-positive time length. '
                f'input_length={input_length}, kernel_size={kernel_size}, '
                f'stride={stride}, padding={padding}, dilation={dilation}, '
                f'output_length={output_length}.'
            )
        return output_length


class AdditiveAttentionPooling(nn.Module):
    """Trainable additive attention pooling from (B, F, T) to (B, F, 1)."""

    def __init__(self, n_features: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = n_features if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError(f'hidden_dim must be positive. Got {hidden_dim}.')

        self.proj = nn.Conv1d(n_features, hidden_dim, kernel_size=1, bias=True)
        self.score = nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.proj(x))
        alpha = torch.softmax(self.score(h), dim=-1)
        return torch.sum(x * alpha, dim=-1, keepdim=True)


class GatedAttentionPooling(nn.Module):
    """Gated attention pooling from (B, F, T) to (B, F, 1)."""

    def __init__(self, n_features: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = n_features if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError(f'hidden_dim must be positive. Got {hidden_dim}.')

        self.value_proj = nn.Conv1d(n_features, hidden_dim, kernel_size=1, bias=True)
        self.gate_proj = nn.Conv1d(n_features, hidden_dim, kernel_size=1, bias=True)
        self.score = nn.Conv1d(hidden_dim, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.tanh(self.value_proj(x))
        gate = torch.sigmoid(self.gate_proj(x))
        alpha = torch.softmax(self.score(value * gate), dim=-1)
        return torch.sum(x * alpha, dim=-1, keepdim=True)


class QueryAttentionPooling(nn.Module):
    """Learnable query pooling from (B, F, T) to (B, F, Q)."""

    def __init__(self, n_features: int, output_length: int, num_heads: int = 1):
        super().__init__()
        if output_length <= 0:
            raise ValueError(
                f'output_length must be positive. Got {output_length}.'
            )
        if num_heads <= 0:
            raise ValueError(f'num_heads must be positive. Got {num_heads}.')
        if n_features % num_heads != 0:
            raise ValueError(
                f'n_features ({n_features}) must be divisible by num_heads ({num_heads}).'
            )

        self.query = nn.Parameter(torch.randn(output_length, n_features) * 0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_features,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_seq = einops.rearrange(x, 'b f t -> b t f')
        query = self.query.unsqueeze(0).expand(x_seq.size(0), -1, -1)
        pooled, _ = self.attention(query, x_seq, x_seq, need_weights=False)
        pooled = self.norm(pooled + query)
        return einops.rearrange(pooled, 'b q f -> b f q')


class TimeReduction(nn.Module):
    """Unified temporal reduction module for embeddings shaped as (B, F, T)."""

    def __init__(
        self,
        reduction: str,
        n_features: int,
        output_length: int | None = None,
        input_length: int | None = None,
        hidden_dim: int | None = None,
        num_heads: int = 1,
    ):
        super().__init__()
        self.reduction_type = reduction
        self.reducer = None

        if reduction == 'mean':
            pass
        elif reduction == 'max':
            pass
        elif reduction == 'linear':
            if output_length is None:
                raise ValueError(
                    'output_length must be provided for reduction="linear".'
                )
            if input_length is None:
                raise ValueError(
                    'input_length must be provided for reduction="linear".'
                )
            if output_length <= 0:
                raise ValueError(
                    f'output_length must be positive. Got {output_length}.'
                )
            if input_length <= 0:
                raise ValueError(f'input_length must be positive. Got {input_length}.')
            self.reducer = nn.Linear(input_length, output_length, bias=True)
        elif reduction == 'adaptive_avg':
            if output_length is None:
                raise ValueError(
                    'output_length must be provided for reduction="adaptive_avg".'
                )
            self.reducer = nn.AdaptiveAvgPool1d(output_length)
        elif reduction == 'adaptive_max':
            if output_length is None:
                raise ValueError(
                    'output_length must be provided for reduction="adaptive_max".'
                )
            self.reducer = nn.AdaptiveMaxPool1d(output_length)
        elif reduction == 'attn':
            if output_length is not None:
                raise ValueError('output_length is not used for reduction="attn".')
            self.reducer = AdditiveAttentionPooling(
                n_features=n_features,
                hidden_dim=hidden_dim,
            )
        elif reduction == 'gated_attn':
            if output_length is not None:
                raise ValueError(
                    'output_length is not used for reduction="gated_attn".'
                )
            self.reducer = GatedAttentionPooling(
                n_features=n_features,
                hidden_dim=hidden_dim,
            )
        elif reduction == 'query_attn':
            if output_length is None:
                raise ValueError(
                    'output_length must be provided for reduction="query_attn".'
                )
            self.reducer = QueryAttentionPooling(
                n_features=n_features,
                output_length=output_length,
                num_heads=num_heads,
            )
        else:
            raise NotImplementedError(f'Unknown time reduction type: {reduction}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected a 3D tensor (B, F, T), got shape {x.shape}.')

        if self.reduction_type == 'mean':
            return x.mean(dim=-1, keepdim=True)
        if self.reduction_type == 'max':
            return x.amax(dim=-1, keepdim=True)

        return self.reducer(x)


class AudioEmbeddingModel(nn.Module):
    """Audio-side embedding adapter for feature and time reduction."""

    def __init__(
        self,
        n_input_features: int,
        n_output_features: int,
        feature_reduction: str = 'none',
        time_reduction: str = 'none',
        time_reduction_dim: int | None = None,
        time_reduction_input_length: int | None = None,
        time_reduction_hidden_dim: int | None = None,
        time_reduction_num_heads: int = 1,
    ):
        super().__init__()
        self.n_input_features = n_input_features
        self.n_output_features = n_output_features
        self.feature_reduction = feature_reduction

        if feature_reduction == 'none':
            if n_input_features != n_output_features:
                raise ValueError(
                    'AudioEmbeddingModel with feature_reduction="none" requires matching input/output dimensions. '
                    f'Got n_input_features={n_input_features}, n_output_features={n_output_features}.'
                )
            self.feature_reducer = None
        elif feature_reduction == 'linear':
            self.feature_reducer = nn.Conv1d(
                in_channels=n_input_features,
                out_channels=n_output_features,
                kernel_size=1,
                bias=True,
            )
        else:
            raise NotImplementedError(
                f'Unknown audio feature reduction type: {feature_reduction}'
            )

        if time_reduction == 'none':
            self.time_reducer = None
        else:
            self.time_reducer = TimeReduction(
                reduction=time_reduction,
                n_features=n_output_features,
                output_length=time_reduction_dim,
                input_length=time_reduction_input_length,
                hidden_dim=time_reduction_hidden_dim,
                num_heads=time_reduction_num_heads,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f'Expected a 3D tensor (B, F, T), got shape {x.shape}.')
        if self.feature_reducer is not None:
            x = self.feature_reducer(x)
        if self.time_reducer is not None:
            x = self.time_reducer(x)
        return x


class FiltFiltConv1d(nn.Conv1d):
    """Conv1d that performs forward-backward (zero-phase) filtering."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        padding='same',
        padding_mode='zeros',
        bias=False,
    ):
        if padding == 'same':
            pad = (kernel_size - 1) // 2
        else:
            pad = padding
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad,
            padding_mode=padding_mode,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_f = super().forward(x)
        y_rev = y_f.flip(dims=[-1])
        y_rev_f = super().forward(y_rev)
        return y_rev_f.flip(dims=[-1])


class SpatialModule(nn.Module):
    """Interpretable front-end: spatial attention, unmixing, subject layer, and temporal filter."""

    def __init__(
        self,
        n_input,
        n_attention,
        n_unmix,
        use_spatial_attention,
        n_spatial_harmonics,
        coords_xy_scaled,
        spatial_dropout_number,
        spatial_dropout_radius,
        coords_xyz,
        temporal_filter_type,
        temporal_filter_bands,
        temporal_filter_fs,
        temporal_filter_freeze,
        temporal_filter_kernel_size,
        temporal_filter_padding_mode,
        tf_gated,
        tf_gate_per_channel,
        tf_gate_init_alpha,
        tf_gelu,
        n_subjects,
        use_unmixing_layer=True,
        use_subject_layer=True,
        dropout_center_low=0.1,
        dropout_center_high=0.9,
        seed=None,
    ):
        """Initialize the spatial front-end.

        Args:
            n_input: Number of MEG channels.
            n_attention: Number of attention channels.
            n_unmix: Number of unmix channels (K).
            use_spatial_attention: "2D", "3D", or None (disabled).
            n_spatial_harmonics: Spatial basis size for attention.
            coords_xy_scaled: Sensor layout (N, 2) scaled to [-1, 1].
            spatial_dropout_number: Number of spatial dropout centers.
            spatial_dropout_radius: Dropout radius in layout space.
            coords_xyz: Sensor coordinates (N, 3) for 3D attention.
            temporal_filter_type: "conv" or "filtfilt".
            temporal_filter_bands: Optional list of (low, high) bands for FIR init. If provided, initializes FIR filters; if None, uses random initialization.
            temporal_filter_fs: Sampling rate (Hz).
            temporal_filter_freeze: Freeze temporal filter weights.
            temporal_filter_kernel_size: Kernel size in samples.
            temporal_filter_padding_mode: Padding mode for temporal filter.
            tf_gated: Enable residual gate around temporal filter.
            tf_gate_per_channel: Learn per-channel gates.
            tf_gate_init_alpha: Initial alpha for gate.
            tf_gelu: Enable GELU activation after temporal filtering.
            n_subjects: Number of subjects.
            use_unmixing_layer: Enable the 1x1 unmixing layer before the subject layer.
            use_subject_layer: Enable the subject-specific linear mapping before the temporal filter.
            dropout_center_low: Lower bound for dropout center sampling range.
            dropout_center_high: Upper bound for dropout center sampling range.
            seed: Optional RNG seed for reproducible init of spatial attention (Z and dropout centers); when None, init is non-deterministic.
        """
        super().__init__()
        self.n_input = n_input
        self.n_attention = n_attention
        self.n_unmix = n_unmix
        self.n_subjects = n_subjects
        self.temporal_filter_kernel_size = temporal_filter_kernel_size
        self.temporal_filter_fs = temporal_filter_fs
        self.tf_gated = tf_gated
        self.tf_gate_per_channel = tf_gate_per_channel
        self.tf_gelu = tf_gelu
        self.use_unmixing_layer = use_unmixing_layer
        self.use_subject_layer = use_subject_layer

        # Spatial attention (optional)
        self.self_attention = get_spatial_attention_layer(
            attention_type=use_spatial_attention,
            n_input=n_input,
            n_output=n_attention,
            K=n_spatial_harmonics,
            coords_xy=coords_xy_scaled,
            n_dropout=spatial_dropout_number,
            dropout_radius=spatial_dropout_radius,
            coords_xyz=coords_xyz,
            seed=seed,
            dropout_center_low=dropout_center_low,
            dropout_center_high=dropout_center_high,
        )

        current_channels = n_attention if self.self_attention is not None else n_input
        if self.use_unmixing_layer:
            unmix_out_channels = n_attention if self.use_subject_layer else n_unmix
            self.unmixing_layer = nn.Conv1d(
                in_channels=current_channels,
                out_channels=unmix_out_channels,
                kernel_size=1,
                bias=True,
            )
            current_channels = unmix_out_channels
        else:
            self.unmixing_layer = None

        if self.use_subject_layer:
            self.subject_layer = SubjectPlusLayer(
                current_channels,
                n_unmix,
                n_subjects,
            )
            current_channels = n_unmix
        else:
            self.subject_layer = None

        if current_channels != n_unmix:
            raise ValueError(
                'SpatialModule must output n_unmix channels before temporal filtering. '
                f'Got {current_channels} channels with use_spatial_attention={use_spatial_attention!r}, '
                f'use_unmixing_layer={use_unmixing_layer}, use_subject_layer={use_subject_layer}, '
                f'n_input={n_input}, n_attention={n_attention}, n_unmix={n_unmix}. '
                'Enable use_unmixing_layer or use_subject_layer, or make the active input channels equal n_unmix.'
            )
        self.output_channels = current_channels

        # Temporal filter (always enabled)
        if temporal_filter_type == 'filtfilt':
            self.temporal_filter = FiltFiltConv1d(
                in_channels=self.output_channels,
                out_channels=self.output_channels,
                kernel_size=temporal_filter_kernel_size,
                padding='same',
                padding_mode=temporal_filter_padding_mode,
                groups=self.output_channels,
                bias=False,
            )
        elif temporal_filter_type == 'conv':
            self.temporal_filter = nn.Conv1d(
                in_channels=self.output_channels,
                out_channels=self.output_channels,
                kernel_size=temporal_filter_kernel_size,
                padding='same',
                padding_mode=temporal_filter_padding_mode,
                groups=self.output_channels,
                bias=False,
            )
        else:
            raise ValueError(f'Unknown temporal_filter_type: {temporal_filter_type}')

        # Initialize with FIR if bands are provided
        if temporal_filter_bands is not None:
            self._init_fir_filter(
                bands=temporal_filter_bands,
                fs=temporal_filter_fs,
                freeze=temporal_filter_freeze,
            )
            self.temporal_filter_kernel_type = 'fir'
        else:
            # Mark as 'temporal' for extract_temporal_filters
            self.temporal_filter_kernel_type = 'temporal'

        # Gating
        if self.tf_gated:
            init_alpha = min(max(tf_gate_init_alpha, 1e-6), 1 - 1e-6)
            init_gamma = math.log(init_alpha) - math.log(1.0 - init_alpha)
            if self.tf_gate_per_channel:
                self.tf_gate = nn.Parameter(torch.ones(self.output_channels) * init_gamma)
            else:
                self.tf_gate = nn.Parameter(torch.ones(1) * init_gamma)
        else:
            self.tf_gate = None

        # GELU activation after temporal filtering (optional)
        if self.tf_gelu:
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

    def _init_fir_filter(self, bands, fs, freeze):
        """Initialize temporal filter with FIR bandpass coefficients."""
        if bands is None or fs is None:
            raise ValueError('bands and fs must be provided for FIR initialization.')

        conv = self.temporal_filter
        n_filters = conv.out_channels
        kernel_size = conv.kernel_size[0]

        # Cycle bands to match number of filters
        bands_cycled = [bands[i % len(bands)] for i in range(n_filters)]
        fir_coeffs_list = []
        for band in bands_cycled:
            fir_coeffs = firwin(
                numtaps=kernel_size,
                cutoff=[band[0], band[1]],
                pass_zero='bandpass',
                fs=fs,
            )
            fir_coeffs_list.append(torch.from_numpy(fir_coeffs).float())
        weights = torch.stack(fir_coeffs_list)[:, None, :].to(conv.weight.device)
        conv.weight.data = weights

        if freeze:
            conv.weight.requires_grad = False

    def extract_spatial_filters(self):
        """Return overall spatial filters (and optional bias) for each subject.

        Returns:
            overall_spatial_filter_weight: (n_subjects, output_channels, n_input)
                Subject indices are 0-indexed: [0] = subject 0, [1] = subject 1, etc.
            overall_spatial_filter_bias: (n_subjects, output_channels, 1) or None
                Subject indices are 0-indexed: [0] = subject 0, [1] = subject 1, etc.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            device = self.temporal_filter.weight.device
            dtype = self.temporal_filter.weight.dtype

            # 1) Attention (n_att × n_in) or identity if disabled
            if self.self_attention is not None:
                ASoftmax = self.self_attention.get_spatial_filter()
                assert ASoftmax.dim() == 2
            else:
                ASoftmax = torch.eye(self.n_input, device=device, dtype=dtype)

            # 2) Unmix (or identity if disabled)
            if self.unmixing_layer is not None:
                W = self.unmixing_layer.weight.detach().clone().squeeze(-1) @ ASoftmax
                bias = None
                if self.unmixing_layer.bias is not None:
                    bias = self.unmixing_layer.bias.detach().clone()[:, None]
            else:
                W = ASoftmax
                bias = None

            # 3) Subject maps (or shared weights if disabled)
            if self.subject_layer is not None:
                A_subj = self.subject_layer.A.detach().clone()
                weights = []
                biases = []
                for s in range(self.n_subjects):
                    W_s = A_subj[s] @ W
                    weights.append(W_s)
                    if bias is not None:
                        biases.append(A_subj[s] @ bias)
                overall_spatial_filter_weight = torch.stack(weights, dim=0).cpu()
                overall_spatial_filter_bias = None
                if biases:
                    overall_spatial_filter_bias = torch.stack(biases, dim=0).cpu()
            else:
                overall_spatial_filter_weight = W[None, :, :].repeat(
                    self.n_subjects, 1, 1
                ).cpu()
                overall_spatial_filter_bias = None
                if bias is not None:
                    overall_spatial_filter_bias = bias[None, :, :].repeat(
                        self.n_subjects, 1, 1
                    ).cpu()

        self.train(was_training)
        return overall_spatial_filter_weight, overall_spatial_filter_bias

    def extract_temporal_filters(self):
        """Return temporal filters as torch tensor.

        Returns:
            temporal_filters: (output_channels, kernel_size)
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            temporal_filters = self.temporal_filter.weight
            temporal_filters = temporal_filters.detach().clone().cpu().squeeze(dim=1)
        self.train(was_training)
        return temporal_filters

    def forward(self, xs, get_in_out_temporal=False):
        """Run spatial/temporal front-end.

        Args:
            xs: Tuple of (meg, subject_idx) where meg is (B, C, T).
            get_in_out_temporal: If True, return (x_in, x_out) around temporal filter.
        """
        x, s = xs
        if self.self_attention is not None:
            x = self.self_attention(x)
        if self.unmixing_layer is not None:
            x = self.unmixing_layer(x)
        if self.subject_layer is not None:
            x = self.subject_layer(x, s)

        x_in = x
        x_conv = self.temporal_filter(x)

        if get_in_out_temporal:
            return x_in, x_conv

        if self.tf_gated and (self.tf_gate is not None):
            gate = torch.sigmoid(self.tf_gate)
            if gate.ndim == 1:  # per-channel
                gate = gate[None, :, None]
            x = x_in + gate * (x_conv - x_in)
        else:
            x = x_conv

        x = self.activation(x)
        return x


class TemporalModule(nn.Module):
    """Stack of temporal convolution blocks."""

    def __init__(self, n_unmix, n_block, n_blocks):
        super().__init__()

        conv_blocks = []
        for block_index in range(0, n_blocks):
            n_in = n_unmix if block_index == 0 else n_block
            conv_blocks.append(ConvBlock(n_in, n_block, block_index))
        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x):
        """Apply temporal conv blocks."""
        return self.conv_blocks(x)


class LISA(nn.Module):
    """Full LISA model: interpretable front + temporal tail + projection.

    Forward returns both the interpretable representation (Z1) and final embedding.
    """

    def __init__(
        self,
        n_channels_input,
        n_channels_attention,
        n_channels_unmix,
        use_spatial_attention,
        n_spatial_harmonics,
        coords_xyz,
        coords_xy_scaled,
        spatial_dropout_number,
        spatial_dropout_radius,
        n_subjects,
        n_channels_block,
        n_features,
        head_pool,
        head_stride,
        meg_sr,
        temporal_filter_type,
        temporal_filter_bands,
        temporal_filter_freeze,
        temporal_filter_kernel_size,
        temporal_filter_padding_mode,
        tf_gated,
        tf_gate_per_channel,
        tf_gate_init_alpha,
        tf_gelu,
        n_temporal_module_blocks,
        time_reduction='none',
        time_reduction_dim=None,
        time_reduction_input_length=None,
        time_reduction_hidden_dim=None,
        time_reduction_num_heads=1,
        use_unmixing_layer=True,
        use_subject_layer=True,
        dropout_center_low=0.1,
        dropout_center_high=0.9,
        seed=None,
    ):
        """Initialize the full LISA model.

        Args:
            n_channels_input: Number of MEG channels.
            n_channels_attention: Attention channels.
            n_channels_unmix: Unmix channels (K).
            use_spatial_attention: "2D", "3D", or None (disabled).
            n_spatial_harmonics: Spatial basis size.
            coords_xyz: Sensor coordinates (N, 3).
            coords_xy_scaled: Sensor layout (N, 2).
            spatial_dropout_number: Spatial dropout centers.
            spatial_dropout_radius: Spatial dropout radius.
            n_subjects: Number of subjects.
            n_channels_block: Channels in temporal blocks.
            n_features: Embedding size.
            head_pool: Head pooling mode.
            head_stride: Head stride.
            time_reduction: Optional trainable time reduction after the head.
            time_reduction_dim: Output sequence length for trainable time reduction.
            time_reduction_input_length: Input sequence length for explicit linear time reduction.
            time_reduction_hidden_dim: Hidden size for attention-style time reduction.
            time_reduction_num_heads: Number of heads for query attention time reduction.
            meg_sr: Sampling rate (Hz).
            temporal_filter_type: "conv" or "filtfilt".
            temporal_filter_bands: Optional list of (low, high) bands. If provided, initializes FIR filters; if None, uses random initialization.
            temporal_filter_freeze: Freeze temporal filter weights.
            temporal_filter_kernel_size: Kernel size in samples.
            temporal_filter_padding_mode: Padding mode for temporal filter.
            tf_gated: Enable residual gate around temporal filter.
            tf_gate_per_channel: Learn per-channel gates.
            tf_gate_init_alpha: Initial alpha for gate.
            tf_gelu: Enable GELU activation after temporal filtering.
            n_temporal_module_blocks: Number of temporal blocks in tail.
            use_unmixing_layer: Enable the unmixing layer inside the spatial module.
            use_subject_layer: Enable the subject-specific layer inside the spatial module.
            dropout_center_low: Lower bound for dropout center sampling range.
            dropout_center_high: Upper bound for dropout center sampling range.
            seed: Optional RNG seed for reproducible init of spatial attention (Z and dropout centers); when None, init is non-deterministic.
        """
        super().__init__()
        self.use_unmixing_layer = use_unmixing_layer
        self.use_subject_layer = use_subject_layer
        self.spatial_module = SpatialModule(
            n_input=n_channels_input,
            n_attention=n_channels_attention,
            n_unmix=n_channels_unmix,
            use_spatial_attention=use_spatial_attention,
            n_spatial_harmonics=n_spatial_harmonics,
            coords_xy_scaled=coords_xy_scaled,
            spatial_dropout_number=spatial_dropout_number,
            spatial_dropout_radius=spatial_dropout_radius,
            coords_xyz=coords_xyz,
            temporal_filter_type=temporal_filter_type,
            temporal_filter_bands=temporal_filter_bands,
            temporal_filter_fs=meg_sr,
            temporal_filter_freeze=temporal_filter_freeze,
            temporal_filter_kernel_size=temporal_filter_kernel_size,
            temporal_filter_padding_mode=temporal_filter_padding_mode,
            tf_gated=tf_gated,
            tf_gate_per_channel=tf_gate_per_channel,
            tf_gate_init_alpha=tf_gate_init_alpha,
            tf_gelu=tf_gelu,
            n_subjects=n_subjects,
            use_unmixing_layer=use_unmixing_layer,
            use_subject_layer=use_subject_layer,
            dropout_center_low=dropout_center_low,
            dropout_center_high=dropout_center_high,
            seed=seed,
        )
        front_end_output_channels = self.spatial_module.output_channels

        assert isinstance(n_temporal_module_blocks, int) and (
            n_temporal_module_blocks >= 0
        ), (
            f'n_temporal_module_blocks must be a non-negative integer. Got {n_temporal_module_blocks}'
        )
        if n_temporal_module_blocks == 0:
            self.temporal_module = nn.Identity()
            head_input_channels = front_end_output_channels
            if n_channels_block != front_end_output_channels:
                warnings.warn(
                    f'n_temporal_module_blocks=0: ignoring n_channels_block={n_channels_block} '
                    f'and using front-end output channels={front_end_output_channels} for ConvHead input. '
                    f'ConvHead will output n_features={n_features} channels regardless.',
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self.temporal_module = TemporalModule(
                n_unmix=front_end_output_channels,
                n_block=n_channels_block,
                n_blocks=n_temporal_module_blocks,
            )
            # When temporal blocks exist, head receives n_channels_block channels
            head_input_channels = n_channels_block

        self.feature_projection = ConvHead(
            n_channels=head_input_channels,
            n_features=n_features,
            pool=head_pool,
            head_stride=head_stride,
        )
        if time_reduction == 'none':
            self.time_reducer = None
        else:
            self.time_reducer = TimeReduction(
                reduction=time_reduction,
                n_features=n_features,
                output_length=time_reduction_dim,
                input_length=time_reduction_input_length,
                hidden_dim=time_reduction_hidden_dim,
                num_heads=time_reduction_num_heads,
            )

    def extract_spatial_filters(self):
        return self.spatial_module.extract_spatial_filters()

    def extract_temporal_filters(self):
        return self.spatial_module.extract_temporal_filters()

    def forward_interpretable(self, xs: torch.Tensor) -> torch.Tensor:
        """Returns Z1: topography-channel x time tensor after interpretable front."""
        return self.spatial_module(xs)

    def forward_tail(self, z: torch.Tensor) -> torch.Tensor:
        """Runs dilated convolutional stack + heads from Z1 to final embedding."""
        h = self.temporal_module(z)
        y = self.feature_projection(h)
        if self.time_reducer is not None:
            y = self.time_reducer(y)
        return y

    def forward(self, xs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full pass: returns both Z1 and embedding."""
        z = self.forward_interpretable(xs)
        y = self.forward_tail(z)
        return z, y
