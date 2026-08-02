"""Comprehensive tests for all LISA model initialization variations.

This test suite ensures that all combinations of model parameters can be
initialized correctly and process batches without errors, with expected output shapes.
"""

import numpy as np
import os
import pytest
import torch

from lisa.model.nn_modules import ConvHead, LISA
from lisa.utils.constants import (
    N_MEG_CHANNELS,
    N_SUBJECTS,
    N_FEATURES,
    PREPROCESSED_DATA_DIR,
)
from lisa.cli.parse_experiment_args import parse_experiment_arguments


# Get default hyperparameters from CLI parser.
# Pass --run-group (required) and --device cpu so collection works without CUDA;
# model variation tests exercise CPU tensors only.
_default_args = parse_experiment_arguments(
    ['--run-group', 'test-unmix', '--device', 'cpu']
)

# Use constants and defaults from the codebase
N_CHANNELS_INPUT = N_MEG_CHANNELS
N_CHANNELS_ATTENTION = _default_args['n_channels_attention']
N_CHANNELS_UNMIX = _default_args['n_channels_unmix']
N_CHANNELS_BLOCK = _default_args['n_channels_block']
MEG_SR = float(_default_args['meg_sr'])
N_SPATIAL_HARMONICS = _default_args['n_spatial_harmonics']
SPATIAL_DROPOUT_NUMBER = int(_default_args['spatial_dropout_number'])
SPATIAL_DROPOUT_RADIUS = _default_args['spatial_dropout_radius']
TEMPORAL_FILTER_KERNEL_SIZE = _default_args['temporal_filter_kernel_size']
TEMPORAL_FILTER_PADDING_MODE = _default_args['temporal_filter_padding_mode']
TF_GATE_INIT_ALPHA = _default_args['tf_gate_init_alpha']
HEAD_STRIDE = _default_args['head_stride']

# Test-specific constants (not in defaults, but needed for testing)
BATCH_SIZE = 4
TIME_STEPS = int(3 * MEG_SR)  # 3 seconds of data at sampling rate


def get_coords():
    """Load or generate coordinate data for testing."""
    coords_dir = os.path.join(PREPROCESSED_DATA_DIR, 'coords')
    # Try to load real coordinates if available
    coords_xyz_path = os.path.join(coords_dir, 'sensor_xyz.npy')
    coords_xy_path = os.path.join(coords_dir, 'coords208_xy_scaled.npy')

    if os.path.exists(coords_xyz_path) and os.path.exists(coords_xy_path):
        coords_xyz = np.load(coords_xyz_path)
        coords_xy_scaled = np.load(coords_xy_path)
    else:
        # Generate synthetic coordinates for testing
        np.random.seed(42)
        coords_xyz = np.random.randn(N_CHANNELS_INPUT, 3)
        coords_xyz = coords_xyz / np.linalg.norm(coords_xyz, axis=1, keepdims=True)

        coords_xy_scaled = np.random.uniform(-1, 1, size=(N_CHANNELS_INPUT, 2))

    return coords_xyz, coords_xy_scaled


@pytest.fixture
def coords():
    """Fixture providing coordinate data."""
    return get_coords()


@pytest.fixture
def dummy_batch():
    """Create a dummy batch for testing."""
    meg_data = torch.randn(BATCH_SIZE, N_CHANNELS_INPUT, TIME_STEPS)
    subject_indices = torch.randint(0, N_SUBJECTS, (BATCH_SIZE,))
    return (meg_data, subject_indices)


def create_lisa_model(
    use_spatial_attention=None,
    temporal_filter_bands=None,
    head_pool=None,
    n_temporal_module_blocks=None,
    temporal_filter_type=None,
    tf_gated=None,
    tf_gate_per_channel=None,
    tf_gelu=None,
    temporal_filter_freeze=None,
    coords_xyz=None,
    coords_xy_scaled=None,
    **kwargs,
):
    """Helper function to create a LISA model with specified parameters.

    Uses defaults from parse_experiment_args.py, only overriding specified parameters.
    Any additional kwargs are passed directly to LISA constructor.
    """
    if coords_xyz is None or coords_xy_scaled is None:
        coords_xyz, coords_xy_scaled = get_coords()

    # Use defaults from CLI parser, override only specified parameters
    model_kwargs = {
        'n_channels_input': N_CHANNELS_INPUT,
        'n_channels_attention': N_CHANNELS_ATTENTION,
        'n_channels_unmix': N_CHANNELS_UNMIX,
        'use_spatial_attention': (
            use_spatial_attention
            if use_spatial_attention is not None
            else _default_args['use_spatial_attention']
        ),
        'n_spatial_harmonics': N_SPATIAL_HARMONICS,
        'coords_xyz': coords_xyz,
        'coords_xy_scaled': coords_xy_scaled,
        'spatial_dropout_number': SPATIAL_DROPOUT_NUMBER,
        'spatial_dropout_radius': SPATIAL_DROPOUT_RADIUS,
        'n_subjects': N_SUBJECTS,
        'n_channels_block': N_CHANNELS_BLOCK,
        'n_features': N_FEATURES,
        'head_pool': (
            head_pool if head_pool is not None else _default_args['head_pool']
        ),
        'head_stride': HEAD_STRIDE,
        'meg_sr': MEG_SR,
        'temporal_filter_type': (
            temporal_filter_type
            if temporal_filter_type is not None
            else _default_args['temporal_filter_type']
        ),
        'temporal_filter_bands': (
            temporal_filter_bands
            if temporal_filter_bands is not None
            else _default_args['temporal_filter_bands']
        ),
        'temporal_filter_freeze': (
            temporal_filter_freeze
            if temporal_filter_freeze is not None
            else _default_args['temporal_filter_freeze']
        ),
        'temporal_filter_kernel_size': TEMPORAL_FILTER_KERNEL_SIZE,
        'temporal_filter_padding_mode': TEMPORAL_FILTER_PADDING_MODE,
        'tf_gated': (tf_gated if tf_gated is not None else _default_args['tf_gated']),
        'tf_gate_per_channel': (
            tf_gate_per_channel
            if tf_gate_per_channel is not None
            else _default_args['tf_gate_per_channel']
        ),
        'tf_gate_init_alpha': TF_GATE_INIT_ALPHA,
        'tf_gelu': (tf_gelu if tf_gelu is not None else _default_args['tf_gelu']),
        'n_temporal_module_blocks': (
            n_temporal_module_blocks
            if n_temporal_module_blocks is not None
            else _default_args['n_temporal_module_blocks']
        ),
    }
    # Allow overriding any parameter via kwargs
    model_kwargs.update(kwargs)
    if (
        model_kwargs.get('time_reduction') == 'linear'
        and model_kwargs.get('time_reduction_input_length') is None
    ):
        head = ConvHead(
            n_channels=1,
            n_features=1,
            pool=model_kwargs['head_pool'],
            head_stride=model_kwargs['head_stride'],
        )
        model_kwargs['time_reduction_input_length'] = head.output_length(TIME_STEPS)

    model = LISA(**model_kwargs)
    return model


class TestAttentionTypes:
    """Test different spatial attention types."""

    @pytest.mark.parametrize('attention_type', [None, '2D', '3D'])
    def test_attention_type_initialization(self, attention_type, coords, dummy_batch):
        """Test that each attention type can be initialized and forward passes."""
        coords_xyz, coords_xy_scaled = coords
        model = create_lisa_model(
            use_spatial_attention=attention_type,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Forward pass
        z, y = model(dummy_batch)

        # Check shapes
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS), (
            f'Z1 shape incorrect for attention_type={attention_type}: {z.shape}'
        )
        # Output shape depends on head_stride and pooling
        assert y.shape[0] == BATCH_SIZE, f'Batch size mismatch: {y.shape[0]}'
        assert y.shape[1] == N_FEATURES, f'Feature size mismatch: {y.shape[1]}'
        # Time dimension should be reduced by stride (approximately)
        assert y.shape[2] > 0, f'Time dimension should be > 0, got {y.shape[2]}'
        assert y.shape[2] <= TIME_STEPS, (
            f'Time dimension should be <= input, got {y.shape[2]}'
        )
        # With head_stride, output should be roughly divided by stride (accounting for padding/kernel)
        expected_time_approx = TIME_STEPS // HEAD_STRIDE
        assert y.shape[2] >= expected_time_approx - 10, (
            f'Time dimension too small: {y.shape[2]} (expected ~{expected_time_approx} with stride={HEAD_STRIDE})'
        )


class TestFIRBands:
    """Test FIR band initialization."""

    @pytest.mark.parametrize('use_fir', [False, True])
    def test_fir_bands(self, use_fir, coords, dummy_batch):
        """Test with and without FIR band initialization."""
        coords_xyz, coords_xy_scaled = coords

        if use_fir:
            # Use multiple frequency bands
            temporal_filter_bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)]
        else:
            temporal_filter_bands = None

        model = create_lisa_model(
            temporal_filter_bands=temporal_filter_bands,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Forward pass
        z, y = model(dummy_batch)

        # Check shapes
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES


class TestConvHead:
    """Test different ConvHead pooling types."""

    @pytest.mark.parametrize('head_pool', ['single_conv', 'max', 'conv'])
    def test_conv_head_pool_types(self, head_pool, coords, dummy_batch):
        """Test all ConvHead pool types."""
        coords_xyz, coords_xy_scaled = coords
        model = create_lisa_model(
            head_pool=head_pool,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Forward pass
        z, y = model(dummy_batch)

        # Check shapes
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES


class TestTemporalModuleBlocks:
    """Test different numbers of temporal module blocks."""

    @pytest.mark.parametrize('n_blocks', [0, 1, 2, 3, 4])
    def test_temporal_module_blocks(self, n_blocks, coords, dummy_batch):
        """Test different numbers of temporal blocks, including 0."""
        coords_xyz, coords_xy_scaled = coords
        model = create_lisa_model(
            n_temporal_module_blocks=n_blocks,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Forward pass
        z, y = model(dummy_batch)

        # Check shapes
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES


class TestTemporalFilterTypes:
    """Test different temporal filter types."""

    @pytest.mark.parametrize('filter_type', ['conv', 'filtfilt'])
    def test_temporal_filter_types(self, filter_type, coords, dummy_batch):
        """Test conv and filtfilt temporal filter types."""
        coords_xyz, coords_xy_scaled = coords
        model = create_lisa_model(
            temporal_filter_type=filter_type,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Forward pass
        z, y = model(dummy_batch)

        # Check shapes
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES


class TestTemporalFilterGating:
    """Test temporal filter gating options."""

    @pytest.mark.parametrize(
        'tf_gated,tf_gate_per_channel',
        [
            (False, False),
            (True, False),
            (True, True),
        ],
    )
    def test_temporal_filter_gating(
        self, tf_gated, tf_gate_per_channel, coords, dummy_batch
    ):
        """Test gating options for temporal filter."""
        coords_xyz, coords_xy_scaled = coords
        model = create_lisa_model(
            tf_gated=tf_gated,
            tf_gate_per_channel=tf_gate_per_channel,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Forward pass
        z, y = model(dummy_batch)

        # Check shapes
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES


class TestTemporalFilterGELU:
    """Test GELU activation after temporal filtering."""

    @pytest.mark.parametrize('tf_gelu', [False, True])
    def test_temporal_filter_gelu(self, tf_gelu, coords, dummy_batch):
        """Test GELU activation option."""
        coords_xyz, coords_xy_scaled = coords
        model = create_lisa_model(
            tf_gelu=tf_gelu,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Forward pass
        z, y = model(dummy_batch)

        # Check shapes
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES


class TestFIRFreeze:
    """Test FIR filter freezing."""

    @pytest.mark.parametrize('freeze', [False, True])
    def test_fir_freeze(self, freeze, coords, dummy_batch):
        """Test freezing FIR filter weights."""
        coords_xyz, coords_xy_scaled = coords
        temporal_filter_bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)]

        model = create_lisa_model(
            temporal_filter_bands=temporal_filter_bands,
            temporal_filter_freeze=freeze,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Check if weights are frozen
        if freeze:
            assert not model.spatial_module.temporal_filter.weight.requires_grad

        # Forward pass
        z, y = model(dummy_batch)

        # Check shapes
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES


class TestComprehensiveCombinations:
    """Test comprehensive combinations of parameters."""

    @pytest.mark.parametrize('attention_type', [None, '2D', '3D'])
    @pytest.mark.parametrize('use_fir', [False, True])
    @pytest.mark.parametrize('head_pool', ['single_conv', 'max', 'conv'])
    @pytest.mark.parametrize('n_blocks', [0, 1, 3])
    @pytest.mark.parametrize('filter_type', ['conv', 'filtfilt'])
    def test_comprehensive_combinations(
        self,
        attention_type,
        use_fir,
        head_pool,
        n_blocks,
        filter_type,
        coords,
        dummy_batch,
    ):
        """Test comprehensive combinations of key parameters."""
        coords_xyz, coords_xy_scaled = coords

        temporal_filter_bands = (
            [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)] if use_fir else None
        )

        model = create_lisa_model(
            use_spatial_attention=attention_type,
            temporal_filter_bands=temporal_filter_bands,
            head_pool=head_pool,
            n_temporal_module_blocks=n_blocks,
            temporal_filter_type=filter_type,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Forward pass
        z, y = model(dummy_batch)

        # Check shapes
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS), (
            f'Z1 shape incorrect for combination: attention={attention_type}, '
            f'fir={use_fir}, pool={head_pool}, blocks={n_blocks}, filter={filter_type}'
        )
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES


class TestEdgeCases:
    """Test edge cases and special configurations."""

    def test_zero_temporal_blocks_with_different_head_pools(self, coords, dummy_batch):
        """Test zero temporal blocks with all head pool types."""
        coords_xyz, coords_xy_scaled = coords

        for head_pool in ['single_conv', 'max', 'conv']:
            model = create_lisa_model(
                n_temporal_module_blocks=0,
                head_pool=head_pool,
                coords_xyz=coords_xyz,
                coords_xy_scaled=coords_xy_scaled,
            )
            model.eval()

            z, y = model(dummy_batch)
            assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
            assert y.shape[0] == BATCH_SIZE
            assert y.shape[1] == N_FEATURES

    def test_fir_with_gating(self, coords, dummy_batch):
        """Test FIR initialization with gating enabled."""
        coords_xyz, coords_xy_scaled = coords
        temporal_filter_bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)]

        model = create_lisa_model(
            temporal_filter_bands=temporal_filter_bands,
            tf_gated=True,
            tf_gate_per_channel=True,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        z, y = model(dummy_batch)
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES

    def test_all_features_combined(self, coords, dummy_batch):
        """Test a configuration with all optional features enabled."""
        coords_xyz, coords_xy_scaled = coords
        temporal_filter_bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)]

        model = create_lisa_model(
            use_spatial_attention='3D',
            temporal_filter_bands=temporal_filter_bands,
            head_pool='conv',
            n_temporal_module_blocks=3,
            temporal_filter_type='filtfilt',
            tf_gated=True,
            tf_gate_per_channel=True,
            tf_gelu=True,
            temporal_filter_freeze=False,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        z, y = model(dummy_batch)
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES

    def test_training_mode_forward(self, coords, dummy_batch):
        """Test that model works in training mode (for dropout)."""
        coords_xyz, coords_xy_scaled = coords

        # Create model with spatial dropout enabled
        model = create_lisa_model(
            use_spatial_attention='2D',
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.train()  # Enable training mode

        # Forward pass should work even with dropout
        z, y = model(dummy_batch)
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES

    def test_forward_interpretable_and_tail_separately(self, coords, dummy_batch):
        """Test forward_interpretable and forward_tail methods separately."""
        coords_xyz, coords_xy_scaled = coords
        model = create_lisa_model(
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        # Test forward_interpretable
        z = model.forward_interpretable(dummy_batch)
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)

        # Test forward_tail
        y = model.forward_tail(z)
        assert y.shape[0] == BATCH_SIZE
        assert y.shape[1] == N_FEATURES

        # Test full forward (should match)
        z_full, y_full = model(dummy_batch)
        assert torch.allclose(z, z_full)
        assert torch.allclose(y, y_full)

    def test_forward_with_time_linear_reduction(self, coords, dummy_batch):
        """Test trainable time reduction after the MEG head."""
        coords_xyz, coords_xy_scaled = coords

        base_model = create_lisa_model(
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        base_model.eval()
        _, base_y = base_model(dummy_batch)

        model = create_lisa_model(
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
            time_reduction='linear',
            time_reduction_dim=32,
        )
        model.eval()

        z, y = model(dummy_batch)
        assert z.shape == (BATCH_SIZE, N_CHANNELS_UNMIX, TIME_STEPS)
        assert y.shape == (BATCH_SIZE, N_FEATURES, 32)


class TestFilterExtraction:
    """Test filter extraction methods."""

    def test_extract_spatial_filters(self, coords):
        """Test spatial filter extraction for different attention types."""
        coords_xyz, coords_xy_scaled = coords

        for attention_type in [None, '2D', '3D']:
            model = create_lisa_model(
                use_spatial_attention=attention_type,
                coords_xyz=coords_xyz,
                coords_xy_scaled=coords_xy_scaled,
            )
            model.eval()

            weights, biases = model.extract_spatial_filters()
            assert weights.shape == (N_SUBJECTS, N_CHANNELS_UNMIX, N_CHANNELS_INPUT)
            assert biases.shape == (N_SUBJECTS, N_CHANNELS_UNMIX, 1)

    def test_extract_temporal_filters(self, coords):
        """Test temporal filter extraction."""
        coords_xyz, coords_xy_scaled = coords

        # Test with random initialization
        model = create_lisa_model(
            temporal_filter_bands=None,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model.eval()

        filters = model.extract_temporal_filters()
        assert filters.shape == (N_CHANNELS_UNMIX, TEMPORAL_FILTER_KERNEL_SIZE)

        # Test with FIR initialization
        temporal_filter_bands = [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)]
        model_fir = create_lisa_model(
            temporal_filter_bands=temporal_filter_bands,
            coords_xyz=coords_xyz,
            coords_xy_scaled=coords_xy_scaled,
        )
        model_fir.eval()

        filters_fir = model_fir.extract_temporal_filters()
        assert filters_fir.shape == (N_CHANNELS_UNMIX, TEMPORAL_FILTER_KERNEL_SIZE)
