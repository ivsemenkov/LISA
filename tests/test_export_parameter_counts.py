import json

import torch

from lisa.cli import export_parameter_counts as parameter_export


def _base_config(**updates):
    config = {
        'run_group': '2conv-seed42-paper',
        'run_name': '2conv-seed42-paper-25branches',
        'seed': 42,
        'n_temporal_module_blocks': 2,
        'n_channels_unmix': 25,
        'n_channels_block': 25,
        'n_channels_attention': 270,
        'use_spatial_attention': '3D',
        'use_unmixing_layer': True,
        'use_subject_layer': True,
        'temporal_filter_type': 'conv',
        'temporal_filter_kernel_size': 15,
        'temporal_filter_freeze': False,
        'head_pool': 'single_conv',
        'head_stride': 2,
        'n_features': 768,
        'feature_reduction': 'none',
        'feature_reduction_dim': None,
        'time_reduction': 'none',
        'time_reduction_dim': None,
        'time_reduction_input_length': None,
        'audio_time_reduction_input_length': None,
        'time_reduction_hidden_dim': None,
        'time_reduction_num_heads': 1,
        'window_tag': '',
        'clip_temperature': 1.0,
    }
    config.update(updates)
    return config


def test_classify_run_covers_paper_families():
    assert parameter_export.classify_run(_base_config(), '2conv-seed42-paper') == (
        'branch_depth_sweep',
        '2conv',
        '',
    )
    assert parameter_export.classify_run(
        _base_config(seed=43), '2conv-seed43-paper'
    ) == ('seed_replication', '2conv', '')
    assert parameter_export.classify_run(
        _base_config(feature_reduction='linear'), 'lineardr32-2conv-seed42-paper'
    ) == ('feature_reduction', 'feature_linear', '')
    assert parameter_export.classify_run(
        _base_config(time_reduction='query_attn'),
        'timepool-queryattention-k8-heads4-2conv-seed42-paper',
    ) == ('time_reduction', 'time_query_attn', '')
    assert parameter_export.classify_run(
        _base_config(window_tag='w5gpu'), '2conv-seed42-paper-w5gpu'
    ) == ('segment_duration', 'duration_gpu', '')
    assert parameter_export.classify_run(
        _base_config(use_subject_layer=False),
        'ablate-nosubject-2conv-seed42-paper',
    ) == ('architecture_ablation', 'nosubject', 'nosubject')


def test_summarize_run_partitions_meg_audio_and_criterion(tmp_path, monkeypatch):
    class DummyMegModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 3)
            self.feature_projection = torch.nn.Linear(3, 4)
            self.time_reducer = torch.nn.Linear(4, 1, bias=False)

    class DummyAudioAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_reducer = torch.nn.Linear(5, 2)
            self.time_reducer = torch.nn.Linear(2, 1, bias=False)

    monkeypatch.setattr(
        parameter_export,
        'load_from_config',
        lambda hyper_params, checkpoint_path, device: (DummyMegModel(), hyper_params),
    )
    monkeypatch.setattr(
        parameter_export,
        'build_audio_model_from_config',
        lambda hyper_params, device: DummyAudioAdapter(),
    )

    run_dir = (
        tmp_path
        / '2conv-seed42-paper'
        / ('2conv-seed42-paper-25branches_offline-deadbeef')
    )
    run_dir.mkdir(parents=True)
    config_path = run_dir / 'config.json'
    config_path.write_text(json.dumps(_base_config()), encoding='utf-8')
    (run_dir / 'metrics.csv').write_text(
        'epoch,loss_final_test,top1s_final_test,top10s_final_test\n'
        '0,,,\n'
        '3,0.25,0.4,0.7\n',
        encoding='utf-8',
    )

    row = parameter_export.summarize_run(config_path)

    assert row['meg_core_total_params'] == 9
    assert row['meg_projection_resampler_total_params'] == 16
    assert row['meg_time_reducer_total_params'] == 4
    assert row['meg_model_total_params'] == 29
    assert row['audio_feature_reducer_total_params'] == 12
    assert row['audio_time_reducer_total_params'] == 2
    assert row['audio_other_total_params'] == 0
    assert row['audio_adapter_total_params'] == 14
    assert row['total_params'] == 43
    assert row['trainable_params'] == 43
    assert row['criterion_trainable_params'] == 1
    assert row['optimizer_trainable_params'] == 44
    assert row['run_id'] == 'deadbeef'
    assert row['metrics_status'] == 'ok'
    assert row['best_epoch'] == 3
    assert row['loss_final_test'] == 0.25
    assert row['top1s_final_test'] == 0.4
    assert row['top10s_final_test'] == 0.7
    assert row['top1_final_test_pct'] == 40.0
    assert row['top10_final_test_pct'] == 70.0


def test_read_final_test_metrics_marks_missing_file(tmp_path):
    metrics = parameter_export._read_final_test_metrics(tmp_path)

    assert metrics['metrics_status'] == 'missing'
    assert metrics['metrics_error'] == 'metrics.csv not found'
    assert metrics['best_epoch'] == ''


def test_architecture_id_ignores_seed_and_window_tag():
    baseline = _base_config()
    replication = _base_config(seed=47, window_tag='w5gpu')

    assert parameter_export._architecture_id(
        baseline
    ) == parameter_export._architecture_id(replication)
