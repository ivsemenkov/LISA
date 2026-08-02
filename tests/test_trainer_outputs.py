import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


def test_tensor_to_numpy_copy_does_not_alias_source_tensor():
    from lisa.training.trainer import _tensor_to_numpy_copy

    tensor = torch.arange(3)
    array = _tensor_to_numpy_copy(tensor)
    tensor[0] = 99

    assert array.tolist() == [0, 1, 2]


def test_final_test_per_window_uses_dataset_idx_for_duplicate_wav_index(
    tmp_path, monkeypatch
):
    plot_filters_stub = ModuleType('lisa.plots.plot_filters')
    plot_filters_stub.plot_filters_fft = lambda *args, **kwargs: None
    plot_filters_stub.plot_spatial_filters = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'lisa.plots.plot_filters', plot_filters_stub)
    from lisa.training.trainer import Trainer

    class DummyDataset:
        def __init__(self):
            self.df = pd.DataFrame(
                {
                    'subject_id': [1, 2, 3],
                    'session_id': [1, 1, 2],
                    'story_id': [10, 10, 11],
                    'wav_index': [5, 5, 7],
                },
                index=[100, 200, 300],
            )

        def get_metadata(self, idxs):
            return self.df.iloc[idxs]

    per_window = {
        'dataset_idxs': np.array([1, 0], dtype=np.int64),
        'wav_index_true': np.array([5, 5], dtype=np.int64),
        'rank_of_true': np.array([3, 0], dtype=np.int64),
        'similarity_to_true': np.array([0.2, 0.9], dtype=np.float32),
        'top_k_wav_indices': np.array([[8, 5], [5, 9]], dtype=np.int64),
        'top_k_similarities': np.array([[0.7, 0.2], [0.9, 0.1]], dtype=np.float32),
        'candidate_ids': np.array([5, 7, 8, 9], dtype=np.int64),
    }

    trainer = SimpleNamespace(experiment=SimpleNamespace(dir=tmp_path))
    Trainer._save_final_test_per_window(
        trainer, per_window=per_window, dataset_test=DummyDataset()
    )

    csv_df = pd.read_csv(tmp_path / 'final_test_per_window.csv')
    assert csv_df['dataset_idx'].tolist() == [1, 0]
    assert csv_df['subject_id'].tolist() == [2, 1]
    assert csv_df['wav_index'].tolist() == [5, 5]
    assert csv_df['rank_of_true'].tolist() == [3, 0]

    with np.load(tmp_path / 'final_test_per_window.npz') as npz:
        assert npz['dataset_idxs'].tolist() == [1, 0]
        assert npz['wav_index_true'].tolist() == [5, 5]
        assert npz['rank_of_true'].tolist() == [3, 0]
        assert npz['top_k_wav_indices'].tolist() == [[8, 5], [5, 9]]
