import numpy as np
import pytest


def test_validation_split_holds_out_requested_story_sound():
    import pandas as pd

    from lisa.cli.train_model import _split_train_val_indices

    df = pd.DataFrame(
        {
            'story_id': [1, 1, 3, 3],
            'sound_id': [0, 1, 4, 4],
            'wav_index': [0, 1, 2, 3],
        }
    )

    train_indices, val_indices = _split_train_val_indices(df, [(3, 4)])
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    assert set(zip(val_df['story_id'], val_df['sound_id'])) == {(3, 4)}
    assert set(train_df['wav_index']).isdisjoint(set(val_df['wav_index']))


def test_validation_split_fails_for_missing_story_sound():
    import pandas as pd

    from lisa.cli.train_model import _split_train_val_indices

    df = pd.DataFrame({'story_id': [1], 'sound_id': [0], 'wav_index': [0]})

    with pytest.raises(ValueError, match='not present'):
        _split_train_val_indices(df, [(3, 4)])


def test_wav_index_bounds_fail_fast():
    import pandas as pd

    from lisa.cli.train_model import _validate_wav_indices

    df = pd.DataFrame({'wav_index': [0, 2]})
    hidden = np.zeros((2, 3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match='out of bounds'):
        _validate_wav_indices(df, hidden, 'train')
