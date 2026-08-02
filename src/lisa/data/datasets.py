"""Dataset utilities for MEG/embedding pairs."""

import torch
from torch.utils.data import Dataset


class DoubleDataset(Dataset):
    """Dataset returning (meg, subject_id, embedding, wav_index, dataset_idx).

    Args:
        meg: Dict-like mapping subset keys to MEG arrays (channels, time).
        hidden: Embedding array indexed by wav_index.
        df: DataFrame with window metadata (subject/session/story, indices).
        meg_sr: MEG sampling rate (Hz).
        meg_offset: Seconds to offset MEG window.
    """

    def __init__(self, meg, hidden, df, meg_sr, meg_offset=0):
        self.meg = meg
        self.hidden = hidden
        self.df = df
        self.meg_sr = meg_sr
        self.meg_offset = int(meg_offset * self.meg_sr)

    def __len__(self):
        return self.df.shape[0]

    def get_metadata(self, idxs):
        """Return per-row metadata for the given dataset row indices.

        Polymorphic on input shape: an integer returns a single-row Series, a
        sequence returns a DataFrame whose row k is the metadata for dataset
        row ``idxs[k]``. The dataframe's original row index is preserved on the
        returned slice (so the natural index is the dataset row index for any
        df with a default range index, which is the only case used here).
        """
        return self.df.iloc[idxs]

    def __getitem__(self, idx):
        row_df = self.get_metadata(idx)

        subject_id, session_id, story_id = (
            row_df['subject_id'],
            row_df['session_id'],
            row_df['story_id'],
        )
        # Convert 1-indexed subject_id from dataframe to 0-indexed for model
        sbj = torch.tensor(subject_id - 1, dtype=torch.long)
        subject_id = str(subject_id)
        subject_id = '0' + subject_id if len(subject_id) == 1 else subject_id
        subset = f'subject{subject_id}_session{session_id}_story{story_id}'

        meg_start, meg_stop = (
            row_df[f'meg{self.meg_sr}_start'],
            row_df[f'meg{self.meg_sr}_stop'],
        )
        meg_start, meg_stop = meg_start + self.meg_offset, meg_stop + self.meg_offset
        wav_index = row_df['wav_index']

        meg = torch.tensor(self.meg[subset][:, meg_start:meg_stop], dtype=torch.float32)
        hid = torch.tensor(self.hidden[wav_index], dtype=torch.float32)
        widx = torch.tensor(wav_index, dtype=torch.long)
        dataset_idx = torch.tensor(idx, dtype=torch.long)

        return meg, sbj, hid, widx, dataset_idx
