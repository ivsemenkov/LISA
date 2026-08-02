"""Training loop and utilities for LISA experiments."""

import os
import time
from pathlib import Path
from typing import Any, Tuple

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from lisa.plots.plot_filters import plot_filters_fft, plot_spatial_filters
from lisa.training.criteria import CLIPLoss, metrics, retrieval_topk


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params) for a torch model."""
    overall_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, overall_params


def _tensor_to_numpy_copy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor into NumPy-owned memory."""
    return tensor.detach().cpu().numpy().copy()


class Trainer:
    """Trainer for LISA models with CLIP loss, logging, and filter plotting.

    This class manages optimization, metrics logging, checkpointing,
    and filter visualization artifacts.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        audio_model: torch.nn.Module,
        clip_temperature: float,
        lr_fe: float,
        weight_decay: float,
        clip_temperature_lr: float,
        optim: str,
        checkpoint: str,
        bids_root: str,
        meg_format: str,
        sampling_rate: float,
        save_test_top_k: int,
        plot_filter_graphs: bool,
        max_branches_to_plot: int = 20,
        tf_unfreeze_epoch: int | None = None,
        lr_temporal_mult: float = 1.0,
        wd_temporal: float | None = None,
        tf_smooth: float = 0.0,
        experiment: Any = None,
    ):
        """Initialize trainer and optimizer state.

        Args:
            model: LISA model instance.
            audio_model: Audio-side embedding adapter (projection/pooling).
            clip_temperature: CLIP temperature value.
            lr_fe: Base learning rate.
            weight_decay: Base weight decay.
            clip_temperature_lr: Learning rate for CLIP temperature parameter.
            optim: Optimizer name ("Adam" or "AdamW").
            checkpoint: Checkpoint basename.
            bids_root: BIDS root or directory with MEG files.
            meg_format: "fif" or "bids".
            sampling_rate: MEG sampling rate (Hz).
            save_test_top_k: Top-K width for the per-window dump written at
                final test. Must be <= test candidate-bank size (validated
                upfront in ``fit``). ``rank_of_true`` is always saved
                separately and supports any post-hoc Top-K accuracy.
            max_branches_to_plot: Limit for plotting filters.
            plot_filter_graphs: Plot and save temporal/spatial filter graphs
                after each epoch.
            tf_unfreeze_epoch: Epoch index to unfreeze temporal filter params.
            lr_temporal_mult: LR multiplier for temporal filter params.
            wd_temporal: Optional weight decay for temporal filter params.
            tf_smooth: L2 smoothness penalty for temporal kernels.
            experiment: ClearMLExperiment-like logger.
        """
        if save_test_top_k < 1:
            raise ValueError(
                f'save_test_top_k must be a positive integer, got {save_test_top_k}.'
            )
        self.model = model
        self.audio_model = audio_model
        print(self.model)
        print(self.audio_model)
        model_trainable_n_params, model_overall_n_params = count_parameters(self.model)
        audio_trainable_n_params, audio_overall_n_params = count_parameters(
            self.audio_model
        )
        print(f'MEG model trainable parameters: {model_trainable_n_params}')
        print(f'MEG model overall parameters: {model_overall_n_params}')
        print(f'Audio model trainable parameters: {audio_trainable_n_params}')
        print(f'Audio model overall parameters: {audio_overall_n_params}')
        print(
            'Total trainable parameters: '
            f'{model_trainable_n_params + audio_trainable_n_params}'
        )
        print(
            'Total overall parameters: '
            f'{model_overall_n_params + audio_overall_n_params}'
        )
        self.experiment = experiment
        self.max_branches_to_plot = max_branches_to_plot
        self.plot_filter_graphs = plot_filter_graphs
        self.bids_root = bids_root
        self.meg_format = meg_format
        self.tf_unfreeze_epoch = tf_unfreeze_epoch
        self.tf_smooth = tf_smooth
        self.save_test_top_k = save_test_top_k

        # CLIP loss
        self.criterion = CLIPLoss(clip_temperature=clip_temperature)

        if wd_temporal is None:
            wd_temporal = weight_decay

        # Separate temporal filter parameters for different LR/WD
        tf_params = []
        sm = getattr(self.model, 'spatial_module', None)
        if sm is not None:
            if getattr(sm, 'temporal_filter', None) is not None:
                tf_params += list(sm.temporal_filter.parameters())
            if getattr(sm, 'tf_gate', None) is not None:
                tf_params.append(sm.tf_gate)

        tf_ids = {id(p) for p in tf_params}
        base_params = [p for p in self.model.parameters() if id(p) not in tf_ids]
        base_params += list(self.audio_model.parameters())

        parameters = [
            {'params': base_params, 'lr': lr_fe, 'weight_decay': weight_decay},
        ]

        if len(tf_params) > 0:
            parameters.append(
                {
                    'params': tf_params,
                    'lr': lr_fe * lr_temporal_mult,
                    'weight_decay': wd_temporal,
                }
            )

        parameters.append(
            {
                'params': self.criterion.parameters(),
                'lr': clip_temperature_lr,
                'weight_decay': 0,
            }
        )

        if optim == 'Adam':
            self.optimizer = torch.optim.Adam(parameters)
        elif optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(parameters)
        else:
            raise ValueError(f'Unknown optimizer: {optim}')

        self.save_path = os.path.join(experiment.dir, f'{checkpoint}.pt')
        self.audio_model_save_path = os.path.join(experiment.dir, 'audio_model_best.pt')
        self.criterion_save_path = os.path.join(experiment.dir, 'criterion_best.pt')
        self.audio_model_has_parameters = any(
            p.numel() > 0 for p in self.audio_model.parameters()
        )
        self.filters_save_path = os.path.join(experiment.dir, 'filters')
        Path(self.filters_save_path).mkdir(parents=True, exist_ok=True)

        self.sampling_rate = sampling_rate

    def _prepare_audio_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Run the separate audio model to match MEG-side outputs."""
        return self.audio_model(x)

    def _prepare_candidate_bank(
        self,
        hidden_bank: np.ndarray,
        candidate_ids: np.ndarray,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare an audio candidate bank and matching IDs for retrieval metrics."""
        hidden_bank_t = torch.tensor(hidden_bank, dtype=torch.float32).to(device)
        hidden_bank_t = einops.rearrange(hidden_bank_t, 'b t f -> b f t')
        candidate_ids_t = torch.tensor(candidate_ids, dtype=torch.long).to(device)
        if candidate_ids_t.ndim != 1:
            raise ValueError(
                f'candidate_ids must be 1D, got shape {candidate_ids_t.shape}.'
            )
        if candidate_ids_t.numel() != hidden_bank_t.shape[0]:
            raise ValueError(
                'candidate_ids length must match hidden bank size. '
                f'Got ids={candidate_ids_t.numel()}, hidden={hidden_bank_t.shape[0]}.'
            )
        with torch.no_grad():
            hidden_bank_t = self._prepare_audio_embeddings(hidden_bank_t)
        return hidden_bank_t, candidate_ids_t

    def switch_temporal_gradients(self, requires_grad: bool) -> None:
        """Enable/disable gradients for temporal filter parameters."""
        sm = getattr(self.model, 'spatial_module', None)
        if sm is None:
            return

        tf_layer = getattr(sm, 'temporal_filter', None)
        if tf_layer is not None:
            for p in tf_layer.parameters():
                p.requires_grad = requires_grad
        if hasattr(sm, 'tf_gate') and sm.tf_gate is not None:
            sm.tf_gate.requires_grad = requires_grad

    def evaluate(
        self,
        *,
        dataloader,
        hidden_bank: np.ndarray,
        candidate_ids: np.ndarray,
        split_size: int,
        device: torch.device,
        desc: str,
        save_top_k: int | None = None,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate a split without changing model weights."""
        if split_size <= 0:
            raise ValueError(f'{desc} split_size must be positive, got {split_size}.')

        self.model.eval()
        self.audio_model.eval()
        self.criterion.eval()
        hidden_metrics, candidate_ids_t = self._prepare_candidate_bank(
            hidden_bank=hidden_bank,
            candidate_ids=candidate_ids,
            device=device,
        )
        n_candidates = candidate_ids_t.numel()
        if n_candidates < 10:
            raise ValueError(
                f'{desc}: candidate count {n_candidates} is smaller than the '
                'top10 metric width.'
            )

        collect_per_window = save_top_k is not None
        if collect_per_window and not (1 <= save_top_k <= n_candidates):
            raise ValueError(
                f'{desc}: save_top_k={save_top_k} must be in '
                f'[1, candidate count={n_candidates}].'
            )
        retrieval_top_k = n_candidates if collect_per_window else 10

        loss_sum = 0.0
        seen = 0
        is_in_top10s, is_in_top1s = [], []
        per_window = None
        if collect_per_window:
            per_window = {
                'dataset_idxs': [],
                'wav_index_true': [],
                'rank_of_true': [],
                'similarity_to_true': [],
                'top_k_wav_indices': [],
                'top_k_similarities': [],
            }
        pbar = tqdm(dataloader, leave=False)
        with torch.no_grad():
            for meg, sbj, hidden, widx, idx in pbar:
                pbar.set_description(
                    desc=f'{desc} epoch {epoch}' if epoch is not None else desc
                )
                hidden = einops.rearrange(hidden, 'b t f -> b f t')
                meg, sbj, hidden, widx = (
                    meg.to(device),
                    sbj.to(device),
                    hidden.to(device),
                    widx.to(device),
                )
                hidden = self._prepare_audio_embeddings(hidden)
                _, result = self.model((meg, sbj))

                _, loss_temperature = self.criterion(
                    result, hidden, positive_ids=widx
                )
                batch_size = meg.shape[0]
                loss_sum += loss_temperature.detach().cpu().item() * batch_size
                seen += batch_size

                ranking = retrieval_topk(
                    result,
                    hidden_metrics,
                    widx,
                    candidate_ids=candidate_ids_t,
                    top_k=retrieval_top_k,
                )
                if collect_per_window:
                    is_in_top10 = (ranking['rank_of_true'] >= 0) & (
                        ranking['rank_of_true'] < 10
                    )
                    per_window['dataset_idxs'].append(_tensor_to_numpy_copy(idx))
                    per_window['wav_index_true'].append(_tensor_to_numpy_copy(widx))
                    per_window['rank_of_true'].append(
                        _tensor_to_numpy_copy(ranking['rank_of_true'])
                    )
                    per_window['similarity_to_true'].append(
                        _tensor_to_numpy_copy(ranking['similarity_to_true'])
                    )
                    per_window['top_k_wav_indices'].append(
                        _tensor_to_numpy_copy(
                            ranking['top_k_wav_indices'][:, :save_top_k]
                        )
                    )
                    per_window['top_k_similarities'].append(
                        _tensor_to_numpy_copy(
                            ranking['top_k_similarities'][:, :save_top_k]
                        )
                    )
                else:
                    is_in_top10 = ranking['is_in_topk']
                is_in_top1 = ranking['is_in_top1']
                is_in_top10s.append(is_in_top10.detach().cpu().numpy())
                is_in_top1s.append(is_in_top1.detach().cpu().numpy())
                pbar.set_postfix({f'{desc}_loss': f'{loss_sum / seen:.4f}'})

        if seen == 0:
            raise ValueError(f'{desc} dataloader produced no samples.')
        if seen != split_size:
            raise ValueError(
                f'{desc} dataloader produced {seen} samples but split_size={split_size}.'
            )
        result = {
            'loss': loss_sum / seen,
            'top10s': np.mean(np.concatenate(is_in_top10s)).item(),
            'top1s': np.mean(np.concatenate(is_in_top1s)).item(),
        }
        if per_window is not None:
            result['per_window'] = {
                'dataset_idxs': np.concatenate(per_window['dataset_idxs']).astype(
                    np.int64
                ),
                'wav_index_true': np.concatenate(per_window['wav_index_true']).astype(
                    np.int64
                ),
                'rank_of_true': np.concatenate(per_window['rank_of_true']).astype(
                    np.int64
                ),
                'similarity_to_true': np.concatenate(
                    per_window['similarity_to_true']
                ).astype(np.float32),
                'top_k_wav_indices': np.concatenate(
                    per_window['top_k_wav_indices'], axis=0
                ).astype(np.int64),
                'top_k_similarities': np.concatenate(
                    per_window['top_k_similarities'], axis=0
                ).astype(
                    np.float32
                ),
                'candidate_ids': candidate_ids_t.detach().cpu().numpy().astype(np.int64),
            }
        return result

    def _save_final_test_per_window(
        self,
        *,
        per_window: dict[str, np.ndarray],
        dataset_test,
    ) -> None:
        """Write per-test-window retrieval results next to ``metrics.csv``.

        Materializes window metadata through the same dataset row indices that
        were returned by ``__getitem__`` during evaluation. The saved
        ``dataset_idx`` column comes from the evaluated batch, not from the
        dataframe index.
        """
        idxs = per_window['dataset_idxs']
        meta = dataset_test.get_metadata(idxs).copy()
        if len(meta) != len(idxs):
            raise ValueError(
                'dataset_test.get_metadata(idxs) returned a different number '
                'of rows than were evaluated.'
            )
        if not np.array_equal(
            meta['wav_index'].to_numpy(dtype=np.int64),
            per_window['wav_index_true'],
        ):
            raise ValueError(
                'dataset_test.get_metadata(idxs) and the wav_index labels recorded '
                'during evaluate() disagree: dataset/dataloader out of sync.'
            )
        if 'dataset_idx' in meta.columns:
            raise ValueError('dataset metadata already contains a dataset_idx column.')

        top_k_wav_indices = per_window['top_k_wav_indices']
        top_k_similarities = per_window['top_k_similarities']
        meta.insert(0, 'dataset_idx', idxs)
        csv_df = meta.assign(
            rank_of_true=per_window['rank_of_true'],
            similarity_to_true=per_window['similarity_to_true'],
            top1_wav_index=top_k_wav_indices[:, 0],
            top1_similarity=top_k_similarities[:, 0],
        )
        csv_path = Path(self.experiment.dir) / 'final_test_per_window.csv'
        csv_df.to_csv(csv_path, index=False)

        npz_path = Path(self.experiment.dir) / 'final_test_per_window.npz'
        np.savez_compressed(
            npz_path,
            dataset_idxs=idxs,
            wav_index_true=per_window['wav_index_true'],
            rank_of_true=per_window['rank_of_true'],
            similarity_to_true=per_window['similarity_to_true'],
            top_k_wav_indices=top_k_wav_indices,
            top_k_similarities=top_k_similarities,
            candidate_ids=per_window['candidate_ids'],
        )

    def fit(
        self,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        hidden_train_bank: np.ndarray,
        hidden_val_bank: np.ndarray,
        hidden_test_bank: np.ndarray,
        train_candidate_ids: np.ndarray,
        val_candidate_ids: np.ndarray,
        test_candidate_ids: np.ndarray,
        train_size: int,
        validation_size: int,
        test_size: int,
        nepoch: int = 1,
        early_stopping_patience: int | None = None,
        device: torch.device = torch.device('cuda'),
    ) -> None:
        """Run training, validation checkpointing, aux test monitoring, and final test."""
        if self.save_test_top_k > test_candidate_ids.size:
            raise ValueError(
                f'save_test_top_k={self.save_test_top_k} exceeds the test '
                f'candidate-bank size {test_candidate_ids.size}; pick a smaller '
                'value via --save-test-top-k.'
            )
        torch.cuda.empty_cache()

        if self.tf_unfreeze_epoch is not None:
            self.switch_temporal_gradients(requires_grad=False)

        self.model = self.model.to(device)
        self.audio_model = self.audio_model.to(device)
        self.criterion = self.criterion.to(device)
        val_loss_min = float('inf')
        best_summary: dict[str, float | int] | None = None
        train_loss_epoch = 0
        epochs_without_val_improvement = 0
        refresh_train_metrics = any(
            p.requires_grad for p in self.audio_model.parameters()
        )

        for epoch in range(nepoch):
            if (self.tf_unfreeze_epoch is not None) and (
                epoch >= self.tf_unfreeze_epoch
            ):
                self.switch_temporal_gradients(requires_grad=True)

            self.model.train()
            self.audio_model.train()
            self.criterion.train()
            train_loss = 0
            train_seen = 0

            hidden_train_metrics, train_candidate_ids_t = self._prepare_candidate_bank(
                hidden_bank=hidden_train_bank,
                candidate_ids=train_candidate_ids,
                device=device,
            )

            # Tracking for temporal filter gradient logging
            tf_energy_ps_sum = 0.0
            tf_batch_count = 0
            tf_zero_batches = 0
            tf_sample_count = 0
            tf_param_numel = None

            pbar = tqdm(dataloader_train, leave=False)
            is_in_top10s_train, is_in_top1s_train = [], []
            for meg, sbj, hidden, widx, _idx in pbar:
                pbar.set_description(desc=f'train epoch {epoch}')

                hidden = einops.rearrange(hidden, 'b t f -> b f t')
                meg, sbj, hidden, widx = (
                    meg.to(device),
                    sbj.to(device),
                    hidden.to(device),
                    widx.to(device),
                )
                hidden = self._prepare_audio_embeddings(hidden)
                _, result = self.model((meg, sbj))

                _, loss_temperature = self.criterion(
                    result, hidden, positive_ids=widx
                )

                self.optimizer.zero_grad()

                loss_total = loss_temperature

                # Smoothness penalty for temporal filter
                if self.tf_smooth and self.tf_smooth > 0.0:
                    sm = getattr(self.model, 'spatial_module', None)
                    if sm is None:
                        raise RuntimeError(
                            'tf_smooth > 0.0 but no spatial_module found'
                        )
                    tfm = getattr(sm, 'temporal_filter', None)
                    if tfm is None or tfm.weight is None:
                        raise RuntimeError(
                            'tf_smooth > 0.0 but temporal_filter is missing'
                        )
                    w = tfm.weight  # (n_unmix, 1, kernel_size)
                    if w.shape[-1] >= 3:
                        d2 = w[:, :, 2:] - 2 * w[:, :, 1:-1] + w[:, :, :-2]
                        smooth = (d2**2).mean()
                        loss_total = loss_total + self.tf_smooth * smooth

                loss_total.backward()

                # Log temporal filter gradient statistics
                tfm = getattr(
                    getattr(self.model, 'spatial_module', None), 'temporal_filter', None
                )
                g = (
                    None
                    if (tfm is None or tfm.weight.grad is None)
                    else tfm.weight.grad
                )
                if g is not None:
                    with torch.no_grad():
                        gg = g.float()
                        if tf_param_numel is None:
                            tf_param_numel = gg.numel()
                        B = meg.shape[0]
                        E = (gg * gg).sum().item()
                        tf_energy_ps_sum += E * B
                        tf_batch_count += 1
                        tf_sample_count += B
                        if E < 1e-24:
                            tf_zero_batches += 1

                if refresh_train_metrics:
                    hidden_train_metrics, train_candidate_ids_t = (
                        self._prepare_candidate_bank(
                            hidden_bank=hidden_train_bank,
                            candidate_ids=train_candidate_ids,
                            device=device,
                        )
                    )

                self.optimizer.step()
                batch_loss = (
                    loss_temperature.detach().cpu().numpy().item() * meg.shape[0]
                )
                train_loss += batch_loss
                train_seen += meg.shape[0]

                with torch.no_grad():
                    is_in_top10_train, is_in_top1_train = metrics(
                        result,
                        hidden_train_metrics,
                        widx,
                        candidate_ids=train_candidate_ids_t,
                    )
                is_in_top10s_train.append(is_in_top10_train.detach().cpu().numpy())
                is_in_top1s_train.append(is_in_top1_train.detach().cpu().numpy())

                pbar.set_postfix(
                    {
                        'train_loss_epoch': f'{train_loss_epoch:.4f}',
                        'train loss': f'{batch_loss:.4f}',
                    }
                )

            if train_seen == 0:
                raise ValueError(
                    'Training dataloader produced no batches. '
                    'This can happen when drop_last=True and batch_size exceeds train size.'
                )
            top10s_train = np.mean(np.concatenate(is_in_top10s_train)).item()
            top1s_train = np.mean(np.concatenate(is_in_top1s_train)).item()

            train_loss /= train_seen

            if self.experiment is not None:
                self.experiment.log(
                    {
                        'loss_train': train_loss,
                        'top10s_train': top10s_train,
                        'top1s_train': top1s_train,
                    },
                    step=epoch,
                )

                # Log temporal filter gradient stats
                tfm = getattr(
                    getattr(self.model, 'spatial_module', None), 'temporal_filter', None
                )
                if tfm is not None and tf_batch_count > 0:
                    tf_grad_rms = (
                        (tf_energy_ps_sum / tf_sample_count) / tf_param_numel
                    ) ** 0.5
                    tf_grad_zero_frac = tf_zero_batches / tf_batch_count
                    tf_weight_l2 = tfm.weight.detach().norm().item()
                    self.experiment.log(
                        {
                            'tf_grad_rms': tf_grad_rms,
                            'tf_grad_zero_frac': tf_grad_zero_frac,
                            'tf_weight_l2': tf_weight_l2,
                        },
                        step=epoch,
                    )

                # Log gate statistics
                tfg = getattr(
                    getattr(self.model, 'spatial_module', None), 'tf_gate', None
                )
                if tfg is not None:
                    with torch.no_grad():
                        gate_raw = self.model.spatial_module.tf_gate.detach()
                        gate_sig = torch.sigmoid(gate_raw)
                        if gate_sig.ndim > 0 and gate_sig.numel() > 1:
                            self.experiment.log(
                                {
                                    'tf_gate_sigma_mean': gate_sig.mean().item(),
                                    'tf_gate_sigma_min': gate_sig.min().item(),
                                    'tf_gate_sigma_max': gate_sig.max().item(),
                                },
                                step=epoch,
                            )
                        else:
                            self.experiment.log(
                                {'tf_gate_sigma': gate_sig.item()}, step=epoch
                            )

            train_loss_epoch = train_loss
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

            time.sleep(1)
            val_stats = self.evaluate(
                dataloader=dataloader_val,
                hidden_bank=hidden_val_bank,
                candidate_ids=val_candidate_ids,
                split_size=validation_size,
                device=device,
                desc='val',
                epoch=epoch,
            )
            aux_test_stats = self.evaluate(
                dataloader=dataloader_test,
                hidden_bank=hidden_test_bank,
                candidate_ids=test_candidate_ids,
                split_size=test_size,
                device=device,
                desc='aux_test',
                epoch=epoch,
            )
            val_loss = val_stats['loss']
            if self.experiment is not None:
                self.experiment.log(
                    {
                        'loss_val': val_stats['loss'],
                        'top10s_val': val_stats['top10s'],
                        'top1s_val': val_stats['top1s'],
                        'loss_aux_test': aux_test_stats['loss'],
                        'top10s_aux_test': aux_test_stats['top10s'],
                        'top1s_aux_test': aux_test_stats['top1s'],
                    },
                    step=epoch,
                )

            val_improved = val_loss < val_loss_min
            if val_improved:
                payload = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'loss_val': val_loss,
                }
                torch.save(payload, self.save_path)
                if self.audio_model_has_parameters:
                    torch.save(
                        {
                            'epoch': epoch,
                            'audio_model_state_dict': self.audio_model.state_dict(),
                            'loss_val': val_loss,
                        },
                        self.audio_model_save_path,
                    )
                torch.save(
                    {
                        'epoch': epoch,
                        'criterion_state_dict': self.criterion.state_dict(),
                        'loss_val': val_loss,
                    },
                    self.criterion_save_path,
                )
                val_loss_min = val_loss
                best_summary = {'best_epoch': int(epoch)}

            # Save weights to file
            spatial_filters_weight, spatial_filters_bias = (
                self.model.extract_spatial_filters()
            )
            temporal_filters = self.model.extract_temporal_filters()
            assert spatial_filters_weight.shape[1] == temporal_filters.shape[0], (
                temporal_filters.shape,
                spatial_filters_weight.shape,
            )
            np_filters = {
                'overall_spatial_filter_weight': spatial_filters_weight,
                'temporal_filters_weight': temporal_filters,
            }
            if spatial_filters_bias is not None:
                np_filters['overall_spatial_filter_bias'] = spatial_filters_bias
            np.savez(
                os.path.join(
                    self.filters_save_path,
                    f'epoch_{epoch}_weights_val1s_{val_stats["top1s"]}.npz',
                ),
                **np_filters,
            )

            if self.plot_filter_graphs:
                temporal_fig = plot_filters_fft(
                    filters=temporal_filters[: self.max_branches_to_plot],
                    fs=self.sampling_rate,
                    out_path=None,
                    return_fig=True,
                )
                self.experiment.log_figure(
                    name='temporal_filters',
                    fig=temporal_fig,
                    step=epoch,
                    series='temporal',
                )
                plt.close(temporal_fig)

                for subj in range(spatial_filters_weight.shape[0]):
                    # subj is 0-indexed, but plot_spatial_filters expects 1-indexed for file names
                    spatial_fig = plot_spatial_filters(
                        filters=spatial_filters_weight[
                            subj, : self.max_branches_to_plot, :
                        ],
                        bids_root=self.bids_root,
                        subject=subj + 1,  # Convert to 1-indexed for file names/display
                        session='0',
                        task='0',
                        meg_format=self.meg_format,
                        meg_type='mag',
                        out_path=None,
                        return_fig=True,
                        add_colorbar=False,
                    )
                    self.experiment.log_figure(
                        name=f'spatial_filters_subj_{subj + 1}',  # Display as 1-indexed
                        fig=spatial_fig,
                        step=epoch,
                        series='spatial',
                    )
                    plt.close(spatial_fig)

            torch.cuda.empty_cache()

            if early_stopping_patience is not None:
                if val_improved:
                    epochs_without_val_improvement = 0
                else:
                    epochs_without_val_improvement += 1
                    if epochs_without_val_improvement >= early_stopping_patience:
                        if self.experiment is not None:
                            self.experiment.log(
                                {'early_stop_epoch': int(epoch)},
                                step=epoch,
                            )
                        break

        if best_summary is None:
            raise RuntimeError('No best validation checkpoint was saved.')

        checkpoint = torch.load(self.save_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.audio_model_has_parameters:
            audio_checkpoint = torch.load(
                self.audio_model_save_path, map_location=device, weights_only=False
            )
            self.audio_model.load_state_dict(audio_checkpoint['audio_model_state_dict'])
        criterion_checkpoint = torch.load(
            self.criterion_save_path, map_location=device, weights_only=False
        )
        self.criterion.load_state_dict(criterion_checkpoint['criterion_state_dict'])
        final_test_stats = self.evaluate(
            dataloader=dataloader_test,
            hidden_bank=hidden_test_bank,
            candidate_ids=test_candidate_ids,
            split_size=test_size,
            device=device,
            desc='final_test',
            save_top_k=self.save_test_top_k,
            epoch=None,
        )
        self._save_final_test_per_window(
            per_window=final_test_stats['per_window'],
            dataset_test=dataloader_test.dataset,
        )
        if self.experiment is not None:
            self.experiment.log(
                {
                    'loss_final_test': float(final_test_stats['loss']),
                    'top10s_final_test': float(final_test_stats['top10s']),
                    'top1s_final_test': float(final_test_stats['top1s']),
                },
                step=best_summary['best_epoch'],
            )
