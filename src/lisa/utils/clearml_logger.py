"""ClearML experiment logger (subclass of LocalExperiment)."""

from __future__ import annotations

from typing import Any, Dict

import torch
from clearml import Task

from lisa.utils.experiment_logger import LocalExperiment


class ClearMLExperiment(LocalExperiment):
    """
    Experiment logger that does local logging (via LocalExperiment) and
    also reports scalars and images to ClearML.
    """

    def __init__(self, hyper_params: Dict[str, Any]) -> None:
        run_name = hyper_params['run_name']
        # ClearML Task.init() calls make_deterministic(task.get_random_seed()), which overwrites
        # Python/NumPy/Torch RNG. The task seed defaults to 1337. Set it to our config seed
        # (required key 'seed') so make_deterministic uses the user seed. Run ID is assigned by
        # the ClearML server (new task per run); same config + same seed on two runs yield two
        # different task IDs and directories.
        Task.set_random_seed(hyper_params['seed'])
        self.task = Task.init(
            project_name=hyper_params['logging_project'],
            task_name=run_name,
            auto_connect_frameworks=False,
            auto_resource_monitoring=False,
            reuse_last_task_id=False,
        )
        connected_hyper_params = {
            key: (str(value) if isinstance(value, torch.device) else value)
            for key, value in hyper_params.items()
        }
        self.task.connect(connected_hyper_params)
        self.logger = self.task.get_logger()
        self.upload_metrics_csv = hyper_params['upload_metrics_csv']
        self.log_images_to_server = hyper_params['log_images_to_server']

        super().__init__(hyper_params, run_id=self.task.id)

        if ('png' not in self.image_formats) and self.log_images_to_server:
            print(
                'png was not selected as an image format, but it is required for server logging. '
                'Adding it automatically.'
            )
            self.image_formats = sorted(set(self.image_formats) | {'png'})

    def log(self, data: Dict[str, Any], step: int | None = None) -> None:
        super().log(data, step)
        iteration = 0 if step is None else int(step)
        for name, value in data.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            self.logger.report_scalar(
                title=name,
                series='value',
                value=v,
                iteration=iteration,
            )

    def log_figure(
        self,
        name: str,
        fig: Any,
        step: int | None = None,
        series: str = 'image',
    ) -> None:
        super().log_figure(name, fig, step, series)
        if not self.log_images_to_server:
            return
        iteration = 0 if step is None else int(step)
        if 'png' in self.image_formats:
            png_path = self.dir / series / 'png' / f'{name}_{iteration}.png'
            self.logger.report_image(
                title=name,
                series=series,
                iteration=iteration,
                local_path=str(png_path),
            )

    def finish(self) -> None:
        super().finish()
        try:
            if self.upload_metrics_csv:
                csv_path = self.dir / 'metrics.csv'
                self.task.upload_artifact(
                    name='metrics_csv',
                    artifact_object=str(csv_path),
                    wait_on_upload=False,
                    retries=1,
                )
        finally:
            self.task.close()
