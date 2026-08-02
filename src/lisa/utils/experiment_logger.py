"""Experiment logging abstraction: protocol and local-only backend."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ExperimentProtocol(Protocol):
    """Protocol for experiment loggers (local-only or ClearML)."""

    dir: Path

    def log(self, data: Dict[str, Any], step: int | None = None) -> None:
        """Log scalar metrics."""
        ...

    def log_figure(
        self,
        name: str,
        fig: Any,
        step: int | None = None,
        series: str = 'image',
    ) -> None:
        """Log a matplotlib figure."""
        ...

    def finish(self) -> None:
        """Finalize the experiment (write metrics, close remote if any)."""
        ...


class LocalExperiment:
    """
    Local experiment logger (base for all backends).

    Creates a local experiment directory and saves config, CLI, scalar metrics,
    and figures to disk. Subclasses (e.g. ClearMLExperiment) add remote logging.
    """

    def __init__(
        self,
        hyper_params: Dict[str, Any],
        run_id: str | None = None,
    ) -> None:
        """Initialize local experiment directory.

        Args:
            hyper_params: Dict of hyperparameters / config.
            run_id: Unique run id for the experiment dir. If None, a short uuid
                is generated (local-only). Subclasses pass their own id (e.g.
                ClearML task.id).
        """
        self.hyper_params = hyper_params
        self.image_formats = sorted(
            set(fmt.lower() for fmt in hyper_params['image_formats'])
        )
        self.id = run_id if run_id is not None else uuid.uuid4().hex[:8]

        run_name = hyper_params['run_name']
        root = Path(hyper_params['experiments_root'])
        run_group = str(hyper_params['run_group'])
        self.dir = root / run_group / f'{run_name}_{self.id}'
        self.dir.mkdir(parents=True, exist_ok=True)

        (self.dir / 'config.json').write_text(
            json.dumps(hyper_params, indent=2, default=str),
            encoding='utf-8',
        )
        (self.dir / 'cli.txt').write_text(
            ' '.join(sys.argv),
            encoding='utf-8',
        )

        self.metrics: Dict[int, Dict[str, float]] = {}

    def log(self, data: Dict[str, Any], step: int | None = None) -> None:
        """Log scalar metrics (stored in memory, written to metrics.csv in finish())."""
        iteration = 0 if step is None else int(step)

        for name, value in data.items():
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue

            if iteration not in self.metrics:
                self.metrics[iteration] = {}
            assert name not in self.metrics[iteration], (
                f"Metric '{name}' already logged for iteration {iteration}"
            )
            self.metrics[iteration][name] = v

    def log_figure(
        self,
        name: str,
        fig: Any,
        step: int | None = None,
        series: str = 'image',
    ) -> None:
        """Save a matplotlib figure to the experiment directory."""
        iteration = 0 if step is None else int(step)

        for fmt in self.image_formats:
            save_dir = self.dir / series / fmt
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'{name}_{iteration}.{fmt}'
            fig.savefig(save_path, bbox_inches='tight')

    def finish(self) -> None:
        """Write metrics.csv to the experiment directory."""
        all_names: set[str] = set()
        for metrics_at_step in self.metrics.values():
            all_names.update(metrics_at_step.keys())

        df: Dict[str, list] = {'epoch': []}
        for name in sorted(all_names):
            df[name] = []

        for step in sorted(self.metrics.keys()):
            df['epoch'].append(step)
            for name in all_names:
                df[name].append(self.metrics[step].get(name, float('nan')))

        pd.DataFrame(df).to_csv(self.dir / 'metrics.csv', index=False)


def create_experiment(hyper_params: Dict[str, Any]) -> ExperimentProtocol:
    """Create an experiment logger based on hyper_params['logger'].

    Returns:
        LocalExperiment when logger='local'; ClearMLExperiment when logger='clearml'
        (requires clearml to be installed).

    Raises:
        ValueError: If logger is not 'local' or 'clearml'.
        ImportError: If logger='clearml' but clearml is not installed.
    """
    logger = hyper_params['logger']
    if logger == 'local':
        return LocalExperiment(hyper_params)
    if logger == 'clearml':
        try:
            from lisa.utils.clearml_logger import ClearMLExperiment
        except ImportError as e:
            raise ImportError(
                'ClearML logger requested but clearml is not installed. '
                'Install with: pip install lisa[clearml]'
            ) from e
        return ClearMLExperiment(hyper_params)
    raise ValueError(f"Unknown logger: {logger!r}. Use 'local' or 'clearml'.")
