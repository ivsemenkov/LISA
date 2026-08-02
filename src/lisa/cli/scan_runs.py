"""Scan experiment directories and build a CSV registry."""

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

from lisa.utils.constants import OUTPUTS_DIR


def _iso_ts(ts: float) -> str:
    """Convert UNIX timestamp to ISO-8601 UTC string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def scan_experiments(experiments_root: Path) -> list[dict]:
    """Scan experiment directories and collect metadata rows.

    Run dirs are directories that contain config.json (layout: experiments_root/run_group/run_name_run_id/).
    For each run dir, run_group is the parent dir name; run_name and n_channels_unmix come from config.
    """
    rows = []
    if not experiments_root.exists():
        return rows

    config_paths = sorted(experiments_root.glob('*/*/config.json'))
    for config_path in tqdm(config_paths, desc='Scanning experiments'):
        entry = config_path.parent
        metrics_path = entry / 'metrics.csv'
        run_group = entry.parent.name if entry.parent != experiments_root else ''
        run_id = ''
        match = re.match(r'(.+)_([^_]+)$', entry.name)
        if match:
            run_id = match.group(2)

        run_name = ''
        n_channels_unmix = ''
        try:
            config = json.loads(config_path.read_text(encoding='utf-8'))
            run_group = config.get('run_group', run_group)
            run_name = config.get('run_name', run_name)
            if 'n_channels_unmix' in config:
                n_channels_unmix = str(config['n_channels_unmix'])
        except (json.JSONDecodeError, OSError):
            pass

        rows.append(
            {
                'run_group': run_group or '',
                'run_name': run_name or '',
                'n_channels_unmix': n_channels_unmix,
                'run_id': run_id or '',
                'experiment_dir': str(entry),
                'config_path': str(config_path),
                'metrics_path': str(metrics_path) if metrics_path.exists() else '',
                'created_utc': _iso_ts(entry.stat().st_ctime),
                'modified_utc': _iso_ts(entry.stat().st_mtime),
            }
        )

    return rows


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for scan-runs."""
    parser = argparse.ArgumentParser(
        description='Scan experiment directories and build a registry CSV.'
    )
    parser.add_argument(
        '--experiments-root',
        type=str,
        default=os.path.join(OUTPUTS_DIR, 'experiments'),
        help='Root directory with experiment logs (default: PROJECT_ROOT/outputs/experiments).',
    )
    parser.add_argument(
        '--out',
        type=str,
        default=os.path.join(OUTPUTS_DIR, 'experiment_registry.csv'),
        help='Output CSV path.',
    )
    return parser.parse_args()


def main_cli() -> None:
    """CLI entry point for scan-runs."""
    args = parse_arguments()
    experiments_root = Path(args.experiments_root)
    rows = scan_experiments(experiments_root)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        'run_group',
        'run_name',
        'n_channels_unmix',
        'run_id',
        'experiment_dir',
        'config_path',
        'metrics_path',
        'created_utc',
        'modified_utc',
    ]
    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, '') for h in header})

    print(f'Wrote {len(rows)} rows to {out_path}')


if __name__ == '__main__':
    main_cli()
