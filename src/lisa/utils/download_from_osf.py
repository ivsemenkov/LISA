#!/usr/bin/env python3
"""
download_from_osf.py

Robust OSF downloader using osfclient Python API (no URL/HTTP code handling in your code).

Invariants:
1) No corrupt final files:
   - download to <dest>.part
   - atomic commit via os.replace(<dest>.part, <dest>) on success

2) Resume-by-rerun:
   - if <dest> exists and provenance says it came from the same (project_id, remote_path), skip it

3) Collision-safe (for flat layouts):
   - if a different source wants to write the same destination path, default is FAIL
   - can be overridden with --overwrite

State:
- provenance TSV file (default): <output_dir>/.osf_provenance.tsv
  Each line:
      <dest_rel>\\t<project_id>\\t<remote_rel>
  Last record wins.

Token:
- If --token is provided, use it.
- Else, try OSF_TOKEN from environment.
- Else, run without a token (works for public projects).
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

from osfclient import OSF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Robust OSF downloader (osfclient API): atomic .part->final, resumable, collision-safe.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--project-ids',
        nargs='+',
        required=True,
        help='One or more OSF project IDs (GUIDs), e.g. ag3kj hqvm3',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Root directory where files will be saved.',
    )
    parser.add_argument(
        '--provider',
        default='osfstorage',
        help='OSF storage provider (usually osfstorage).',
    )

    # Avoid using the word "prefix": these are just remote/local subdir controls.
    parser.add_argument(
        '--remote-subdir',
        default='',
        help="Download only files under this remote folder (path inside storage), e.g. 'clean/'. Empty = download all.",
    )
    parser.add_argument(
        '--drop-remote-subdir',
        default='',
        help="If the remote path starts with this folder, drop it from the local path, e.g. 'clean/'.",
    )

    parser.add_argument(
        '--project-subdir',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If enabled (default), save under <output>/<project_id>/... (safer). Disable with --no-project-subdir (flat layout).',
    )
    parser.add_argument(
        '--overwrite',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='If enabled, overwrite files on collision (not recommended).',
    )
    parser.add_argument(
        '--adopt-existing',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'If enabled, when a destination file exists but has no provenance entry, record it as owned and skip. '
            'Fast, but unsafe if that existing file is corrupt/partial from a legacy manual download.'
        ),
    )

    parser.add_argument(
        '--token',
        default='',
        help='OSF token. If not set, OSF_TOKEN env var is used. If neither is set, runs without a token (public projects).',
    )
    parser.add_argument(
        '--provenance-file',
        default='',
        help='Path to provenance TSV. If empty, uses <output_dir>/.osf_provenance.tsv',
    )

    parser.add_argument(
        '--retries',
        type=int,
        default=8,
        help='Number of attempts per file on failures.',
    )
    parser.add_argument(
        '--backoff-base',
        type=float,
        default=0.5,
        help='Backoff base in seconds (exponential).',
    )
    parser.add_argument(
        '--backoff-cap',
        type=float,
        default=20.0,
        help='Maximum backoff sleep in seconds.',
    )
    parser.add_argument(
        '--quiet',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Reduce output.',
    )

    return parser.parse_args()


def log(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg, flush=True)


def backoff_sleep(attempt: int, base: float, cap: float) -> None:
    delay = min(cap, base * (2**attempt))
    delay *= 1.0 + random.uniform(0.0, 0.25)
    time.sleep(delay)


def safe_rel(remote_path: str) -> str:
    rel = remote_path.lstrip('/')
    parts = [p for p in rel.split('/') if p]
    if any(p == '..' for p in parts):
        raise ValueError(f"Refusing path with '..': {remote_path}")
    return '/'.join(parts)


def load_prov(path: Path) -> Dict[str, Tuple[str, str]]:
    prov: Dict[str, Tuple[str, str]] = {}
    if not path.exists():
        return prov
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            dest_rel, pid, remote_rel = parts
            prov[dest_rel] = (pid, remote_rel)
    return prov


def append_prov(path: Path, dest_rel: str, pid: str, remote_rel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(f'{dest_rel}\t{pid}\t{remote_rel}\n')


def ensure_parent(dest: Path, created_dirs: set[Path]) -> None:
    parent = dest.parent
    if parent not in created_dirs:
        parent.mkdir(parents=True, exist_ok=True)
        created_dirs.add(parent)


def download_atomic(
    osf_file, dest: Path, retries: int, base: float, cap: float
) -> None:
    """
    Download one OSF file atomically:
    - write to <dest>.part
    - on success: os.replace(part, dest)

    Uses osfclient's file.write_to(fp) (no custom URL/HTTP handling).
    """
    part = dest.with_name(dest.name + '.part')

    if part.exists():
        try:
            part.unlink()
        except OSError:
            pass

    last: Optional[Exception] = None
    for attempt in range(max(1, retries)):
        try:
            with part.open('wb') as fp:
                osf_file.write_to(fp)
            os.replace(part, dest)
            return
        except Exception as e:
            last = e
            if part.exists():
                try:
                    part.unlink()
                except OSError:
                    pass
            if attempt < retries - 1:
                backoff_sleep(attempt, base, cap)

    raise RuntimeError(f'Failed to download {osf_file.path}: {last}')


def main() -> int:
    args = parse_args()

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prov_path = (
        Path(args.provenance_file).expanduser().resolve()
        if args.provenance_file
        else (out_dir / '.osf_provenance.tsv')
    )
    prov = load_prov(prov_path)

    created_dirs: set[Path] = set()

    token = (
        (args.token or '').strip() or os.environ.get('OSF_TOKEN', '').strip() or None
    )
    osf = OSF(token=token)

    failures = 0

    for pid in args.project_ids:
        log(f'\n== {pid} ==', args.quiet)
        storage = osf.project(pid).storage(provider=args.provider)

        for f in storage.files:
            remote_rel = safe_rel(str(f.path))

            if args.remote_subdir and not remote_rel.startswith(args.remote_subdir):
                continue

            local_rel = remote_rel
            if args.drop_remote_subdir and local_rel.startswith(
                args.drop_remote_subdir
            ):
                local_rel = local_rel[len(args.drop_remote_subdir) :]
            if not local_rel:
                continue

            dest_rel = f'{pid}/{local_rel}' if args.project_subdir else local_rel
            dest = out_dir / dest_rel

            owner = prov.get(dest_rel)

            if dest.exists():
                # Resume: exact same source already committed
                if owner == (pid, remote_rel):
                    continue

                # Existing file without provenance
                if owner is None and args.adopt_existing:
                    prov[dest_rel] = (pid, remote_rel)
                    append_prov(prov_path, dest_rel, pid, remote_rel)
                    continue

                # Collision
                if not args.overwrite:
                    log(
                        f'COLLISION {dest_rel} existing_owner={owner} new=({pid},{remote_rel})',
                        args.quiet,
                    )
                    failures += 1
                    continue

                try:
                    dest.unlink()
                except OSError:
                    pass

            ensure_parent(dest, created_dirs)

            try:
                download_atomic(
                    f, dest, args.retries, args.backoff_base, args.backoff_cap
                )
                prov[dest_rel] = (pid, remote_rel)
                append_prov(prov_path, dest_rel, pid, remote_rel)
                log(f'OK {dest_rel}', args.quiet)
            except Exception as e:
                failures += 1
                log(f'FAIL {dest_rel}: {e}', args.quiet)

    return 2 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
