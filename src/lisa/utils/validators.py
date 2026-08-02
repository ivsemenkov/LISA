"""Shared validation helpers."""

import re

# Run group must be safe as a single path segment: no path separators, no .., alphanumeric start.
_RUN_GROUP_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')


def validate_run_group(run_group: str) -> None:
    """Raise ValueError if run_group is not safe for use as a path segment.

    Use this at train time (parse_experiment_args) and plot time so we never
    silently mangle or write surprising paths.
    """
    if not run_group or not run_group.strip():
        raise ValueError('run-group must be non-empty.')
    if '/' in run_group or '\\' in run_group or '..' in run_group:
        raise ValueError(
            f"run-group must not contain path separators or '..' (got {run_group!r}). "
            'Use only letters, digits, dots, underscores, or hyphens.'
        )
    if not _RUN_GROUP_RE.match(run_group):
        raise ValueError(
            f'run-group must start with a letter or digit and contain only '
            f'[a-zA-Z0-9._-] (got {run_group!r}).'
        )
