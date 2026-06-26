"""Shared MLflow helpers for Kedro nodes."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import mlflow
from mlflow.entities import Run


@contextmanager
def active_or_new_mlflow_run(run_name: str) -> Iterator[Run]:
    """Reuse Kedro's active MLflow run, or create one for standalone node calls."""
    active_run = mlflow.active_run()
    if active_run is not None:
        yield active_run
        return

    with mlflow.start_run(run_name=run_name) as run:
        yield run


def prefix_keys(values: dict, prefix: str) -> dict:
    """Return a copy of ``values`` with MLflow-safe stage-prefixed keys."""
    return {f"{prefix}.{key}": value for key, value in values.items()}
