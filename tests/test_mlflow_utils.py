from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.mlflow_utils import active_or_new_mlflow_run


def test_active_or_new_mlflow_run_reuses_active_run(monkeypatch) -> None:
    active_run = SimpleNamespace(info=SimpleNamespace(run_id="active-run-id"))
    start_run = MagicMock()

    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.mlflow_utils.mlflow.active_run",
        lambda: active_run,
    )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.mlflow_utils.mlflow.start_run",
        start_run,
    )

    with active_or_new_mlflow_run(run_name="node-run") as run:
        assert run is active_run

    start_run.assert_not_called()


def test_active_or_new_mlflow_run_starts_run_without_active_run(monkeypatch) -> None:
    created_run = SimpleNamespace(info=SimpleNamespace(run_id="created-run-id"))
    run_context = MagicMock()
    run_context.__enter__.return_value = created_run
    run_context.__exit__.return_value = False
    start_run = MagicMock(return_value=run_context)

    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.mlflow_utils.mlflow.active_run",
        lambda: None,
    )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.mlflow_utils.mlflow.start_run",
        start_run,
    )

    with active_or_new_mlflow_run(run_name="node-run") as run:
        assert run is created_run

    start_run.assert_called_once_with(run_name="node-run")
