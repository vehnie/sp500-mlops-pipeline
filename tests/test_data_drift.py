from pathlib import Path
import json
import sys

import pandas as pd
import pytest
from kedro_datasets.json import JSONDataset
from kedro_datasets.text import TextDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_drift.nodes import (
    DRIFT_FEATURE_COLUMNS,
    create_data_drift_report,
    select_numeric_drift_features,
    split_reference_current_data,
)


def make_feature_data(rows: int = 100) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=rows)
    data = pd.DataFrame(
        {
            "Date": dates,
            "simple_return": [index / 10_000 for index in range(rows)],
            "log_return": [index / 11_000 for index in range(rows)],
            "sma_10": [100 + index / 10 for index in range(rows)],
            "sma_20": [99 + index / 12 for index in range(rows)],
            "sma_ratio_10": [index / 20_000 for index in range(rows)],
            "rsi_14": [30 + index % 40 for index in range(rows)],
            "volatility_10": [0.01 + index / 100_000 for index in range(rows)],
            "volume_change": [(index % 10 - 5) / 100 for index in range(rows)],
            "target_next_day_up": [index % 2 for index in range(rows)],
            "Close": [4_000 + index for index in range(rows)],
        }
    )
    return data.sample(frac=1.0, random_state=42).reset_index(drop=True)


@pytest.fixture()
def drift_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_data, current_data = split_reference_current_data(make_feature_data())
    return select_numeric_drift_features(reference_data, current_data)


def test_chronological_reference_current_split_uses_oldest_70_percent() -> None:
    data = make_feature_data(rows=10)

    reference_data, current_data = split_reference_current_data(data)

    assert len(reference_data) == 7
    assert len(current_data) == 3
    assert reference_data["Date"].is_monotonic_increasing
    assert current_data["Date"].is_monotonic_increasing
    assert reference_data["Date"].max() < current_data["Date"].min()
    assert reference_data["Date"].tolist() == sorted(data["Date"].tolist())[:7]
    assert current_data["Date"].tolist() == sorted(data["Date"].tolist())[7:]


def test_numeric_feature_selection_uses_only_required_columns() -> None:
    reference_data, current_data = split_reference_current_data(make_feature_data())

    reference_features, current_features = select_numeric_drift_features(
        reference_data,
        current_data,
    )

    assert list(reference_features.columns) == DRIFT_FEATURE_COLUMNS
    assert list(current_features.columns) == DRIFT_FEATURE_COLUMNS
    assert all(
        pd.api.types.is_numeric_dtype(reference_features[column])
        for column in DRIFT_FEATURE_COLUMNS
    )
    assert all(
        pd.api.types.is_numeric_dtype(current_features[column])
        for column in DRIFT_FEATURE_COLUMNS
    )


def test_drift_report_files_are_created(
    tmp_path: Path,
    drift_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    html_report, summary = create_data_drift_report(*drift_inputs)
    html_path = tmp_path / "data_drift_report.html"
    summary_path = tmp_path / "data_drift_summary.json"

    TextDataset(
        filepath=str(html_path),
        fs_args={"open_args_save": {"encoding": "utf-8"}},
    ).save(html_report)
    JSONDataset(filepath=str(summary_path)).save(summary)

    assert html_path.is_file()
    assert summary_path.is_file()
    saved_html = html_path.read_text(encoding="utf-8").lower()
    assert "<!doctype html>" in saved_html
    assert "<script" in saved_html
    assert len(saved_html) > 10_000


def test_summary_json_contains_drift_information(
    tmp_path: Path,
    drift_inputs: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, summary = create_data_drift_report(*drift_inputs)
    summary_path = tmp_path / "data_drift_summary.json"
    JSONDataset(filepath=str(summary_path)).save(summary)

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert saved_summary["features"] == DRIFT_FEATURE_COLUMNS
    assert saved_summary["number_of_features"] == len(DRIFT_FEATURE_COLUMNS)
    assert isinstance(saved_summary["dataset_drift"], bool)
    assert isinstance(saved_summary["drifted_columns_count"], int)
    assert isinstance(saved_summary["drifted_columns_share"], float)
    assert set(saved_summary["column_drift"]) == set(DRIFT_FEATURE_COLUMNS)
    assert all(
        "drift_detected" in feature_summary
        and "drift_score" in feature_summary
        for feature_summary in saved_summary["column_drift"].values()
    )
