from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_ingestion.nodes import load_market_data


def write_sample_csv(csv_path: Path) -> pd.DataFrame:
    sample_data = pd.DataFrame(
        {
            "Date": ["2026-01-02", "2026-01-05"],
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.5],
            "Close": [101.5, 102.5],
            "Adj Close": [101.5, 102.5],
            "Volume": [1_000_000, 1_100_000],
        }
    )
    sample_data.to_csv(csv_path, index=False)
    return sample_data


def test_load_market_data_returns_dataframe(tmp_path: Path) -> None:
    csv_path = tmp_path / "market_data.csv"
    write_sample_csv(csv_path)

    loaded_data = load_market_data(csv_path)

    assert isinstance(loaded_data, pd.DataFrame)


def test_load_market_data_preserves_rows_and_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "market_data.csv"
    expected_data = write_sample_csv(csv_path)

    loaded_data = load_market_data(csv_path)

    pd.testing.assert_frame_equal(loaded_data, expected_data)


def test_load_market_data_accepts_path_object(tmp_path: Path) -> None:
    csv_path = tmp_path / "market_data.csv"
    write_sample_csv(csv_path)

    loaded_data = load_market_data(csv_path)

    assert len(loaded_data) == 2


def test_load_market_data_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_market_data(missing_path)
