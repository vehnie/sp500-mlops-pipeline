from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_ingestion.nodes import (
    EXPECTED_MARKET_COLUMNS,
    download_sp500_market_data,
    load_market_data,
    normalize_yahoo_finance_data,
)


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


def make_yahoo_response(index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    dates = index if index is not None else pd.to_datetime(["2026-01-05", "2026-01-02"])
    return pd.DataFrame(
        {
            "Open": [101.0, 100.0],
            "High": [103.0, 102.0],
            "Low": [100.5, 99.0],
            "Close": [102.5, 101.5],
            "Adj Close": [102.5, 101.5],
            "Volume": [1_100_000, 1_000_000],
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_download_sp500_market_data_uses_yfinance_and_normalizes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    yahoo_response = make_yahoo_response()
    configured_cache_locations = []
    cache_dir = tmp_path / "yfinance_cache"

    def fake_download(*args, **kwargs):
        assert args[0] == "^GSPC"
        assert kwargs["start"] == "2001-01-01"
        assert kwargs["auto_adjust"] is False
        return yahoo_response

    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.data_ingestion.nodes.yf.download",
        fake_download,
    )
    monkeypatch.setattr(
        "sp500_mlops_pipeline.pipelines.data_ingestion.nodes.yf.set_tz_cache_location",
        configured_cache_locations.append,
    )

    result = download_sp500_market_data(yfinance_cache_dir=cache_dir)

    assert configured_cache_locations == [str(cache_dir)]
    assert result.columns.tolist() == EXPECTED_MARKET_COLUMNS
    assert result["Date"].is_monotonic_increasing
    assert result["Date"].is_unique


def test_normalize_yahoo_finance_data_accepts_flat_columns() -> None:
    result = normalize_yahoo_finance_data(make_yahoo_response())

    assert result.columns.tolist() == EXPECTED_MARKET_COLUMNS
    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-02",
        "2026-01-05",
    ]


def test_normalize_yahoo_finance_data_accepts_price_ticker_multiindex() -> None:
    flat_data = make_yahoo_response()
    multiindex_data = flat_data.copy()
    multiindex_data.columns = pd.MultiIndex.from_product(
        [multiindex_data.columns, ["^GSPC"]]
    )

    result = normalize_yahoo_finance_data(multiindex_data)

    assert result.columns.tolist() == EXPECTED_MARKET_COLUMNS
    assert result["Date"].is_monotonic_increasing


def test_normalize_yahoo_finance_data_accepts_ticker_price_multiindex() -> None:
    flat_data = make_yahoo_response()
    multiindex_data = flat_data.copy()
    multiindex_data.columns = pd.MultiIndex.from_product(
        [["^GSPC"], multiindex_data.columns]
    )

    result = normalize_yahoo_finance_data(multiindex_data)

    assert result.columns.tolist() == EXPECTED_MARKET_COLUMNS
    assert result["Date"].is_monotonic_increasing


def test_normalize_yahoo_finance_data_raises_for_empty_download() -> None:
    with pytest.raises(ValueError, match="returned no rows"):
        normalize_yahoo_finance_data(pd.DataFrame())


def test_normalize_yahoo_finance_data_raises_for_missing_market_columns() -> None:
    malformed_data = make_yahoo_response().drop(columns=["Adj Close"])

    with pytest.raises(ValueError, match="missing required market columns"):
        normalize_yahoo_finance_data(malformed_data)


def test_normalize_yahoo_finance_data_raises_for_duplicate_dates() -> None:
    duplicate_index = pd.to_datetime(["2026-01-02", "2026-01-02"])
    duplicate_data = make_yahoo_response(index=duplicate_index)

    with pytest.raises(ValueError, match="duplicate dates"):
        normalize_yahoo_finance_data(duplicate_data)
