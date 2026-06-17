from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    MARKET_FEATURE_COLUMNS,
    TARGET_CATEGORIES,
    create_market_feature_dataset,
    load_clean_and_create_market_feature_dataset,
)


RAW_SAMPLE_PATH = Path("data/01_raw/sp500_yahoo_finance_raw.csv")


def make_clean_ohlcv_data(rows: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=rows)
    close = pd.Series(range(100, 100 + rows), dtype=float)

    return pd.DataFrame(
        {
            "Date": dates[::-1],
            "Open": close[::-1] + 0.1,
            "High": close[::-1] + 1.0,
            "Low": close[::-1] - 1.0,
            "Close": close[::-1],
            "Adj Close": close[::-1],
            "Volume": range(1_000_000 + rows, 1_000_000, -1),
        }
    )


def test_feature_dataframe_is_not_empty() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert not features.empty


def test_expected_feature_columns_exist() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    for column in MARKET_FEATURE_COLUMNS:
        assert column in features.columns


def test_target_column_exists() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert "target" in features.columns


def test_target_only_contains_expected_classes() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert set(features["target"].dropna().astype(str)).issubset(TARGET_CATEGORIES)


def test_target_is_ordered_categorical() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert isinstance(features["target"].dtype, pd.CategoricalDtype)
    assert features["target"].cat.ordered
    assert list(features["target"].cat.categories) == TARGET_CATEGORIES


def test_future_return_5d_exists_and_has_no_missing_values() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert "future_return_5d" in features.columns
    assert not features["future_return_5d"].isna().any()


def test_feature_rows_are_sorted_by_date() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert features["Date"].is_monotonic_increasing


def test_real_raw_csv_can_generate_valid_feature_dataset() -> None:
    features = load_clean_and_create_market_feature_dataset(RAW_SAMPLE_PATH)

    assert not features.empty
    assert set(TARGET_CATEGORIES).issuperset(features["target"].astype(str).unique())
    assert features["Date"].is_monotonic_increasing
    assert not features["future_return_5d"].isna().any()
