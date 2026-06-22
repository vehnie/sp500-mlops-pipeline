from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    FEATURE_PRICE_COLUMN,
    MARKET_FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_market_feature_dataset,
)


RAW_SAMPLE_PATH = PROJECT_ROOT / "data/01_raw/sp500_yahoo_finance_raw.csv"


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

    assert set(MARKET_FEATURE_COLUMNS).issubset(features.columns)


def test_target_column_exists() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert TARGET_COLUMN == "target_next_day_up"
    assert TARGET_COLUMN in features.columns


def test_target_is_binary() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert set(features[TARGET_COLUMN].unique()).issubset({0, 1})


def test_feature_dataset_has_no_missing_values() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert not features.isna().any().any()


def test_feature_dataset_contains_only_finite_feature_values() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert np.isfinite(features[MARKET_FEATURE_COLUMNS].to_numpy()).all()


def test_zero_previous_volume_does_not_generate_infinite_volume_change() -> None:
    data = make_clean_ohlcv_data().sort_values("Date").reset_index(drop=True)
    zero_volume_date = data.loc[30, "Date"]
    following_date = data.loc[31, "Date"]
    data.loc[30, "Volume"] = 0

    features = create_market_feature_dataset(data)

    assert zero_volume_date in set(features["Date"])
    assert following_date not in set(features["Date"])
    assert np.isfinite(features["volume_change"]).all()


def test_target_semantics_are_unchanged_when_zero_volume_row_is_removed() -> None:
    data = make_clean_ohlcv_data().sort_values("Date").reset_index(drop=True)
    data.loc[30, "Volume"] = 0
    expected_target_by_date = (
        data["Close"].shift(-1).gt(data["Close"]).astype(int)
    )
    expected_target_by_date.index = data["Date"]

    features = create_market_feature_dataset(data)

    expected_target = features["Date"].map(expected_target_by_date)
    pd.testing.assert_series_equal(
        features[TARGET_COLUMN],
        expected_target.rename(TARGET_COLUMN),
        check_index=False,
    )


def test_feature_rows_are_sorted_by_date() -> None:
    features = create_market_feature_dataset(make_clean_ohlcv_data())

    assert features["Date"].is_monotonic_increasing


def test_feature_engineering_uses_close_as_official_price_source() -> None:
    original_data = make_clean_ohlcv_data()
    changed_adjusted_close = original_data.copy()
    changed_adjusted_close["Adj Close"] = (
        changed_adjusted_close["Adj Close"] * 3.0 + 500.0
    )

    original_features = create_market_feature_dataset(original_data)
    changed_features = create_market_feature_dataset(changed_adjusted_close)

    assert FEATURE_PRICE_COLUMN == "Close"
    pd.testing.assert_frame_equal(
        original_features[MARKET_FEATURE_COLUMNS],
        changed_features[MARKET_FEATURE_COLUMNS],
    )
    pd.testing.assert_series_equal(
        original_features[TARGET_COLUMN],
        changed_features[TARGET_COLUMN],
    )


def test_real_raw_csv_can_generate_valid_feature_dataset() -> None:
    raw_data = pd.read_csv(RAW_SAMPLE_PATH)
    features = create_market_feature_dataset(raw_data)

    assert not features.empty
    assert set(MARKET_FEATURE_COLUMNS).issubset(features.columns)
    assert TARGET_COLUMN in features.columns
    assert features["Date"].is_monotonic_increasing
    assert not features.isna().any().any()
    assert np.isfinite(features[MARKET_FEATURE_COLUMNS].to_numpy()).all()
    assert set(features[TARGET_COLUMN].unique()).issubset({0, 1})
