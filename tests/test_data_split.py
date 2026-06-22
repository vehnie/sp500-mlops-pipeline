from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    MARKET_FEATURE_COLUMNS,
    TARGET_COLUMN,
    create_market_feature_dataset,
)
from sp500_mlops_pipeline.pipelines.data_split.nodes import (
    create_time_based_train_val_test_split,
)


RAW_SAMPLE_PATH = PROJECT_ROOT / "data/01_raw/sp500_yahoo_finance_raw.csv"
EXPECTED_SPLIT_KEYS = {
    "X_train",
    "y_train",
    "dates_train",
    "X_val",
    "y_val",
    "dates_val",
    "X_test",
    "y_test",
    "dates_test",
}


@pytest.fixture()
def feature_data():
    raw_data = pd.read_csv(RAW_SAMPLE_PATH)
    return create_market_feature_dataset(raw_data)


@pytest.fixture()
def split_data(feature_data):
    return create_time_based_train_val_test_split(feature_data)


def test_split_output_contains_expected_keys(split_data) -> None:
    assert set(split_data.keys()) == EXPECTED_SPLIT_KEYS


def test_train_validation_and_test_splits_are_non_empty(split_data) -> None:
    assert not split_data["X_train"].empty
    assert not split_data["X_val"].empty
    assert not split_data["X_test"].empty
    assert not split_data["y_train"].empty
    assert not split_data["y_val"].empty
    assert not split_data["y_test"].empty


def test_split_sizes_follow_integer_slicing(feature_data, split_data) -> None:
    total_rows = len(feature_data)
    expected_train_rows = int(total_rows * 0.70)
    expected_val_rows = int(total_rows * 0.15)
    expected_test_rows = total_rows - expected_train_rows - expected_val_rows

    assert len(split_data["X_train"]) == expected_train_rows
    assert len(split_data["X_val"]) == expected_val_rows
    assert len(split_data["X_test"]) == expected_test_rows


def test_dates_are_sorted_within_each_split(split_data) -> None:
    assert split_data["dates_train"].is_monotonic_increasing
    assert split_data["dates_val"].is_monotonic_increasing
    assert split_data["dates_test"].is_monotonic_increasing


def test_train_dates_end_before_validation_dates_start(split_data) -> None:
    assert split_data["dates_train"].max() < split_data["dates_val"].min()


def test_validation_dates_end_before_test_dates_start(split_data) -> None:
    assert split_data["dates_val"].max() < split_data["dates_test"].min()


def test_x_contains_only_market_feature_columns(split_data) -> None:
    assert list(split_data["X_train"].columns) == MARKET_FEATURE_COLUMNS
    assert list(split_data["X_val"].columns) == MARKET_FEATURE_COLUMNS
    assert list(split_data["X_test"].columns) == MARKET_FEATURE_COLUMNS


def test_y_equals_target_values(feature_data, split_data) -> None:
    sorted_features = feature_data.sort_values("Date").reset_index(drop=True)
    train_end = int(len(sorted_features) * 0.70)
    val_end = train_end + int(len(sorted_features) * 0.15)

    assert split_data["y_train"].equals(
        sorted_features[TARGET_COLUMN].iloc[:train_end].reset_index(drop=True)
    )
    assert split_data["y_val"].equals(
        sorted_features[TARGET_COLUMN].iloc[train_end:val_end].reset_index(drop=True)
    )
    assert split_data["y_test"].equals(
        sorted_features[TARGET_COLUMN].iloc[val_end:].reset_index(drop=True)
    )


def test_real_raw_csv_can_generate_features_and_split_successfully() -> None:
    raw_data = pd.read_csv(RAW_SAMPLE_PATH)
    feature_data = create_market_feature_dataset(raw_data)
    split_data = create_time_based_train_val_test_split(feature_data)

    assert set(split_data.keys()) == EXPECTED_SPLIT_KEYS
    assert (
        len(split_data["X_train"])
        + len(split_data["X_val"])
        + len(split_data["X_test"])
        == len(feature_data)
    )


def test_invalid_split_fractions_raise_value_error(feature_data) -> None:
    with pytest.raises(ValueError, match="fractions must sum to 1"):
        create_time_based_train_val_test_split(
            feature_data,
            train_fraction=0.70,
            val_fraction=0.20,
            test_fraction=0.20,
        )


def test_missing_required_columns_raise_value_error(feature_data) -> None:
    incomplete_data = feature_data.drop(columns=[MARKET_FEATURE_COLUMNS[0]])

    with pytest.raises(ValueError, match="Missing required columns"):
        create_time_based_train_val_test_split(incomplete_data)
