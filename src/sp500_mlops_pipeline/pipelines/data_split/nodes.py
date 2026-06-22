"""Time-aware train/validation/test split helpers."""

import math

import pandas as pd

from sp500_mlops_pipeline.pipelines.data_feat_engineering.nodes import (
    MARKET_FEATURE_COLUMNS,
)


TARGET_COLUMN = "target_next_day_up"
DATE_COLUMN = "Date"
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15


def create_time_based_train_val_test_split(
    feature_data: pd.DataFrame,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Split a feature dataset chronologically into train, validation, and test."""
    _validate_split_fractions(train_fraction, val_fraction, test_fraction)
    _validate_required_columns(feature_data)

    data = feature_data.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="raise")
    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)

    train_end = int(len(data) * train_fraction)
    val_end = train_end + int(len(data) * val_fraction)

    train = data.iloc[:train_end].reset_index(drop=True)
    validation = data.iloc[train_end:val_end].reset_index(drop=True)
    test = data.iloc[val_end:].reset_index(drop=True)

    _validate_non_empty_splits(train, validation, test)

    return {
        "X_train": train.loc[:, MARKET_FEATURE_COLUMNS].copy(),
        "y_train": train[TARGET_COLUMN].copy(),
        "dates_train": train[DATE_COLUMN].copy(),
        "X_val": validation.loc[:, MARKET_FEATURE_COLUMNS].copy(),
        "y_val": validation[TARGET_COLUMN].copy(),
        "dates_val": validation[DATE_COLUMN].copy(),
        "X_test": test.loc[:, MARKET_FEATURE_COLUMNS].copy(),
        "y_test": test[TARGET_COLUMN].copy(),
        "dates_test": test[DATE_COLUMN].copy(),
    }


def _validate_required_columns(feature_data: pd.DataFrame) -> None:
    required_columns = [DATE_COLUMN, TARGET_COLUMN, *MARKET_FEATURE_COLUMNS]
    missing_columns = [
        column for column in required_columns if column not in feature_data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _validate_split_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    fractions = [train_fraction, val_fraction, test_fraction]
    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("Split fractions must each be between 0 and 1")

    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test fractions must sum to 1")


def _validate_non_empty_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test splits must be non-empty")
