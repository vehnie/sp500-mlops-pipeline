"""Market feature engineering for S&P 500 OHLCV data."""

import numpy as np
import pandas as pd

from sp500_mlops_pipeline.pipelines.data_cleaning.nodes import clean_raw_market_data


FEATURE_PRICE_COLUMN = "Close"
MARKET_FEATURE_COLUMNS = [
    "simple_return",
    "log_return",
    "sma_10",
    "sma_20",
    "sma_ratio_10",
    "rsi_14",
    "volatility_10",
    "volume_change",
]
TARGET_COLUMN = "target_next_day_up"


def create_market_feature_dataset(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """Create market features and the next-day direction target."""
    features = clean_raw_market_data(cleaned_data)
    features = features.sort_values("Date").reset_index(drop=True)
    price = features[FEATURE_PRICE_COLUMN]

    features["simple_return"] = price.pct_change()
    features["log_return"] = np.log(price / price.shift(1))
    features["sma_10"] = price.rolling(window=10).mean()
    features["sma_20"] = price.rolling(window=20).mean()
    features["sma_ratio_10"] = price / features["sma_10"] - 1

    price_change = price.diff()
    average_gain = price_change.clip(lower=0).rolling(window=14).mean()
    average_loss = -price_change.clip(upper=0).rolling(window=14).mean()
    relative_strength = average_gain / average_loss
    features["rsi_14"] = 100 - (100 / (1 + relative_strength))

    features["volatility_10"] = features["log_return"].rolling(window=10).std()
    previous_volume = features["Volume"].shift(1)
    features["volume_change"] = np.where(
        previous_volume > 0,
        features["Volume"] / previous_volume - 1,
        np.nan,
    )

    next_day_close = price.shift(-1)
    features[TARGET_COLUMN] = (next_day_close > price).astype("Int64")
    features.loc[next_day_close.isna(), TARGET_COLUMN] = pd.NA

    features[MARKET_FEATURE_COLUMNS] = features[MARKET_FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    feature_dataset = features.dropna(
        subset=[*MARKET_FEATURE_COLUMNS, TARGET_COLUMN]
    ).reset_index(drop=True)
    feature_dataset[TARGET_COLUMN] = feature_dataset[TARGET_COLUMN].astype(int)

    if not feature_dataset[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 or 1")

    return feature_dataset
