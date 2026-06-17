"""Market feature engineering for S&P 500 OHLCV data."""

from pathlib import Path

import numpy as np
import pandas as pd

from sp500_mlops_pipeline.pipelines.data_cleaning.nodes import clean_raw_market_data


RAW_SAMPLE_PATH = Path("data/01_raw/sp500_yahoo_finance_raw.csv")
TARGET_CATEGORIES = ["bearish", "neutral", "bullish"]
UP_THRESHOLD = 0.015
DOWN_THRESHOLD = -0.015

MARKET_FEATURE_COLUMNS = [
    "simple_return",
    "log_return",
    "sma_10",
    "sma_20",
    "ema_10",
    "ema_20",
    "sma_ratio_10",
    "sma_ratio_20",
    "ema_ratio_10",
    "ema_ratio_20",
    "mom_5",
    "mom_10",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility_10",
    "volatility_20",
    "high_low_range",
    "drawdown_10",
    "drawdown_20",
    "max_drawdown_20",
    "drawdown_change",
    "drawdown_abs",
    "drawdown_persistence",
    "volatility_20_sq",
    "ret_5",
    "ret_10",
    "log_volume",
    "volume_change",
    "volume_ma_10",
    "volume_ratio_10",
]


def create_market_feature_dataset(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """Create thesis-style market features and the 5-day direction target."""
    features = clean_raw_market_data(cleaned_data)
    features = features.sort_values("Date").reset_index(drop=True)

    features["simple_return"] = features["Close"].pct_change()
    features["log_return"] = np.log(features["Close"] / features["Close"].shift(1))

    features["sma_10"] = features["Close"].rolling(10).mean()
    features["sma_20"] = features["Close"].rolling(20).mean()
    features["ema_10"] = features["Close"].ewm(span=10, adjust=False).mean()
    features["ema_20"] = features["Close"].ewm(span=20, adjust=False).mean()

    features["sma_ratio_10"] = features["Close"] / features["sma_10"] - 1
    features["sma_ratio_20"] = features["Close"] / features["sma_20"] - 1
    features["ema_ratio_10"] = features["Close"] / features["ema_10"] - 1
    features["ema_ratio_20"] = features["Close"] / features["ema_20"] - 1

    features["mom_5"] = features["Close"] / features["Close"].shift(5) - 1
    features["mom_10"] = features["Close"] / features["Close"].shift(10) - 1

    delta = features["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    features["rsi_14"] = 100 - (100 / (1 + rs))

    ema_12 = features["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = features["Close"].ewm(span=26, adjust=False).mean()
    features["macd"] = ema_12 - ema_26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    features["volatility_10"] = features["log_return"].rolling(10).std()
    features["volatility_20"] = features["log_return"].rolling(20).std()
    features["high_low_range"] = (features["High"] - features["Low"]) / features["Close"]

    rolling_max_10 = features["Close"].rolling(10).max()
    rolling_max_20 = features["Close"].rolling(20).max()
    features["drawdown_10"] = (features["Close"] - rolling_max_10) / rolling_max_10
    features["drawdown_20"] = (features["Close"] - rolling_max_20) / rolling_max_20
    features["max_drawdown_20"] = features["drawdown_20"].rolling(20).min()
    features["drawdown_change"] = features["drawdown_20"].diff()
    features["drawdown_abs"] = features["drawdown_20"].abs()
    features["drawdown_persistence"] = (
        (features["drawdown_20"] < -0.02).rolling(10).sum()
    )

    features["volatility_20_sq"] = features["log_return"].pow(2).rolling(20).mean()
    features["ret_5"] = features["log_return"].rolling(5).sum()
    features["ret_10"] = features["log_return"].rolling(10).sum()

    features["log_volume"] = np.log1p(features["Volume"])
    features["volume_change"] = features["Volume"].pct_change()
    features["volume_ma_10"] = features["Volume"].rolling(10).mean()
    features["volume_ratio_10"] = features["Volume"] / features["volume_ma_10"]

    features["future_return_5d"] = (
        features["log_return"].rolling(window=5).sum().shift(-5)
    )
    features["target"] = np.select(
        [
            features["future_return_5d"] > UP_THRESHOLD,
            features["future_return_5d"] < DOWN_THRESHOLD,
        ],
        ["bullish", "bearish"],
        default="neutral",
    )
    features["target"] = pd.Categorical(
        features["target"],
        categories=TARGET_CATEGORIES,
        ordered=True,
    )

    feature_dataset = features.dropna().reset_index(drop=True)

    return feature_dataset


def load_clean_and_create_market_feature_dataset(
    csv_path: str | Path = RAW_SAMPLE_PATH,
) -> pd.DataFrame:
    """Load the raw sample CSV, clean it, and create the market feature dataset."""
    raw_data = pd.read_csv(csv_path)
    cleaned_data = clean_raw_market_data(raw_data)
    return create_market_feature_dataset(cleaned_data)
