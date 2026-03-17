"""
Leakage-safe feature engineering for S&P 500 daily data.

Every feature is computed from historical data only. The final .shift(1) on each
feature ensures that row t's features use only data available before day t's close,
preventing look-ahead bias.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual feature groups
# ---------------------------------------------------------------------------

def _add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log returns over multiple horizons."""
    close = df["Close"]
    for window in [1, 5, 10, 21]:
        df[f"return_{window}d"] = np.log(close / close.shift(window))
    return df


def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD line minus signal line (12/26/9 standard)."""
    close = df["Close"]
    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, min_periods=9).mean()
    df["macd_signal_delta"] = macd_line - signal_line
    return df


def _add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling standard deviation of 1-day returns."""
    ret = np.log(df["Close"] / df["Close"].shift(1))
    for window in [10, 21]:
        df[f"volatility_{window}d"] = ret.rolling(window).std()
    return df


def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """SMA ratio and distance from 52-week high."""
    close = df["Close"]
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    df["sma_50_200_ratio"] = sma50 / sma200

    high_252 = close.rolling(252).max()
    df["pct_off_52w_high"] = close / high_252
    return df


def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume ratio: 10-day avg volume / 21-day avg volume."""
    vol = df["Volume"].astype(float)
    df["volume_ratio_10_21"] = vol.rolling(10).mean() / vol.rolling(21).mean()
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_21d",
    "rsi_14", "macd_signal_delta",
    "volatility_10d", "volatility_21d",
    "sma_50_200_ratio", "pct_off_52w_high",
    "volume_ratio_10_21",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features, shift them forward by 1 day to prevent leakage,
    then drop rows with NaN (warm-up period).

    Returns a copy — the original DataFrame is not modified.
    """
    df = df.copy()

    df = _add_return_features(df)
    df = _add_rsi(df)
    df = _add_macd(df)
    df = _add_volatility_features(df)
    df = _add_trend_features(df)
    df = _add_volume_features(df)

    # Shift features forward so row t uses only data known before day t's close
    for col in FEATURE_COLS:
        df[col] = df[col].shift(1)

    df = df.dropna(subset=FEATURE_COLS + ["Target"])
    return df
