"""
Download and clean S&P 500 historical data from Yahoo Finance.

Includes a CSV cache so the data only needs to be downloaded once.
Delete data/sp500_raw.csv to force a fresh download.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


CACHE_DIR = Path("data")
CACHE_FILE = CACHE_DIR / "sp500_raw.csv"


def download_sp500(start: str = "2000-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Download S&P 500 OHLCV data from Yahoo Finance (cached to CSV)."""
    if CACHE_FILE.exists():
        print(f"       Loading cached data from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return df

    print("       Downloading from Yahoo Finance (first run only)...")
    ticker = yf.Ticker("^GSPC")
    df = ticker.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance. Check your network or ticker symbol.")

    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    df = df.sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE)
    print(f"       Cached to {CACHE_FILE}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill small gaps, drop remaining NaNs, and add the target column."""
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df = df[required].copy()
    df = df.ffill(limit=2)
    df = df.dropna()
    df = df[df["Volume"] > 0]

    # Target: 1 if tomorrow's close > today's close, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop the last row (no future close to compare against)
    df = df.iloc[:-1]

    return df


def load_data(start: str = "2000-01-01", end: str = "2025-12-31") -> pd.DataFrame:
    """Convenience wrapper: download then clean."""
    raw = download_sp500(start, end)
    return clean_data(raw)
