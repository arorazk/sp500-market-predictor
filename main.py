"""
S&P 500 Market Predictor — main entry point.

Usage:
    python main.py
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mpl_cache"))

import numpy as np

from src.data_loader import load_data
from src.features import add_features, FEATURE_COLS
from src.backtest import expanding_window_backtest
from src.evaluate import (
    compute_metrics,
    print_metrics,
    tune_threshold,
    plot_equity_curve,
    plot_confusion_matrix,
)


def main() -> None:
    # 1. Load & clean data
    print("\n[1/5] Downloading S&P 500 data from Yahoo Finance...")
    df = load_data(start="2000-01-01")
    print(f"       {len(df)} trading days  ({df.index[0].date()} → {df.index[-1].date()})")

    # 2. Feature engineering
    print("\n[2/5] Engineering features (leakage-safe)...")
    df = add_features(df)
    print(f"       {len(df)} rows after warm-up drop  |  {len(FEATURE_COLS)} features")

    # 3. Walk-forward backtest
    print("\n[3/5] Running expanding-window backtest...")
    results = expanding_window_backtest(
        df,
        feature_cols=FEATURE_COLS,
        target_col="Target",
        initial_train_days=1260,
        step_days=63,
    )
    print(f"       {len(results)} out-of-sample predictions collected")

    y_true = results["y_true"].values
    y_prob = results["y_prob"].values

    # 4. Threshold tuning
    print("\n[4/5] Tuning probability threshold...")
    best_threshold = tune_threshold(y_true, y_prob)

    # 5. Final evaluation at the tuned threshold
    print("\n[5/5] Evaluating at tuned threshold...")
    metrics = compute_metrics(y_true, y_prob, threshold=best_threshold)
    print_metrics(metrics)

    y_pred = (y_prob >= best_threshold).astype(int)
    plot_confusion_matrix(y_true, y_pred)
    plot_equity_curve(results, threshold=best_threshold)

    # Baseline comparison
    majority_class_rate = max(y_true.mean(), 1 - y_true.mean())
    print(f"  Baseline (always predict majority class): {majority_class_rate:.4f} accuracy")
    print(f"  Model accuracy at threshold {best_threshold:.2f}:          {metrics['accuracy']:.4f}")
    print(f"  Model precision at threshold {best_threshold:.2f}:         {metrics['precision']:.4f}")
    print()

    # Honest disclaimer
    if metrics["precision"] < 0.56:
        print(
            "  ⚠  Precision is below 56%. This is common for daily equity prediction.\n"
            "     The model captures weak signal — useful for learning, not for trading.\n"
        )

    print("  Charts saved to ./outputs/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
