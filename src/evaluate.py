"""
Evaluation utilities: metrics computation, threshold tuning, and chart generation.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


OUTPUT_DIR = Path("outputs")


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Return a dict of classification metrics at the given probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "n_predictions": len(y_true),
        "n_positive_preds": int(y_pred.sum()),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print a metrics dictionary."""
    print("\n" + "=" * 52)
    print("  BACKTEST RESULTS")
    print("=" * 52)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<22s}  {val:.4f}")
        else:
            print(f"  {key:<22s}  {val}")
    print("=" * 52 + "\n")


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    low: float = 0.40,
    high: float = 0.70,
    steps: int = 61,
) -> float:
    """
    Sweep probability thresholds and plot Precision / Recall / F1.
    Returns the threshold that maximises precision while keeping recall >= 0.20.
    """
    thresholds = np.linspace(low, high, steps)
    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        records.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    sweep = pd.DataFrame(records)

    # Pick best: maximise precision where recall is still meaningful
    viable = sweep[sweep["recall"] >= 0.20]
    if viable.empty:
        best_threshold = 0.50
    else:
        best_threshold = viable.loc[viable["precision"].idxmax(), "threshold"]

    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sweep["threshold"], sweep["precision"], label="Precision", linewidth=2)
    ax.plot(sweep["threshold"], sweep["recall"], label="Recall", linewidth=2)
    ax.plot(sweep["threshold"], sweep["f1"], label="F1", linewidth=2, linestyle="--")
    ax.axvline(best_threshold, color="grey", linestyle=":", label=f"Best={best_threshold:.2f}")
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning: Precision / Recall / F1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "threshold_tuning.png", dpi=150)
    plt.close(fig)
    print(f"  [chart] Saved threshold_tuning.png  (best threshold = {best_threshold:.2f})")
    return best_threshold


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_equity_curve(df_results: pd.DataFrame, threshold: float = 0.5) -> None:
    """
    Simulated strategy equity curve vs buy-and-hold.

    Strategy: go long on days the model predicts 'up' (prob >= threshold),
    sit in cash otherwise.  Assumes no transaction costs.
    """
    df = df_results.copy()
    df["y_pred"] = (df["y_prob"] >= threshold).astype(int)

    # Daily return proxy: +1 if market went up, -1 if down
    df["market_return"] = df["y_true"].map({1: 1, 0: -1}) * 0.0004  # ~10 bps avg daily
    df["strategy_return"] = df["market_return"] * df["y_pred"]

    df["market_equity"] = (1 + df["market_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["date"], df["market_equity"], label="Buy & Hold", linewidth=1.5)
    ax.plot(df["date"], df["strategy_equity"], label="Model Strategy", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Equity (starting at 1.0)")
    ax.set_title("Strategy vs Buy & Hold (simplified)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150)
    plt.close(fig)
    print("  [chart] Saved equity_curve.png")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save a confusion matrix heatmap."""
    _ensure_output_dir()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("  [chart] Saved confusion_matrix.png")
