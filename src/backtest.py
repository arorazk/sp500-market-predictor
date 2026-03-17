"""
Expanding-window walk-forward backtester.

The engine trains on all data up to a cutoff, predicts the next `step_days`
trading days, then slides the cutoff forward and repeats.  This mimics how
a real trading system would be retrained on a quarterly schedule.
"""

from typing import List

import pandas as pd
import numpy as np

from src.model import build_model, train, predict_proba


def expanding_window_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Target",
    initial_train_days: int = 1260,
    step_days: int = 63,
) -> pd.DataFrame:
    """
    Run an expanding-window backtest.

    Parameters
    ----------
    df : DataFrame with features, target, and a DatetimeIndex (sorted).
    feature_cols : columns used as model inputs.
    target_col : name of the binary target column.
    initial_train_days : rows used for the first training window (~5 years).
    step_days : how many rows to predict before retraining (~1 quarter).

    Returns
    -------
    DataFrame with columns: date, y_true, y_prob
    """
    results = []
    n = len(df)

    if n < initial_train_days + step_days:
        raise ValueError(
            f"Dataset has {n} rows but needs at least "
            f"{initial_train_days + step_days} for one backtest fold."
        )

    cutoff = initial_train_days
    fold = 1

    while cutoff < n:
        test_end = min(cutoff + step_days, n)

        train_df = df.iloc[:cutoff]
        test_df = df.iloc[cutoff:test_end]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        model = build_model()
        train(model, X_train, y_train)
        probs = predict_proba(model, X_test)

        fold_results = pd.DataFrame({
            "date": test_df.index,
            "y_true": y_test.values,
            "y_prob": probs,
        })
        results.append(fold_results)

        print(
            f"  Fold {fold:>3d}  |  "
            f"Train {train_df.index[0].date()}→{train_df.index[-1].date()}  |  "
            f"Test {test_df.index[0].date()}→{test_df.index[-1].date()}  |  "
            f"{len(test_df)} days"
        )

        cutoff = test_end
        fold += 1

    return pd.concat(results, ignore_index=True)
