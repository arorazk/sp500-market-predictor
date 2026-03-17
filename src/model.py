"""
Thin wrapper around scikit-learn's RandomForestClassifier.

Keeping model construction separate makes it easy to swap classifiers later
without touching backtest or evaluation code.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def build_model(
    n_estimators: int = 300,
    max_depth: Optional[int] = 8,
    min_samples_leaf: int = 50,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Return a configured (but untrained) RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def train(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Fit the model in-place and return it."""
    model.fit(X, y)
    return model


def predict_proba(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return probability of the positive class (market goes up)."""
    return model.predict_proba(X)[:, 1]
