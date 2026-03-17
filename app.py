"""
Streamlit web app for the S&P 500 Market Predictor.

Deploy for free at: https://streamlit.io/cloud
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.mpl_cache")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_data
from src.features import add_features, FEATURE_COLS
from src.backtest import expanding_window_backtest


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="S&P 500 Market Predictor",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached pipeline — runs once, then serves from cache
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_pipeline():
    df_raw = load_data(start="2000-01-01")
    df = add_features(df_raw)
    results = expanding_window_backtest(
        df,
        feature_cols=FEATURE_COLS,
        target_col="Target",
        initial_train_days=1260,
        step_days=63,
    )
    return results, df_raw, df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def make_threshold_chart(y_true, y_prob, best_threshold):
    thresholds = np.linspace(0.40, 0.70, 61)
    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        records.append({
            "threshold": t,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall":    recall_score(y_true, y_pred, zero_division=0),
            "F1":        f1_score(y_true, y_pred, zero_division=0),
        })
    sweep = pd.DataFrame(records).set_index("threshold")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    for col, ls in zip(["Precision", "Recall", "F1"], ["-", "-", "--"]):
        ax.plot(sweep.index, sweep[col], label=col, linewidth=2, linestyle=ls)
    ax.axvline(best_threshold, color="grey", linestyle=":", linewidth=1.5,
               label=f"Selected={best_threshold:.2f}")
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep: Precision / Recall / F1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def make_equity_curve(results, threshold):
    df = results.copy()
    df["y_pred"] = (df["y_prob"] >= threshold).astype(int)
    df["market_return"]   = df["y_true"].map({1: 1, 0: -1}) * 0.0004
    df["strategy_return"] = df["market_return"] * df["y_pred"]
    df["market_equity"]   = (1 + df["market_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(df["date"], df["market_equity"],   label="Buy & Hold",      linewidth=1.5)
    ax.plot(df["date"], df["strategy_equity"], label="Model Strategy",  linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Equity (start = 1.0)")
    ax.set_title("Strategy vs Buy & Hold (no transaction costs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_feature_importance_chart(df, feature_cols):
    from src.model import build_model, train
    sample = df.iloc[:1260]
    model = build_model()
    train(model, sample[feature_cols], sample["Target"])
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importances (first training fold)")
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("📈 S&P 500 Market Predictor")
st.markdown(
    "Predicts whether the S&P 500 will close **up or down** tomorrow using a "
    "Random Forest trained on 25+ years of data. "
    "Built with **leakage-safe feature engineering** and **expanding-window backtesting**."
)
st.divider()

# Run pipeline with progress message
with st.spinner("Running backtest across 80 quarterly folds (cached after first load)..."):
    results, df_raw, df = run_pipeline()

y_true = results["y_true"].values
y_prob = results["y_prob"].values

# ---------------------------------------------------------------------------
# Sidebar: threshold control
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Probability threshold",
    min_value=0.40, max_value=0.70,
    value=0.44, step=0.01,
    help="Model predicts 'Up' when P(up) ≥ this value.",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**How it works**\n\n"
    "- Lower threshold → more aggressive, higher recall\n"
    "- Higher threshold → more selective, higher precision\n"
    "- Default 0.44 maximises precision while keeping recall ≥ 20%"
)

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

y_pred = (y_prob >= threshold).astype(int)
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
auc  = roc_auc_score(y_true, y_prob)
baseline = max(y_true.mean(), 1 - y_true.mean())

st.subheader("Live Metrics")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Accuracy",  f"{acc:.1%}",  f"+{acc - baseline:.1%} vs baseline")
c2.metric("Precision", f"{prec:.1%}")
c3.metric("Recall",    f"{rec:.1%}")
c4.metric("F1",        f"{f1:.1%}")
c5.metric("AUC-ROC",   f"{auc:.3f}")
c6.metric("Predictions", f"{len(y_true):,}")

st.caption(
    f"Baseline (majority class): **{baseline:.1%}** accuracy  |  "
    f"5,024 out-of-sample days (2006–2025)  |  80 quarterly retraining folds"
)
st.divider()

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

st.subheader("Charts")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Equity Curve", "Threshold Sweep", "Confusion Matrix", "Feature Importances"]
)

with tab1:
    st.pyplot(make_equity_curve(results, threshold))
    st.caption("Simplified simulation: long on predicted up-days, cash otherwise. No transaction costs.")

with tab2:
    st.pyplot(make_threshold_chart(y_true, y_prob, threshold))
    st.caption("Move the sidebar slider to shift the vertical line and see metrics update above.")

with tab3:
    st.pyplot(make_confusion_matrix(y_true, y_pred))

with tab4:
    st.pyplot(make_feature_importance_chart(df, FEATURE_COLS))
    st.caption("Trained on the first 5-year window (2001–2006).")

st.divider()

# ---------------------------------------------------------------------------
# Methodology summary
# ---------------------------------------------------------------------------

with st.expander("Methodology", expanded=False):
    st.markdown("""
**Data**: Yahoo Finance `^GSPC`, Jan 2000 – Dec 2025 (~6,500 trading days)

**Features (11 total, all leakage-safe)**

| Category | Features |
|---|---|
| Returns | 1-day, 5-day, 10-day, 21-day log returns |
| Momentum | RSI-14, MACD signal delta |
| Volatility | 10-day and 21-day rolling std |
| Trend | SMA 50/200 ratio, price-to-52-week-high |
| Volume | 10-day / 21-day volume ratio |

Every feature is **shifted forward 1 day** before training — row *t* uses only data available before day *t*'s close.

**Backtesting**: Expanding-window walk-forward — initial 5-year training window, quarterly retraining, no data shuffle.

**Model**: `RandomForestClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced")`
    """)

with st.expander("Honest Limitations", expanded=False):
    st.markdown("""
Daily equity direction is one of the hardest prediction problems in ML:

- **AUC-ROC near 0.50** — consistent with efficient market hypothesis; short-term price movements are close to random
- **Precision only slightly above baseline** — the signal is real but weak
- **No transaction costs** in the equity curve — real trading would reduce returns further

This project demonstrates rigorous ML methodology, not a profitable trading strategy.
    """)
