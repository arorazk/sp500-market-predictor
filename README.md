# S&P 500 Market Predictor

A Python ML project that predicts whether the S&P 500 will close **up or down** on the next trading day using a Random Forest classifier trained on 20+ years of historical data.

Built with proper time-series methodology: **no data leakage**, chronological expanding-window backtesting, and probability threshold tuning.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/sp500-market-predictor.git
cd sp500-market-predictor
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

The script downloads data, engineers features, runs the backtest, tunes the threshold, and saves charts to `./outputs/`.

## Project Structure

```
sp500-market-predictor/
├── src/
│   ├── data_loader.py   # Download & clean OHLCV data from Yahoo Finance
│   ├── features.py      # Leakage-safe feature engineering
│   ├── model.py         # RandomForestClassifier configuration
│   ├── backtest.py      # Expanding-window walk-forward engine
│   └── evaluate.py      # Metrics, threshold tuning, charts
├── main.py              # Orchestration entry point
├── requirements.txt
└── README.md
```

## Methodology

### Data

- **Source**: Yahoo Finance via `yfinance` (`^GSPC`)
- **Range**: January 2000 to present (~6,000+ trading days)
- **Target**: Binary — 1 if next day's close > today's close, 0 otherwise

### Features (11 total, all leakage-safe)

| Category   | Features                                                |
|------------|---------------------------------------------------------|
| Returns    | 1-day, 5-day, 10-day, 21-day log returns               |
| Momentum   | RSI-14, MACD signal delta                               |
| Volatility | 10-day and 21-day rolling std of daily returns          |
| Trend      | SMA 50/200 ratio, price-to-52-week-high ratio           |
| Volume     | 10-day / 21-day average volume ratio                    |

Every feature is **shifted forward by 1 day** before training so that row *t* uses only information available before day *t*'s close.

### Backtesting

- **Method**: Expanding-window walk-forward
- **Initial training window**: ~5 years (1,260 trading days)
- **Step size**: 63 days (~1 quarter)
- **No shuffle**: Strict chronological ordering preserved throughout
- **Retraining**: Full retrain at each fold on all available history

### Model

- `RandomForestClassifier` with `max_depth=8`, `min_samples_leaf=50`, `class_weight="balanced"`
- Probability threshold tuned by sweeping 0.40–0.70 and selecting the threshold that maximizes precision with recall ≥ 0.20

## Output

Running `python main.py` produces:

1. **Printed metrics table** — accuracy, precision, recall, F1, AUC-ROC at the tuned threshold
2. `outputs/threshold_tuning.png` — Precision/Recall/F1 vs. threshold sweep
3. `outputs/confusion_matrix.png` — Standard confusion matrix heatmap
4. `outputs/equity_curve.png` — Simplified strategy equity curve vs. buy-and-hold

## Honest Limitations

Daily equity direction is one of the hardest prediction problems in ML. Expect:

- **Precision around 53–56%** — a few points above the ~53% majority-class baseline
- **Weak but real signal** — enough to demonstrate methodology, not enough to trade on
- **No transaction costs** modeled in the equity curve

This project demonstrates rigorous ML methodology, not a profitable trading strategy.

## Resume Bullets

> - Built end-to-end ML pipeline predicting S&P 500 daily direction on 20+ years of market data using Random Forest with 11 engineered features and expanding-window backtesting
> - Implemented leakage-safe feature engineering (RSI, MACD, rolling volatility, trend ratios) with proper temporal shifting to prevent look-ahead bias
> - Designed walk-forward backtesting engine with quarterly retraining, probability threshold tuning, and automated evaluation producing precision, AUC-ROC, and equity curve analysis

## Future Improvements

1. **Macro features** — add VIX, 10-year Treasury yield, and dollar index for market regime context
2. **Gradient boosting** — swap Random Forest for LightGBM or XGBoost, which typically improve AUC by 3–5% on financial tabular data
3. **Walk-forward hyperparameter tuning** — use `sklearn.model_selection.TimeSeriesSplit` inside each backtest fold for nested cross-validation

## License

MIT
