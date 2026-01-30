# Real-Time–Ready Market Anomaly Detection Using Unsupervised Learning

## Overview
This project implements a **real-time–ready market anomaly detection system** designed to identify unusual behavior in financial markets using **unsupervised machine learning**. The system processes market data sequentially, generates anomaly scores using multiple models, and triggers alerts when abnormal price, volatility, or volume patterns emerge.

The framework is evaluated using **point-in-time backtesting** and **event-study analysis** to quantify whether detected anomalies are economically meaningful rather than merely statistical outliers.

---

## Key Objectives
- Detect abnormal market behavior without labeled anomaly data
- Combine interpretable rules with machine learning–based detectors
- Evaluate anomaly signals using forward returns and extreme-move hit rates
- Design a pipeline that is **streaming-ready** and production-oriented

---

## Data
- **Assets:** 8 highly liquid ETFs (e.g., SPY, QQQ, IWM, TLT, GLD, HYG, XLF, XLK)
- **Frequency:** Daily OHLCV data
- **Source:** Yahoo Finance
- **Period:** 2010–present (varies by asset)

---

## Feature Engineering
At each timestamp, features are computed using only historical information:

### Price & Returns
- Daily returns
- Rolling return z-scores

### Volatility
- Short-term realized volatility
- Volatility z-scores relative to long-term baselines

### Volume & Liquidity
- Log volume
- Rolling volume z-scores

### Range / Shock Measures
- High–low price range
- Range z-scores

These features are designed to capture **market shocks**, **liquidity stress**, and **abnormal price dynamics**.

---

## Models

### 1. Rule-Based Baseline (Z-Score Detector)
- Flags extreme deviations in returns, volatility, and volume
- Fully interpretable and fast to compute

### 2. Isolation Forest
- Unsupervised anomaly detection on engineered feature vectors
- Trained separately per asset to avoid cross-asset scale effects
- Produces continuous anomaly scores

### 3. Ensemble Scoring
Final anomaly score is a weighted combination of:
- Z-score–based anomaly signal
- Isolation Forest anomaly score

Anomalies are defined as observations in the **top tail of the ensemble score distribution** per asset.

---

## Alerting & Monitoring
For each new observation:
1. Features are updated
2. Anomaly scores are computed
3. Alerts are generated when thresholds are breached

Outputs include:
- `alerts.csv`: all detected anomalies
- `top_anomalies_daily.csv`: top anomalies across assets per day

This design mirrors how a live monitoring system would operate, with historical data used solely for validation.

---

## Evaluation Methodology

Because anomalies are unlabeled, performance is evaluated using **proxy metrics commonly used in practice**:

### 1. Event Study (Forward Returns)
Forward returns following anomaly events are compared to non-anomalous periods.

| Horizon | Anomaly Mean Return | Non-Anomaly Mean Return |
|-------|---------------------|-------------------------|
| 1 Day | −0.14% | 0.05% |
| 5 Days | −0.04% | 0.25% |
| **20 Days** | **~3.3%** | **~1.0%** |

Anomalies exhibit **significantly larger medium-term price impact**, particularly over a 20-day horizon.

---

### 2. Extreme-Move Hit Rate
Anomalies are evaluated on their ability to identify extreme market events.

- **~41% hit rate** for detecting days in the **top 1% of absolute return moves**
- Far exceeds random selection, validating signal relevance

---

## Key Findings
- Anomaly signals are **not random noise**; they are associated with economically meaningful future price movements.
- Detected anomalies show **strong medium-term impact**, even when short-term returns are muted.
- Combining interpretable rules with unsupervised ML improves robustness and stability.
- The framework balances **model sophistication** with **operational simplicity**, making it suitable for real-world monitoring.

---

## Project Structure
market-anomaly-detection/
configs/
scripts/
src/
outputs/ (generated, ignored by git)
---

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m scripts.run_monitor
python -m scripts.run_backtest_eval
```
Results are saved to the outputs/ directory, including alerts, evaluation reports, and plots.

Real-Time Note

While historical data is used for validation, the system operates sequentially and point-in-time, making it real-time–ready by design. Replacing the data source with a live feed would require no changes to the core detection logic.

Key Takeaway

This project demonstrates how unsupervised learning, feature engineering, and event-based evaluation can be combined to build a practical market monitoring system that detects and validates anomalous behavior without labeled data.