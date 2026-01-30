from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def _ensure(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_anomaly_counts(df_scores: pd.DataFrame, outpath: str):
    _ensure(outpath)
    counts = df_scores[df_scores["is_anomaly"]].groupby("ticker").size().sort_values(ascending=False)
    plt.figure()
    counts.plot(kind="bar")
    plt.title("Anomaly Count by Ticker")
    plt.xlabel("Ticker")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_score_timeseries(df_scores: pd.DataFrame, ticker: str, outpath: str):
    _ensure(outpath)
    g = df_scores[df_scores["ticker"] == ticker].sort_index()
    plt.figure()
    plt.plot(g.index, g["score_ensemble"], label="ensemble_score")
    plt.title(f"Ensemble Anomaly Score — {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
