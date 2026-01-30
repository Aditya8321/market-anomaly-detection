from __future__ import annotations
import os
import yaml
import pandas as pd

from src.data.fetch import fetch_ohlcv
from src.features.build import build_features
from src.monitoring.detect import detect_anomalies
from src.monitoring.alerting import write_alerts
from src.reporting.plots import plot_anomaly_counts, plot_score_timeseries

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_yaml("configs/run.yaml")
    uni = load_yaml(cfg["data"]["universe_config"])

    tickers = uni["tickers"]
    start = uni["start"]
    end = uni["end"]

    base_dir = cfg["outputs"]["base_dir"]
    out_reports = os.path.join(base_dir, "reports")
    out_figs = os.path.join(base_dir, "figures")
    os.makedirs(out_reports, exist_ok=True)
    os.makedirs(out_figs, exist_ok=True)

    ohlcv = fetch_ohlcv(tickers, start, end, cache_path=cfg["data"]["cache_path"])

    feats = build_features(ohlcv, cfg)
    feats.to_csv(os.path.join(out_reports, "features.csv"))

    scores = detect_anomalies(feats, cfg)
    scores.to_csv(os.path.join(out_reports, "scores.csv"))

    alerts_path, top_path = write_alerts(
        scores,
        out_dir=out_reports,
        top_k_daily=int(cfg["monitoring"]["top_k_daily"])
    )

    # plots
    plot_anomaly_counts(scores, os.path.join(out_figs, "anomaly_counts.png"))
    # plot one representative ticker
    plot_score_timeseries(scores, ticker=tickers[0], outpath=os.path.join(out_figs, f"score_{tickers[0]}.png"))

    print("\n=== Monitoring Run Complete ===")
    print("Saved:")
    print(f"- Features: {out_reports}/features.csv")
    print(f"- Scores:   {out_reports}/scores.csv")
    print(f"- Alerts:   {alerts_path}")
    print(f"- Top-K:    {top_path}")
    print(f"- Figures:  {out_figs}/")

if __name__ == "__main__":
    main()
