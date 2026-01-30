from __future__ import annotations
import os
import yaml
import pandas as pd

from src.data.fetch import fetch_ohlcv
from src.features.build import build_features
from src.monitoring.detect import detect_anomalies
from src.evaluation.event_study import event_study
from src.evaluation.metrics import hit_rate_extreme_moves

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
    os.makedirs(out_reports, exist_ok=True)

    ohlcv = fetch_ohlcv(tickers, start, end, cache_path=cfg["data"]["cache_path"])
    adj = ohlcv["Adj Close"].copy().dropna(how="all")

    feats = build_features(ohlcv, cfg)
    scores = detect_anomalies(feats, cfg)

    horizons = [int(x) for x in cfg["evaluation"]["horizons_days"]]
    es = event_study(scores, adj, horizons)
    es_path = os.path.join(out_reports, "event_study.csv")
    es.to_csv(es_path, index=False)

    hr = hit_rate_extreme_moves(scores, feats, q=0.99)
    hr_path = os.path.join(out_reports, "hit_rate_extreme_moves.txt")
    with open(hr_path, "w") as f:
        f.write(f"HitRate_Top1pctAbsReturn: {hr:.6f}\n")

    print("\n=== Backtest Evaluation Complete ===")
    print(f"- Event study: {es_path}")
    print(f"- Hit rate (top 1% abs moves): {hr:.4f}")

if __name__ == "__main__":
    main()
