from __future__ import annotations
import numpy as np
import pandas as pd

def hit_rate_extreme_moves(df_scores: pd.DataFrame, feats: pd.DataFrame, q: float = 0.99) -> float:
    """
    Proxy precision: what fraction of detected anomalies coincide with top q quantile abs returns (per ticker).
    """
    tmp = df_scores.copy()
    tmp["abs_ret"] = feats["ret"].abs().reindex(tmp.index)
    tmp = tmp.dropna(subset=["abs_ret"])

    def per_ticker_hr(g: pd.DataFrame) -> float:
        thr = g["abs_ret"].quantile(q)
        denom = g["is_anomaly"].sum()
        if denom == 0:
            return np.nan
        return float(((g["is_anomaly"]) & (g["abs_ret"] >= thr)).sum() / denom)

    hrs = tmp.groupby("ticker").apply(per_ticker_hr).dropna()
    return float(hrs.mean()) if len(hrs) else float("nan")
