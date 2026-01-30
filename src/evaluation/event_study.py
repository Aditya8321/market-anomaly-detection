from __future__ import annotations
import numpy as np
import pandas as pd

def compute_forward_returns(adj_close: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = []
    for t in adj_close.columns:
        df = pd.DataFrame(index=adj_close.index)
        df["ticker"] = t
        for h in horizons:
            df[f"fwd_ret_{h}d"] = adj_close[t].shift(-h) / adj_close[t] - 1.0
        out.append(df)
    fwd = pd.concat(out).sort_index()
    fwd.index.name = "date"
    return fwd

def event_study(df_scores: pd.DataFrame, adj_close: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Join anomaly events with forward returns and compute summary stats:
    mean/median forward returns for anomaly vs non-anomaly.
    """
    fwd = compute_forward_returns(adj_close, horizons)

    # Build MultiIndex (date, ticker) for both tables to avoid column overlap
    scores = df_scores.copy()
    scores.index.name = "date"
    scores = scores.reset_index()[["date", "ticker", "is_anomaly", "score_ensemble"]]
    scores = scores.set_index(["date", "ticker"]).sort_index()

    fwd = fwd.reset_index().set_index(["date", "ticker"]).sort_index()

    joined = scores.join(fwd, how="inner").dropna()

    rows = []
    for h in horizons:
        col = f"fwd_ret_{h}d"
        for flag in [True, False]:
            g = joined.loc[joined["is_anomaly"] == flag, col]
            rows.append({
                "horizon_days": h,
                "group": "anomaly" if flag else "non_anomaly",
                "count": int(g.shape[0]),
                "mean": float(g.mean()),
                "median": float(g.median()),
            })

    return pd.DataFrame(rows)
