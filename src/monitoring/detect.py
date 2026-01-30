from __future__ import annotations
import numpy as np
import pandas as pd

from src.models.zscore import zscore_anomaly_score
from src.models.isolation_forest import fit_predict_iforest

def detect_anomalies(feats: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Returns DataFrame with:
      date index, ticker, score_z, score_iforest, score_ensemble, is_anomaly
    """
    s_z = zscore_anomaly_score(feats, cfg).reindex(feats.index)
    s_if = fit_predict_iforest(feats, cfg).reindex(feats.index)


    df = feats.copy()
    df["score_z"] = s_z
    df["score_iforest"] = s_if

    w_z = float(cfg["monitoring"]["score_ensemble"]["w_zscore"])
    w_if = float(cfg["monitoring"]["score_ensemble"]["w_iforest"])

    # fill missing
    df["score_z"] = df["score_z"].fillna(0.0)
    df["score_iforest"] = df["score_iforest"].fillna(0.0)

    df["score_ensemble"] = w_z * df["score_z"] + w_if * df["score_iforest"]
    
    # anomaly label: top X% per ticker (uses contamination as proxy)
    contam = float(cfg["models"]["isolation_forest"]["contamination"])

    thr = df.groupby("ticker")["score_ensemble"].transform(
        lambda s: s.quantile(1.0 - contam)
    )
    df["is_anomaly"] = df["score_ensemble"] >= thr

    df.index.name = "date"
    return df.sort_index()


