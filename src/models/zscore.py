from __future__ import annotations
import numpy as np
import pandas as pd

def zscore_anomaly_score(feats: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Produces a per-row anomaly score using rule-based z-score thresholds.
    Returns a Series indexed exactly like feats.index.
    """
    mcfg = cfg["models"]["zscore"]
    thr_ret = float(mcfg["ret_z_thresh"])
    thr_vol = float(mcfg["vol_z_thresh"])
    thr_volu = float(mcfg["volu_z_thresh"])

    def excess(z: pd.Series, thr: float) -> pd.Series:
        return (z.abs() - thr).clip(lower=0.0)

    raw = (
        1.0 * excess(feats["ret_z"], thr_ret) +
        0.7 * excess(feats["vol_z"].fillna(0.0), thr_vol) +
        0.5 * excess(feats["volu_z"], thr_volu) +
        0.3 * feats["range_z"].abs().fillna(0.0)
    )

    # IMPORTANT: use transform (not apply) so index stays identical
    z = raw.groupby(feats["ticker"]).transform(
        lambda x: (x - x.mean()) / (x.std(ddof=1) + 1e-12)
    )

    return z.rename("score_z")
