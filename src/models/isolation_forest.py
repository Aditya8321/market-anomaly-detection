from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLS = ["ret", "vol", "vol_z", "volu_z", "range", "range_z", "ret_z"]

def fit_predict_iforest(feats: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Fit an IsolationForest per ticker and output anomaly score.
    Using per-ticker models avoids cross-ticker scale issues.
    """
    mcfg = cfg["models"]["isolation_forest"]
    contamination = float(mcfg["contamination"])
    n_estimators = int(mcfg["n_estimators"])
    random_state = int(mcfg["random_state"])

    scores = []

    for t, g in feats.groupby("ticker"):
        x = g[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 200:
            # not enough history
            continue

        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(x.values)

        # decision_function: higher = more normal; we invert to get anomaly score
        normality = model.decision_function(x.values)
        s = -normality
        s = (s - s.mean()) / (s.std(ddof=1) + 1e-12)

        ss = pd.Series(s, index=x.index, name="score_iforest")
        ss = ss.reindex(g.index)  # align back
        scores.append(ss)

    if not scores:
        raise RuntimeError("IsolationForest produced no scores (insufficient history).")

    out = pd.concat(scores).sort_index()
    return out
