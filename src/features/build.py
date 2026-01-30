from __future__ import annotations
import numpy as np
import pandas as pd

def _zscore(x: pd.Series, window: int) -> pd.Series:
    m = x.rolling(window).mean()
    s = x.rolling(window).std(ddof=1)
    return (x - m) / s

def build_features(ohlcv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Returns a long-form DataFrame indexed by date with columns:
      ticker, ret, ret_z, vol, vol_z, volu_z, range, range_z, ...
    """
    vol_window = int(cfg["features"]["vol_window"])
    vol_long_window = int(cfg["features"]["vol_long_window"])
    volume_window = int(cfg["features"]["volume_window"])
    range_window = int(cfg["features"]["range_window"])
    min_history = int(cfg["features"]["min_history"])

    # Pull panels
    adj = ohlcv["Adj Close"].copy()
    high = ohlcv["High"].copy()
    low = ohlcv["Low"].copy()
    volu = ohlcv["Volume"].copy()

    # Daily returns
    ret = adj.pct_change()

    # Realized vol proxies
    vol = ret.rolling(vol_window).std(ddof=1)
    vol_long = ret.rolling(vol_long_window).std(ddof=1)

    # Volume features
    volu_log = np.log1p(volu)
    volu_z = volu_log.apply(lambda s: _zscore(s, volume_window))

    # Range feature (high-low relative to adj close)
    hl_range = (high - low) / adj
    range_z = hl_range.apply(lambda s: _zscore(s, range_window))

    # Return z-score (rolling)
    ret_z = ret.apply(lambda s: _zscore(s, vol_window))

    # Vol z-score relative to longer-term vol
    # (vol - long_mean)/long_std computed on vol itself
    vol_z = (vol - vol_long.rolling(vol_long_window).mean()) / vol_long.rolling(vol_long_window).std(ddof=1)

    # Build long-form table
    out = []
    for t in adj.columns:
        df = pd.DataFrame({
            "ticker": t,
            "ret": ret[t],
            "ret_z": ret_z[t],
            "vol": vol[t],
            "vol_z": vol_z[t],
            "volu_z": volu_z[t],
            "range": hl_range[t],
            "range_z": range_z[t],
        }, index=adj.index)
        out.append(df)

    feats = pd.concat(out).sort_index()
    feats = feats.dropna()
    # require minimum history per ticker
    feats = feats.groupby("ticker").filter(lambda g: len(g) >= min_history)
    return feats
