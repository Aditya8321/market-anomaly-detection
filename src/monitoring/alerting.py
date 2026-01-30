from __future__ import annotations
import os
import pandas as pd

def write_alerts(df_scores: pd.DataFrame, out_dir: str, top_k_daily: int) -> tuple[str, str]:
    """
    Writes:
      outputs/alerts.csv: all anomaly rows
      outputs/top_anomalies_daily.csv: top K anomalies per day across tickers
    """
    os.makedirs(out_dir, exist_ok=True)

    alerts = df_scores[df_scores["is_anomaly"]].copy()
    alerts_path = os.path.join(out_dir, "alerts.csv")
    alerts.to_csv(alerts_path, index=True)

    # top K anomalies per day
    top = (df_scores
           .reset_index()
           .rename(columns={"index": "date"})
           .sort_values(["date", "score_ensemble"], ascending=[True, False])
           .groupby("date")
           .head(int(top_k_daily))
           .set_index("date")
           )
    top_path = os.path.join(out_dir, "top_anomalies_daily.csv")
    top.to_csv(top_path, index=True)

    return alerts_path, top_path
