"""M5 — Anomaly detection orchestrator.

Runs the three detector layers, fuses hits, assigns severity, and writes:

- forecasts/m5_anomaly_scores.parquet — per-(product, month) layer scores
- anomalies/alerts.parquet            — long-form alert log with severity
                                         and suspected driver
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.models.anomaly.forecast_deviation import detect_forecast_deviation
from src.models.anomaly.isolation_forest import detect_iforest
from src.models.anomaly.severity import (
    _combine_hits, _max_stl_z, assign_severity,
)
from src.models.anomaly.stl_detector import detect_stl
from src.models.risk.known_disasters_v2 import state_monthly_indicator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
FC_DIR = ROOT / "forecasts"
ANOM_DIR = ROOT / "anomalies"


def _project_known_severity(panel: pd.DataFrame) -> pd.DataFrame:
    """Map per-state known-event severity to per-product via customer mix."""
    import sqlite3
    with sqlite3.connect(ROOT / "data" / "dataco.db") as conn:
        mix = pd.read_sql(
            "SELECT order_date, product_card_id, customer_country FROM supply_chain",
            conn, parse_dates=["order_date"])
    mix["year_month"] = mix["order_date"].dt.to_period("M").dt.to_timestamp()
    mix = (mix.assign(_one=1)
              .pivot_table(index=["product_card_id", "year_month"],
                           columns="customer_country", values="_one",
                           aggfunc="sum", fill_value=0))
    mix = mix.div(mix.sum(axis=1).replace(0, 1), axis=0).reset_index()

    # Use country-level severity (state-level is too granular for the panel)
    from src.models.risk.known_disasters_v2 import country_monthly_indicator
    cty = country_monthly_indicator()
    sev = cty.pivot(index="year_month", columns="customer_country",
                    values="known_severity").reset_index().fillna(0)
    out = mix.merge(sev, on="year_month", how="left", suffixes=("_mix", "_sev"))
    out = out.fillna(0)
    countries = list(cty["customer_country"].unique())
    out["known_severity"] = sum(
        out.get(f"{c}_mix", 0) * out.get(f"{c}_sev", 0) for c in countries
    )
    return out[["product_card_id", "year_month", "known_severity"]]


def run() -> None:
    log.info("loading inputs")
    panel = pd.read_parquet(ROOT / "data" / "processed" / "monthly_panel.parquet")
    risk = pd.read_parquet(FC_DIR / "m3_risk_drag.parquet")
    elast = pd.read_parquet(FC_DIR / "m2_elasticity_monthly.parquet")
    m1 = pd.read_parquet(FC_DIR / "m1_demand.parquet")

    log.info("layer 1: STL residuals")
    stl_hits = detect_stl(panel, elast)
    log.info("  %d STL hits", len(stl_hits))

    log.info("layer 2: IsolationForest")
    if_hits = detect_iforest(panel, risk, elast)
    log.info("  %d IsolationForest hits", len(if_hits))

    log.info("layer 3: forecast-deviation")
    fdev_hits = detect_forecast_deviation(m1, panel, risk)
    log.info("  %d forecast-deviation hits", len(fdev_hits))

    log.info("combining layers + severity")
    combined = _combine_hits(stl_hits, if_hits, fdev_hits)
    stl_summary = _max_stl_z(stl_hits)
    known_sev = _project_known_severity(panel)
    alerts = assign_severity(combined, stl_summary, panel, risk, known_sev)
    # Attach the known-disaster flag for downstream filtering / dashboards
    alerts = alerts.merge(known_sev, on=["product_card_id", "year_month"], how="left")
    alerts["known_disaster"] = alerts["known_severity"].fillna(0) > 0

    # Per-(product, month) scores for the dashboard
    scores = (combined.merge(stl_summary, on=["product_card_id", "year_month"], how="left")
                      .merge(if_hits.rename(columns={"score": "if_score"})
                                    [["product_card_id", "year_month", "if_score"]],
                             on=["product_card_id", "year_month"], how="left")
                      .merge(fdev_hits.rename(columns={"score": "fdev_score"})
                                      [["product_card_id", "year_month", "fdev_score"]],
                             on=["product_card_id", "year_month"], how="left"))

    ANOM_DIR.mkdir(parents=True, exist_ok=True)
    FC_DIR.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(ANOM_DIR / "alerts.parquet", index=False)
    scores.to_parquet(FC_DIR / "m5_anomaly_scores.parquet", index=False)
    log.info("wrote alerts.parquet (%d rows) + m5_anomaly_scores.parquet (%d rows)",
             len(alerts), len(scores))

    # Headline summary
    print("\n===== M5 ALERT SUMMARY =====")
    print(alerts.groupby("severity").size().rename("count").to_string())
    print("\nBy suspected driver:")
    print(alerts.groupby("suspected_driver").size().sort_values(ascending=False)
                .rename("count").to_string())
    print(f"\nAlerts in known-disaster months: {alerts['known_disaster'].sum()}/{len(alerts)}")
    print("\nTop 10 critical alerts:")
    crit = alerts[alerts["severity"] == "critical"].sort_values("max_stl_z", ascending=False)
    if len(crit):
        print(crit[["product_card_id", "year_month", "n_layers",
                    "max_stl_z", "suspected_driver", "known_disaster"]].head(10).to_string(index=False))


if __name__ == "__main__":
    run()
