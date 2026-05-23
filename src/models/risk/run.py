"""Run the full M3 risk pipeline and write artifacts.

Trains M3a-c (fraud, cancel, late delivery), computes M3d (disaster proxy),
aggregates everything into a product-month risk_drag table.

Outputs
-------
- forecasts/m3_fraud_orders.parquet           — order-level calibrated p_fraud
- forecasts/m3_cancel_orders.parquet          — order-level calibrated p_cancel
- forecasts/m3_late_orders.parquet            — order-level calibrated p_late
- forecasts/m3_<x>_metrics.json               — split-wise AUC/PR/Brier
- forecasts/m3_<x>_importance.parquet         — feature importance
- forecasts/m3_fraud_monthly.parquet          — product-month aggregated rate
- forecasts/m3_cancel_monthly.parquet
- forecasts/m3_late_monthly.parquet
- forecasts/m3d_disaster_region.parquet       — region-level disaster index
- forecasts/m3d_disaster_market.parquet       — market-level disaster index
- forecasts/m3d_disaster_product.parquet      — product-month disaster index
- forecasts/m3_risk_drag.parquet              — combined risk_drag for M4
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.risk.aggregate import combine
from src.models.risk.cancel import train as train_cancel
from src.models.risk.classifier import aggregate_to_product_month
from src.models.risk.data import add_split_and_features, load_orders
from src.models.risk import disaster as disaster_mod
from src.models.risk.fraud import train as train_fraud
from src.models.risk.late_delivery import train as train_late

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "forecasts"


def _save_metrics(name: str, metrics: dict) -> None:
    obj = {k: v for k, v in metrics.items() if k in ("uncalibrated", "calibrated")}
    (OUT_DIR / f"m3_{name}_metrics.json").write_text(json.dumps(obj, indent=2))


def _save_importance(name: str, metrics: dict) -> None:
    metrics["feature_importance"].to_parquet(OUT_DIR / f"m3_{name}_importance.parquet")


def _train_and_save(name: str, train_fn):
    log.info("training M3 sub-model: %s", name)
    (model, metrics), data = train_fn()

    _save_metrics(name, metrics)
    _save_importance(name, metrics)

    # Build full-period predictions by concatenating each split's
    # already-computed calibrated predictions (training did them with
    # consistent category dtypes).
    parts: list[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        d = data[split_name]
        cal_pred = metrics["predictions"][split_name]
        parts.append(pd.DataFrame({
            "order_date":      d["order_date"].to_numpy(),
            "product_card_id": d["product_card_id"].to_numpy(),
            "split":           split_name,
            "p":               cal_pred,
        }))
    order_df = pd.concat(parts, ignore_index=True)
    order_df.to_parquet(OUT_DIR / f"m3_{name}_orders.parquet", index=False)

    # Aggregate to product-month using actuals from the original data dict
    actual_parts = []
    for split_name in ("train", "val", "test"):
        actual_parts.append(data[split_name]["y"])
    actuals = np.concatenate(actual_parts)
    monthly = aggregate_to_product_month(
        order_df["p"].to_numpy(),
        pd.to_datetime(order_df["order_date"]),
        order_df["product_card_id"],
        actuals=actuals,
    )
    monthly.to_parquet(OUT_DIR / f"m3_{name}_monthly.parquet", index=False)
    log.info("  saved orders (%d rows) + monthly (%d rows)", len(order_df), len(monthly))

    val_m = metrics["calibrated"]["val"]
    test_m = metrics["calibrated"]["test"]
    log.info("  AUC val=%.4f test=%.4f  Brier val=%.4f test=%.4f",
             val_m["roc_auc"], test_m["roc_auc"],
             val_m["brier"], test_m["brier"])

    return monthly


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fraud_pm  = _train_and_save("fraud",  train_fraud)
    cancel_pm = _train_and_save("cancel", train_cancel)
    late_pm   = _train_and_save("late",   train_late)

    log.info("computing M3d disaster proxy")
    dis = disaster_mod.run()
    dis["region_index"].to_parquet(OUT_DIR / "m3d_disaster_region.parquet", index=False)
    dis["market_index"].to_parquet(OUT_DIR / "m3d_disaster_market.parquet", index=False)
    dis["country_index"].to_parquet(OUT_DIR / "m3d_disaster_country.parquet", index=False)
    dis["state_index"].to_parquet(OUT_DIR / "m3d_disaster_state.parquet", index=False)
    dis["product_index"].to_parquet(OUT_DIR / "m3d_disaster_product.parquet", index=False)

    log.info("combining sub-models into risk_drag")
    panel = pd.read_parquet(ROOT / "data" / "processed" / "monthly_panel.parquet")
    panel_clean = panel[panel["data_quality"] == "ok"]
    risk = combine(fraud_pm, cancel_pm, late_pm, dis["product_index"], panel_clean)
    risk.to_parquet(OUT_DIR / "m3_risk_drag.parquet", index=False)
    log.info("wrote m3_risk_drag.parquet  shape=%s", risk.shape)

    print("\n===== RISK_DRAG SUMMARY =====")
    print(risk[["p_fraud", "p_cancel", "p_late", "disaster_index", "risk_drag"]]
          .describe().round(4).to_string())

    # Per-split portfolio average
    print("\nrisk_drag by year-quarter:")
    risk["q"] = risk["year_month"].dt.to_period("Q").astype(str)
    print(risk.groupby("q")["risk_drag"].mean().round(4).to_string())


if __name__ == "__main__":
    run()
