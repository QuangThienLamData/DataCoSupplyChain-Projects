"""Reshape daily pipeline outputs into unified per-product time series.

Produces three parquet files with a single shared schema:

    product_card_id  date  data_type  p10  p50  p90  actual

- `data_type` ∈ {'Actual', 'Prediction'}.
- `p10/p50/p90` are the prediction quantiles (NaN on Actual rows).
- `actual` carries the observed value (NaN on Prediction rows).

Outputs:
    forecasts/m1_daily.parquet  — gross_qty (demand)
    forecasts/m3_daily.parquet  — disaster_drag_index (risk drag the M4 math uses)
    forecasts/m4_daily.parquet  — revenue_realized ($)

Usage:
    .venv312/Scripts/python.exe -m src.models.reshape_daily_outputs
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ROOT = Path(__file__).resolve().parents[2]
FC_DIR = ROOT / "forecasts"


def _reshape(src: pd.DataFrame, *,
              actual_col: str | None,
              forecast_cols: tuple[str, str, str],
              has_product: bool = True) -> pd.DataFrame:
    """Unify a daily pipeline parquet into the {Actual, Prediction} schema."""
    src = src.copy()
    # Actual rows
    actual_rows = src[src["data_type"] == "actual"].copy()
    actual_rows["data_type"] = "Actual"
    actual_rows["p10"] = np.nan
    actual_rows["p50"] = np.nan
    actual_rows["p90"] = np.nan
    actual_rows["actual"] = actual_rows[actual_col] if actual_col else np.nan

    # Prediction rows
    pred_rows = src[src["data_type"] == "forecast"].copy()
    pred_rows["data_type"] = "Prediction"
    pred_rows["p10"] = pred_rows[forecast_cols[0]]
    pred_rows["p50"] = pred_rows[forecast_cols[1]]
    pred_rows["p90"] = pred_rows[forecast_cols[2]]
    pred_rows["actual"] = np.nan

    keep = ["date", "data_type", "p10", "p50", "p90", "actual"]
    if has_product:
        keep = ["product_card_id"] + keep
    out = pd.concat([actual_rows[keep], pred_rows[keep]], ignore_index=True)
    return out.sort_values(
        (["product_card_id", "date"] if has_product else ["date"])
    ).reset_index(drop=True)


def main():
    # ---- M1 demand ----------------------------------------------------
    m1 = pd.read_parquet(FC_DIR / "m1_pipeline_daily.parquet")
    m1_out = _reshape(m1,
                        actual_col="actual_gross_qty",
                        forecast_cols=("q10", "q50", "q90"))
    m1_out.to_parquet(FC_DIR / "m1_daily.parquet", index=False)
    log.info("wrote m1_daily.parquet  %s  (Actual=%d, Prediction=%d)",
             m1_out.shape,
             (m1_out["data_type"] == "Actual").sum(),
             (m1_out["data_type"] == "Prediction").sum())

    # ---- M3 disaster / risk drag --------------------------------------
    # M3 is a deterministic signal (one value per (product, day) per scenario),
    # so p10=p50=p90 = disaster_drag_index on Prediction rows. Actual rows
    # carry the historical value in `actual` and NaN quantiles.
    # Keep only the baseline scenario forecast rows to match M1/M4 single-
    # track design (avoids 3x duplicated (product, date) keys with no
    # scenario column to disambiguate).
    m3 = pd.read_parquet(FC_DIR / "m3_pipeline_daily.parquet")
    m3 = m3[(m3["data_type"] == "actual") |
            ((m3["data_type"] == "forecast") & (m3["scenario"] == "baseline"))].copy()
    m3["q10"] = m3["disaster_drag_index"]
    m3["q50"] = m3["disaster_drag_index"]
    m3["q90"] = m3["disaster_drag_index"]
    m3_out = _reshape(m3,
                        actual_col="actual_disaster_index",
                        forecast_cols=("q10", "q50", "q90"))
    m3_out.to_parquet(FC_DIR / "m3_daily.parquet", index=False)
    log.info("wrote m3_daily.parquet  %s  (Actual=%d, Prediction=%d)",
             m3_out.shape,
             (m3_out["data_type"] == "Actual").sum(),
             (m3_out["data_type"] == "Prediction").sum())

    # ---- M4 sales -----------------------------------------------------
    m4 = pd.read_parquet(FC_DIR / "m4_pipeline_daily.parquet")
    m4_out = _reshape(m4,
                        actual_col="actual_revenue_realized",
                        forecast_cols=("q10", "q50", "q90"))
    m4_out.to_parquet(FC_DIR / "m4_daily.parquet", index=False)
    log.info("wrote m4_daily.parquet  %s  (Actual=%d, Prediction=%d)",
             m4_out.shape,
             (m4_out["data_type"] == "Actual").sum(),
             (m4_out["data_type"] == "Prediction").sum())

    # Print summary
    print("\n===== UNIFIED DAILY OUTPUTS =====")
    for name, df in [("M1 (gross_qty)", m1_out),
                       ("M3 (disaster_drag_index)", m3_out),
                       ("M4 (revenue $)", m4_out)]:
        print(f"\n{name}: {df.shape[0]:,} rows, {df['product_card_id'].nunique()} products")
        print(df.head(3).to_string(index=False))
        pred = df[df["data_type"] == "Prediction"]
        if len(pred):
            print(f"  Prediction window: {pred['date'].min().date()} -> {pred['date'].max().date()}")
            print(f"  p50 portfolio sum: {pred['p50'].sum():,.1f}")


if __name__ == "__main__":
    main()
