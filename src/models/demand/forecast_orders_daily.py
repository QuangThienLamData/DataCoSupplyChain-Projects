"""Daily forecast of TOTAL ORDERS (count distinct order_id) at portfolio level.

This is NOT a per-product model: one order can contain multiple products, so
distinct-order count is only meaningful at the company aggregate (matches the
"Total Orders by Month-Year" panel in the dashboard).

Source: raw `DataCoSupplyChainDataset.csv`, count distinct `Order Id` per day.
Model:  TimesFM, pre-storm origin 2017-08-31, 334-day horizon, slice to
        Feb 1 - Jul 31, 2018.

Output schema matches `forecasts/m1_daily.parquet`:

    product_card_id  date  data_type  p10  p50  p90  actual

`product_card_id = 0` is used as a sentinel for the portfolio aggregate
(no real product has card_id 0).

Run:
    .venv312/Scripts/python.exe -m src.models.demand.forecast_orders_daily
"""
from __future__ import annotations

import torch  # noqa: F401

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = ROOT / "data" / "DataCoSupplyChainDataset.csv"
FC_DIR = ROOT / "forecasts"

ORIGIN = pd.Timestamp("2018-01-31")
PRE_STORM_ORIGIN = pd.Timestamp("2017-08-31")
TARGET_DATES = pd.date_range("2018-02-01", "2018-07-31", freq="D")
PRE_STORM_HORIZON = (TARGET_DATES.max() - PRE_STORM_ORIGIN).days  # 334
PORTFOLIO_PID = 0  # sentinel — no real product uses this id


def _daily_distinct_orders() -> pd.Series:
    df = pd.read_csv(RAW_PATH, encoding="latin-1", low_memory=False)
    df["day"] = pd.to_datetime(df["order date (DateOrders)"]).dt.normalize()
    by_day = df.groupby("day")["Order Id"].nunique()
    # Reindex to full daily calendar (zero-fill missing days)
    all_days = pd.date_range(by_day.index.min(), by_day.index.max(), freq="D")
    return by_day.reindex(all_days, fill_value=0).astype(float).rename("total_orders")


def run():
    from src.models.demand.timesfm_model import TimesFMForecaster

    series = _daily_distinct_orders()
    log.info("daily series: %d days (%s..%s), mean=%.1f, max=%d",
             len(series), series.index.min().date(), series.index.max().date(),
             series.mean(), int(series.max()))

    # Pre-storm history slice
    hist_vals = series[series.index <= PRE_STORM_ORIGIN].to_numpy(dtype=float)
    log.info("pre-storm history: %d days through %s",
             len(hist_vals), PRE_STORM_ORIGIN.date())

    # TimesFM single-series forecast
    tfm_max_horizon = (((PRE_STORM_HORIZON + 127) // 128) * 128)
    tfm = TimesFMForecaster(max_context=1024, max_horizon=tfm_max_horizon)
    log.info("TimesFM batched: origin=%s horizon=%d (compiled=%d)",
             PRE_STORM_ORIGIN.date(), PRE_STORM_HORIZON, tfm_max_horizon)
    quantiles = tfm.forecast_batch([hist_vals], PRE_STORM_HORIZON)[0]

    fc_dates = pd.date_range(PRE_STORM_ORIGIN + pd.Timedelta(days=1),
                              periods=PRE_STORM_HORIZON, freq="D")
    fc = pd.DataFrame({
        "product_card_id": PORTFOLIO_PID,
        "date": fc_dates,
        "data_type": "Prediction",
        "p10": quantiles["q10"],
        "p50": quantiles["q50"],
        "p90": quantiles["q90"],
        "actual": np.nan,
    })
    fc = fc[fc["date"].isin(TARGET_DATES)].reset_index(drop=True)

    # Actuals: all observed days through ORIGIN
    actual = pd.DataFrame({
        "product_card_id": PORTFOLIO_PID,
        "date": series.index,
        "data_type": "Actual",
        "p10": np.nan,
        "p50": np.nan,
        "p90": np.nan,
        "actual": series.to_numpy(),
    })
    actual = actual[actual["date"] <= ORIGIN]

    out = pd.concat([actual, fc], ignore_index=True).sort_values("date").reset_index(drop=True)
    dest = FC_DIR / "m1_orders_daily.parquet"
    out.to_parquet(dest, index=False)
    log.info("wrote %s  rows=%d  Actual=%d  Prediction=%d",
             dest.name, len(out),
             (out["data_type"] == "Actual").sum(),
             (out["data_type"] == "Prediction").sum())

    pred = out[out["data_type"] == "Prediction"]
    print(f"\n6-mo total-order forecast: "
          f"p10={pred['p10'].sum():,.0f}  "
          f"p50={pred['p50'].sum():,.0f}  "
          f"p90={pred['p90'].sum():,.0f}  orders")
    print(f"  daily mean p50: {pred['p50'].mean():.1f} orders/day "
          f"(pre-storm mean was {series[series.index <= PRE_STORM_ORIGIN].mean():.1f})")


if __name__ == "__main__":
    run()
