"""Daily total-sales forecast at portfolio level.

  Total Sales(d) = AOV(d) × Number_of_Orders(d)

where:
  - AOV (Average Order Value) is forecast here from the raw daily series
    using TimesFM with the pre-storm origin (2017-08-31).
  - Number_of_Orders is read from `forecasts/m1_orders_daily.parquet`
    (produced by `forecast_orders_daily.py`).

Output schema matches `forecasts/m1_daily.parquet`:

    product_card_id  date  data_type  p10  p50  p90  actual

`product_card_id = 0` is a sentinel for the portfolio aggregate.

Run:
    .venv312/Scripts/python.exe -m src.models.sales.forecast_total_sales_daily
"""
from __future__ import annotations

import torch  # noqa: F401   force torch-first import on Windows

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
PORTFOLIO_PID = 0


def _daily_aov_and_orders() -> pd.DataFrame:
    """Per-day AOV (= total revenue / distinct orders) and order count from raw."""
    df = pd.read_csv(RAW_PATH, encoding="latin-1", low_memory=False)
    df["day"] = pd.to_datetime(df["order date (DateOrders)"]).dt.normalize()
    by_order = df.groupby(["day", "Order Id"])["Sales"].sum().reset_index()
    by_day = by_order.groupby("day").agg(
        total_rev=("Sales", "sum"),
        n_orders=("Order Id", "size"),
    )
    by_day["aov"] = by_day["total_rev"] / by_day["n_orders"]
    all_days = pd.date_range(by_day.index.min(), by_day.index.max(), freq="D")
    return by_day.reindex(all_days).fillna(0.0).rename_axis("date").reset_index()


def run():
    from src.models.demand.timesfm_model import TimesFMForecaster

    series = _daily_aov_and_orders()
    log.info("daily series: %d days (%s..%s)", len(series),
             series["date"].min().date(), series["date"].max().date())
    log.info("AOV   pre-storm mean = %.2f", series.loc[series["date"] <= PRE_STORM_ORIGIN, "aov"].mean())
    log.info("Orders pre-storm mean = %.1f", series.loc[series["date"] <= PRE_STORM_ORIGIN, "n_orders"].mean())

    # 1) Forecast AOV with TimesFM (pre-storm context to preserve variance)
    hist_aov = series.loc[series["date"] <= PRE_STORM_ORIGIN, "aov"].to_numpy(dtype=float)
    tfm_max_horizon = (((PRE_STORM_HORIZON + 127) // 128) * 128)
    tfm = TimesFMForecaster(max_context=1024, max_horizon=tfm_max_horizon)
    log.info("TimesFM AOV: pre-storm origin=%s horizon=%d",
             PRE_STORM_ORIGIN.date(), PRE_STORM_HORIZON)
    aov_q = tfm.forecast_batch([hist_aov], PRE_STORM_HORIZON)[0]

    fc_dates = pd.date_range(PRE_STORM_ORIGIN + pd.Timedelta(days=1),
                              periods=PRE_STORM_HORIZON, freq="D")
    aov_fc = pd.DataFrame({
        "date": fc_dates,
        "aov_p10": aov_q["q10"],
        "aov_p50": aov_q["q50"],
        "aov_p90": aov_q["q90"],
    })
    aov_fc = aov_fc[aov_fc["date"].isin(TARGET_DATES)].reset_index(drop=True)

    # 2) Read num_orders forecast (produced by forecast_orders_daily.py)
    orders = pd.read_parquet(FC_DIR / "m1_orders_daily.parquet")
    orders = orders[(orders["data_type"] == "Prediction")][
        ["date", "p10", "p50", "p90"]].rename(
        columns={"p10": "ord_p10", "p50": "ord_p50", "p90": "ord_p90"})
    log.info("loaded num_orders forecast: %d rows", len(orders))

    # 3) Combine: total_sales(d) = AOV(d) × num_orders(d)
    fc = aov_fc.merge(orders, on="date", how="inner")
    fc["p10"] = fc["aov_p10"] * fc["ord_p10"]
    fc["p50"] = fc["aov_p50"] * fc["ord_p50"]
    fc["p90"] = fc["aov_p90"] * fc["ord_p90"]
    fc["product_card_id"] = PORTFOLIO_PID
    fc["data_type"] = "Prediction"
    fc["actual"] = np.nan
    fc = fc[["product_card_id", "date", "data_type", "p10", "p50", "p90", "actual"]]

    # 4) Actuals = total_rev per day (already in `series`)
    actual = pd.DataFrame({
        "product_card_id": PORTFOLIO_PID,
        "date": series["date"],
        "data_type": "Actual",
        "p10": np.nan, "p50": np.nan, "p90": np.nan,
        "actual": series["total_rev"].to_numpy(),
    })
    actual = actual[actual["date"] <= ORIGIN]

    out = pd.concat([actual, fc], ignore_index=True).sort_values("date").reset_index(drop=True)
    dest = FC_DIR / "m4_total_sales_daily.parquet"
    out.to_parquet(dest, index=False)
    log.info("wrote %s  rows=%d  Actual=%d  Prediction=%d",
             dest.name, len(out),
             (out["data_type"] == "Actual").sum(),
             (out["data_type"] == "Prediction").sum())

    pred = out[out["data_type"] == "Prediction"]
    print(f"\n6-month total-sales forecast (Feb-Jul 2018):")
    print(f"  p10 = ${pred['p10'].sum():>12,.0f}")
    print(f"  p50 = ${pred['p50'].sum():>12,.0f}   <- point forecast")
    print(f"  p90 = ${pred['p90'].sum():>12,.0f}")
    print(f"\n  daily mean p50: ${pred['p50'].mean():,.0f}")
    print(f"  pre-storm daily mean: ${series.loc[series['date']<=PRE_STORM_ORIGIN,'total_rev'].mean():,.0f}")


if __name__ == "__main__":
    run()
