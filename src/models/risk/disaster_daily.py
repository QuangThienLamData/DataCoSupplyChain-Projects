"""Daily-frequency M3 disaster module.

Builds a per-(product, date) `disaster_index` and `disaster_drag_index`
panel covering the full date range (history + forward window). The
monthly upstream signals are broadcast to daily granularity:

- historical `disaster_index`: `forecasts/m3d_disaster_product.parquet`
  is keyed by (product_card_id, year_month). Every day in month M
  receives the monthly value for that product. (Step-function in time —
  upgrading to a sub-monthly storm profile is future work; this
  preserves total monthly mass.)
- forward `disaster_index`: `build_forward_disaster_for_backtest()` from
  `src/models/sales/forward_forecast.py` produces the Tier-2+3 forward
  monthly signal per product. Same broadcast logic.

The daily drag is computed by convolving with the **daily kernel**
obtained from the monthly kernel via `revenue_lag_daily.monthly_to_daily_kernel`.

Output: `forecasts/m3_disaster_daily.parquet`
Schema: product_card_id, date, year_month, data_type, scenario,
        disaster_index, disaster_drag_index, actual_disaster_index
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.sales.revenue_lag_daily import (
    apply_revenue_lag_daily, REVENUE_LAG_WEIGHTS_MONTHLY,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "daily_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "daily_panel_meta.parquet"
FC_DIR = ROOT / "forecasts"

ORIGIN = pd.Timestamp("2018-01-31")
HORIZON_DAYS = 181  # Feb 1 → Jul 31 2018 (inclusive)

SCENARIOS = ("pessimistic", "baseline", "optimistic")

# Per-scenario monthly kernels (same as the monthly pipeline)
SCENARIO_KERNELS_MONTHLY: dict[str, tuple[float, ...]] = {
    "pessimistic": (0.05, 0.15, 0.40, 0.30, 0.20, 0.10),
    "baseline":    (0.05, 0.20, 0.80),
    "optimistic":  (0.05, 0.10, 0.30),
}


def broadcast_monthly_to_daily(monthly_df: pd.DataFrame,
                                value_col: str,
                                date_index: pd.DatetimeIndex,
                                product_ids: list) -> pd.DataFrame:
    """Expand a (product, year_month) frame to (product, date) by broadcasting
    each monthly value to every day in that month."""
    m = monthly_df[["product_card_id", "year_month", value_col]].copy()
    m["year_month"] = pd.to_datetime(m["year_month"]).dt.to_period("M").dt.to_timestamp()
    # Build daily grid
    grid = pd.MultiIndex.from_product(
        [product_ids, date_index], names=["product_card_id", "date"]
    ).to_frame(index=False)
    grid["year_month"] = grid["date"].dt.to_period("M").dt.to_timestamp()
    out = grid.merge(m, on=["product_card_id", "year_month"], how="left")
    out[value_col] = out[value_col].fillna(0.0)
    return out


def build_daily_disaster() -> pd.DataFrame:
    log.info("===== M3 daily disaster_index + drag =====")
    meta = pd.read_parquet(META_PATH)
    product_ids = sorted(meta["product_card_id"].tolist())

    # Daily date range = panel range + forecast horizon
    panel = pd.read_parquet(PANEL_PATH)
    hist_dates = pd.date_range(panel["date"].min(), ORIGIN, freq="D")
    fc_dates = pd.date_range(ORIGIN + pd.Timedelta(days=1),
                              ORIGIN + pd.Timedelta(days=HORIZON_DAYS), freq="D")
    log.info("history: %s → %s (%d days), forecast: %s → %s (%d days)",
             hist_dates.min().date(), hist_dates.max().date(), len(hist_dates),
             fc_dates.min().date(), fc_dates.max().date(), len(fc_dates))

    # ------- Historical daily disaster_index --------------------------
    # Source: forecasts/m3d_disaster_product.parquet (monthly).
    risk_hist = pd.read_parquet(FC_DIR / "m3d_disaster_product.parquet")[
        ["product_card_id", "year_month", "disaster_index"]]
    hist_daily = broadcast_monthly_to_daily(
        risk_hist, "disaster_index", hist_dates, product_ids)
    hist_daily["data_type"] = "actual"
    hist_daily["scenario"] = None
    hist_daily["actual_disaster_index"] = hist_daily["disaster_index"]

    # Apply baseline daily kernel to history for drag (history kernel choice
    # is monthly baseline; scenarios only differ over the forecast window).
    hist_daily = apply_revenue_lag_daily(
        hist_daily, src_col="disaster_index", dst_col="disaster_drag_index",
        monthly_weights=REVENUE_LAG_WEIGHTS_MONTHLY)

    # ------- Forward daily disaster_index (Tier-2+3) ------------------
    from src.models.sales.forward_forecast import build_forward_disaster_for_backtest
    target_months = pd.date_range("2018-02-01", "2018-07-01", freq="MS")
    forward, _combined = build_forward_disaster_for_backtest(list(target_months))
    forward = forward.rename(columns={"forward_disaster_index": "disaster_index"})[
        ["product_card_id", "year_month", "disaster_index"]]

    rows_out = [hist_daily]

    # ------- Forecast daily per scenario ------------------------------
    # Concatenate history + forecast monthly disaster_index, broadcast to
    # daily over (hist_dates + fc_dates), convolve with scenario kernel,
    # keep only forecast rows.
    full_monthly = pd.concat([risk_hist, forward], ignore_index=True)
    full_dates = pd.date_range(hist_dates.min(), fc_dates.max(), freq="D")

    for scen in SCENARIOS:
        full_daily = broadcast_monthly_to_daily(
            full_monthly, "disaster_index", full_dates, product_ids)
        scen_drag = apply_revenue_lag_daily(
            full_daily, src_col="disaster_index", dst_col="disaster_drag_index",
            monthly_weights=SCENARIO_KERNELS_MONTHLY[scen])
        fc = scen_drag[scen_drag["date"].isin(fc_dates)].copy()
        fc["data_type"] = "forecast"
        fc["scenario"] = scen
        fc["actual_disaster_index"] = np.nan
        rows_out.append(fc)

    out = pd.concat(rows_out, ignore_index=True)
    out = out[["product_card_id", "date", "year_month", "data_type", "scenario",
                "disaster_index", "disaster_drag_index", "actual_disaster_index"]]
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = build_daily_disaster()
    FC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FC_DIR / "m3_disaster_daily.parquet"
    df.to_parquet(out_path, index=False)
    log.info("wrote %s  shape=%s", out_path, df.shape)
    log.info("scenario rows: %s",
             df[df["data_type"] == "forecast"].groupby("scenario").size().to_dict())
    # Sanity: peak day of forecast drag per scenario
    for scen in SCENARIOS:
        sub = df[(df["data_type"] == "forecast") & (df["scenario"] == scen)]
        agg = sub.groupby("date")["disaster_drag_index"].mean()
        if (agg > 0).any():
            log.info("  %s: peak drag date=%s value=%.4f",
                     scen, agg.idxmax().date(), agg.max())
        else:
            log.info("  %s: no drag in forecast window", scen)


if __name__ == "__main__":
    main()
