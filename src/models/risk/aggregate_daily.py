"""Daily M3 risk module — produces per-(product, date) p_fraud/p_cancel/p_late.

Daily order counts per product are sparse, so a one-day rate is too noisy.
We smooth with a **30-day trailing average** per product. Where the trailing
window has zero orders, we fall back to the product's lifetime average
(and then to the portfolio average if the product itself never had risk
events).

For the **forward window** (Feb-Jul 2018), the rate is the trailing 90 days
of order-weighted history through the as-of date — constant across the
horizon (assumption: risk regime is stable in the absence of a specific
forward signal).

Output: forecasts/m3_risk_drag_daily.parquet
Schema: product_card_id, date, year_month, p_fraud, p_cancel, p_late
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "daily_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "daily_panel_meta.parquet"
FC_DIR = ROOT / "forecasts"

ORIGIN = pd.Timestamp("2018-01-31")
HORIZON_DAYS = 181
ROLLING_WINDOW = 30


def _smooth_rate(panel: pd.DataFrame, count_col: str, rate_col: str,
                  window: int = ROLLING_WINDOW) -> pd.Series:
    """Order-weighted rolling rate per product.

    rolling_rate = rolling_sum(count_col * rate_col) / rolling_sum(count_col)
    Computed per product. Where the rolling order count is zero, returns NaN
    so the caller can backfill.
    """
    g = panel.sort_values(["product_card_id", "date"]).groupby(
        "product_card_id", group_keys=False)
    weighted = (panel[count_col].fillna(0) * panel[rate_col].fillna(0))
    rolled_num = g.apply(
        lambda s: weighted.loc[s.index].rolling(window, min_periods=1).sum(),
        include_groups=False,
    )
    rolled_den = g.apply(
        lambda s: panel.loc[s.index, count_col].fillna(0)
                       .rolling(window, min_periods=1).sum(),
        include_groups=False,
    )
    rate = rolled_num / rolled_den.replace(0, np.nan)
    return rate


def build_daily_risk() -> pd.DataFrame:
    log.info("===== M3 daily risk rates (rolling-30d smoothed) =====")
    panel = pd.read_parquet(PANEL_PATH)
    panel = panel.sort_values(["product_card_id", "date"]).reset_index(drop=True)

    # Smoothed daily rates over history
    log.info("smoothing fraud/cancel/late with %d-day window", ROLLING_WINDOW)
    panel["p_fraud"] = _smooth_rate(panel, "n_orders_total", "fraud_rate")
    panel["p_cancel"] = _smooth_rate(panel, "n_orders_total", "cancel_rate")
    panel["p_late"] = _smooth_rate(panel, "n_orders_total", "late_rate")

    # Fallback: lifetime average per product, then portfolio average
    portfolio = {
        "p_fraud": float(panel["p_fraud"].mean(skipna=True) or 0.0),
        "p_cancel": float(panel["p_cancel"].mean(skipna=True) or 0.0),
        "p_late": float(panel["p_late"].mean(skipna=True) or 0.0),
    }
    log.info("portfolio mean rates (post-smoothing): %s",
             {k: round(v, 4) for k, v in portfolio.items()})

    def _fill(g: pd.DataFrame) -> pd.DataFrame:
        for col in ("p_fraud", "p_cancel", "p_late"):
            lifetime = float(g[col].mean(skipna=True))
            fallback = lifetime if np.isfinite(lifetime) else portfolio[col]
            g[col] = g[col].fillna(fallback)
        return g

    panel = (panel.groupby("product_card_id", group_keys=False)
                  .apply(_fill, include_groups=False)
                  .reset_index(drop=True))
    # The groupby drop loses the pid; merge back from a copy keyed on row order
    panel_orig = pd.read_parquet(PANEL_PATH).sort_values(
        ["product_card_id", "date"]).reset_index(drop=True)
    panel["product_card_id"] = panel_orig["product_card_id"].values
    panel["date"] = panel_orig["date"].values
    panel["year_month"] = panel_orig["year_month"].values

    hist = panel[panel["date"] <= ORIGIN][
        ["product_card_id", "date", "year_month",
         "p_fraud", "p_cancel", "p_late"]].copy()

    # ---- Forecast rates (trailing 90-day each product, broadcast) -------
    cutoff = ORIGIN - pd.Timedelta(days=90)
    trailing = panel[(panel["date"] > cutoff) & (panel["date"] <= ORIGIN)]
    fwd_rates = (trailing.groupby("product_card_id")
                          [["p_fraud", "p_cancel", "p_late"]].mean()
                          .reset_index())
    fc_dates = pd.date_range(ORIGIN + pd.Timedelta(days=1),
                              ORIGIN + pd.Timedelta(days=HORIZON_DAYS), freq="D")
    grid = pd.MultiIndex.from_product(
        [fwd_rates["product_card_id"], fc_dates],
        names=["product_card_id", "date"]).to_frame(index=False)
    fc = grid.merge(fwd_rates, on="product_card_id", how="left")
    fc["year_month"] = fc["date"].dt.to_period("M").dt.to_timestamp()
    for col in ("p_fraud", "p_cancel", "p_late"):
        fc[col] = fc[col].fillna(portfolio[col])

    out = pd.concat([hist, fc[["product_card_id", "date", "year_month",
                                  "p_fraud", "p_cancel", "p_late"]]],
                      ignore_index=True)
    out = out.sort_values(["product_card_id", "date"]).reset_index(drop=True)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = build_daily_risk()
    out_path = FC_DIR / "m3_risk_drag_daily.parquet"
    df.to_parquet(out_path, index=False)
    log.info("wrote %s  shape=%s", out_path, df.shape)
    log.info("portfolio mean rates: %s",
             df[["p_fraud", "p_cancel", "p_late"]].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
