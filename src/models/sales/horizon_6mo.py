"""6-month-ahead operating forecast — as-of 2018-01-31, predicting Feb-Jul 2018.

This is the production forward forecast: stand at the last day of the
available data and predict the next 6 months of business operation,
using the same M1 + M2 + M3 + storm-prediction + revenue-lag stack that
serves the val/test backtests.

What's different from the backtest:
- No actuals to compare against — pure forward forecast
- The Tier-2 forward-disaster walk has nothing to find: Atlantic
  hurricane season is June–Nov, and most named-storm activity is Aug-Oct.
  HURDAT2 simulations for Feb-Apr 2018 produce zero hits.
- The Maria recovery tail still feeds the revenue-lag convolution:
  Sep 2017 severity 1.00 → Jan 2018 severity 0.10. At Feb 2018 the
  convolution looks back at Dec/Jan and produces a residual drag.
- Tier-3 seasonal outlook for 2018 is a stub ("Near Normal") because
  NOAA CPC issues the first outlook in May; standing at Jan, we don't
  have one yet.
- M3 risk rates (fraud/cancel/late) extrapolated as the last-3-month
  trailing average per product.

Output: `forecasts/m4_horizon_6mo.parquet`
"""
from __future__ import annotations

# torch-first import to avoid Windows DLL conflicts
import torch  # noqa: F401

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "panel_meta.parquet"
FC_DIR = ROOT / "forecasts"

AS_OF = pd.Timestamp("2018-01-31")
ORIGIN = pd.Timestamp("2018-01-01")   # last available month
HORIZON = 6
TARGET_MONTHS = pd.date_range("2018-02-01", "2018-07-01", freq="MS")


# Self-consistent recovery weights — derived from the M3 disaster-drag
# model rather than a hand-picked curve. The principle:
#
#     recovery_weight[product, t] = 1 - DISASTER_DAMPING × disaster_drag_index[product, t]
#
# - When the drag model says "no Maria residual at month t" (e.g., Apr+),
#   recovery_weight = 1.0 → forecast = pure seasonal-naive (full recovery)
# - When the drag model says "significant residual at month t" (Feb-Mar
#   tail), recovery_weight < 1.0 → blend in TimesFM (continued-crash prior)
#
# Per-product weighting matters: PR-heavy products inherit larger drag,
# so they retain more TimesFM (slower implied recovery) than US-only
# products. The blend is internally consistent with the rest of the
# stack — no extra free parameters.


def _timesfm_forecast(panel: pd.DataFrame, cohort_a: list,
                       horizon: int) -> pd.DataFrame:
    """Pure TimesFM forecast — assumes the recent crash trajectory continues."""
    from src.models.demand.timesfm_model import TimesFMForecaster
    log.info("M1 TimesFM forecast for horizon=%d", horizon)
    p = panel.copy()
    p = p.rename(columns={"qty": "qty_realized"})
    p["qty"] = p["gross_qty"]
    model = TimesFMForecaster(max_context=24, max_horizon=horizon)
    histories = []
    for pid in cohort_a:
        hist = p[(p["product_card_id"] == pid) & (p["year_month"] <= ORIGIN)]
        histories.append(hist.sort_values("year_month")["qty"].to_numpy())
    quantiles = model.forecast_batch(histories, horizon)
    frames = []
    for pid, q in zip(cohort_a, quantiles):
        frames.append(pd.DataFrame({
            "product_card_id": pid,
            "year_month": TARGET_MONTHS[:horizon],
            "horizon": np.arange(1, horizon + 1),
            "q10": q["q10"], "q50": q["q50"], "q90": q["q90"],
        }))
    return pd.concat(frames, ignore_index=True)


def _seasonal_naive_forecast(panel: pd.DataFrame, cohort_a: list,
                              horizon: int) -> pd.DataFrame:
    """Seasonal-naive: y[Feb 2018] = y[Feb 2017]. Assumes full recovery
    to last-year same-month levels — the upper bound of recovery scenarios."""
    log.info("M1 seasonal-naive forecast for horizon=%d", horizon)
    p = panel.copy()
    p = p.rename(columns={"qty": "qty_realized"})
    p["qty"] = p["gross_qty"]
    rows = []
    for pid in cohort_a:
        for h, target in enumerate(TARGET_MONTHS[:horizon], start=1):
            ref_month = target - pd.DateOffset(years=1)
            ref = p[(p["product_card_id"] == pid) & (p["year_month"] == ref_month)]
            q50 = float(ref["qty"].iloc[0]) if not ref.empty else 0.0
            # Uncertainty band: ±30% of q50 (reflecting recovery uncertainty)
            rows.append({
                "product_card_id": pid,
                "year_month": target,
                "horizon": h,
                "q10": max(0.0, q50 * 0.40),
                "q50": q50,
                "q90": q50 * 1.40,
            })
    return pd.DataFrame(rows)


def _blend_forecasts(timesfm: pd.DataFrame, snaive: pd.DataFrame,
                      weights: pd.DataFrame) -> pd.DataFrame:
    """Blend TimesFM and seasonal-naive forecasts using **per (product,
    year_month) recovery weights**:

        forecast = w · snaive + (1 - w) · timesfm

    where `w = 1 - DISASTER_DAMPING × disaster_drag_index` and is supplied
    as a DataFrame with cols [product_card_id, year_month, recovery_weight].

    Products / months not present in `weights` default to w=1.0 (full
    seasonal-naive) since absence of drag info ≈ no storm impact."""
    tfm = timesfm.set_index(["product_card_id", "year_month"]).sort_index()
    snv = snaive.set_index(["product_card_id", "year_month"]).sort_index()
    w_idx = weights.set_index(["product_card_id", "year_month"])["recovery_weight"]

    out = snv.copy()
    for col in ("q10", "q50", "q90"):
        snv_col = snv[col].fillna(0)
        tfm_col = tfm[col].reindex(snv.index).fillna(0)
        w_col = w_idx.reindex(snv.index).fillna(1.0).clip(0.0, 1.0)
        out[col] = w_col * snv_col + (1 - w_col) * tfm_col
    out = out.reset_index()
    out["recovery_weight"] = w_idx.reindex(snv.index).fillna(1.0).clip(0.0, 1.0).to_numpy()
    return out[["product_card_id", "year_month", "horizon",
                "q10", "q50", "q90", "recovery_weight"]]


def derive_recovery_weights(
    risk_drag_target: pd.DataFrame, damping: float,
) -> pd.DataFrame:
    """Compute per (product, year_month) recovery_weight from the
    `disaster_drag_index` already in the risk frame.

        recovery_weight = 1 - DISASTER_DAMPING × disaster_drag_index

    Self-consistent with M4's drag math: when M4 says "Feb 2018 has 7%
    drag from Maria's tail", we use 93% seasonal-naive + 7% TimesFM for
    that product's baseline M1 forecast.
    """
    if "disaster_drag_index" not in risk_drag_target.columns:
        raise KeyError("risk_drag_target needs 'disaster_drag_index' column")
    df = risk_drag_target[["product_card_id", "year_month",
                            "disaster_drag_index"]].copy()
    df["recovery_weight"] = (1.0 - damping * df["disaster_drag_index"]).clip(0.0, 1.0)
    return df


def m1_forecast_horizon(panel: pd.DataFrame, cohort_a: list,
                         horizon: int = HORIZON,
                         scenario: str = "baseline",
                         recovery_weights: pd.DataFrame | None = None) -> pd.DataFrame:
    """M1 forecast with recovery-aware scenarios.

    Args:
        scenario:
          - "pessimistic": pure TimesFM (assumes crash continues)
          - "baseline":    blend TimesFM + seasonal-naive using
                           **per-(product, month) recovery weights**
                           derived from `disaster_drag_index` (Option A,
                           self-consistent with M4's drag model)
          - "optimistic":  pure seasonal-naive (assumes full recovery to
                           prior-year same-month levels)
        recovery_weights: per (product, year_month) recovery_weight df.
                          Required when scenario='baseline'.

    Returns DataFrame with `model = 'timesfm'`, `slice = 'future'`, and
    the chosen scenario tagged in a `scenario` column.
    """
    tfm = _timesfm_forecast(panel, cohort_a, horizon)
    snv = _seasonal_naive_forecast(panel, cohort_a, horizon)
    if scenario == "pessimistic":
        out = tfm.copy()
    elif scenario == "optimistic":
        out = snv.copy()
    elif scenario == "baseline":
        if recovery_weights is None:
            raise ValueError("baseline scenario requires recovery_weights "
                              "(derived from disaster_drag_index — see "
                              "derive_recovery_weights)")
        out = _blend_forecasts(tfm, snv, recovery_weights)
    else:
        raise ValueError(f"unknown scenario {scenario}")
    out["model"] = "timesfm"
    out["slice"] = "future"
    out["scenario"] = scenario
    return out


def estimate_forward_risk_rates(
    panel: pd.DataFrame, risk_drag_hist: pd.DataFrame,
    target_months: list, lookback_months: int = 3,
) -> pd.DataFrame:
    """For each product, extrapolate fraud/cancel/late rates as the
    last-N-month trailing mean (computed up to ORIGIN). Returns a frame
    keyed by (product_card_id, year_month) for target months."""
    log.info("estimating forward risk rates (trailing %d months)", lookback_months)
    recent = risk_drag_hist[
        risk_drag_hist["year_month"].between(
            ORIGIN - pd.DateOffset(months=lookback_months - 1), ORIGIN)
    ]
    avg = (recent.groupby("product_card_id")
                  [["p_fraud", "p_cancel", "p_late"]].mean()
                  .reset_index())
    rows = []
    for pid, sub in avg.groupby("product_card_id"):
        for tm in target_months:
            r = sub.iloc[0]
            rows.append({
                "product_card_id": float(pid),
                "year_month": pd.Timestamp(tm),
                "p_fraud": float(r["p_fraud"]),
                "p_cancel": float(r["p_cancel"]),
                "p_late": float(r["p_late"]),
            })
    return pd.DataFrame(rows)


def build_horizon_disaster(target_months: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tier-2 simulation + Tier-3 outlook for the forward target months,
    then the revenue-lag convolution.

    Returns (per_product_severity, per_product_drag).
    """
    from src.models.sales.forward_forecast import build_forward_disaster_for_backtest
    log.info("building Tier-2+3 forward disaster for Feb-Jul 2018")
    forward_product, combined = build_forward_disaster_for_backtest(target_months)
    return forward_product, combined


def baseline_price_horizon(panel: pd.DataFrame, cohort_a: list) -> pd.Series:
    """Trailing 3-month effective-price average per product, as of ORIGIN."""
    p = panel[(panel["product_card_id"].isin(cohort_a))
               & (panel["year_month"] <= ORIGIN)
               & (panel["p_eff"].notna())].sort_values(["product_card_id", "year_month"])
    return (p.groupby("product_card_id")["p_eff"]
              .apply(lambda s: s.tail(3).mean())
              .rename("baseline_price"))


def run() -> dict:
    log.info("===== M4 HORIZON FORECAST — as-of %s, target=%s =====",
             AS_OF.date(), [m.date() for m in TARGET_MONTHS])

    panel = pd.read_parquet(PANEL_PATH)
    meta = pd.read_parquet(META_PATH)
    cohort_a = meta.loc[meta["cohort"] == "A_active", "product_card_id"].tolist()
    log.info("cohort A: %d products", len(cohort_a))

    # ------------------------------------------------------------------
    # Step 1: Build forward disaster + risk_drag FIRST (so we can derive
    # self-consistent recovery weights from disaster_drag_index)
    # ------------------------------------------------------------------
    forward_product, combined = build_horizon_disaster(list(TARGET_MONTHS))
    forward_product = forward_product.rename(
        columns={"forward_disaster_index": "disaster_index_forward"})

    risk_hist = pd.read_parquet(FC_DIR / "m3_risk_drag.parquet")
    risk_hist = risk_hist[["product_card_id", "year_month",
                            "p_fraud", "p_cancel", "p_late", "disaster_index"]]

    risk_future = estimate_forward_risk_rates(panel, risk_hist, list(TARGET_MONTHS))
    risk_future = risk_future.merge(forward_product[
        ["product_card_id", "year_month", "disaster_index_forward"]],
        on=["product_card_id", "year_month"], how="left")
    risk_future["disaster_index"] = risk_future["disaster_index_forward"].fillna(0.0)
    risk_future = risk_future[["product_card_id", "year_month",
                                "p_fraud", "p_cancel", "p_late", "disaster_index"]]

    # Stack historical + future, convolve to get disaster_drag_index
    from src.models.sales.revenue_lag import apply_revenue_lag_per_product
    risk_all = pd.concat([risk_hist, risk_future], ignore_index=True)
    risk_all = apply_revenue_lag_per_product(
        risk_all, src_col="disaster_index", dst_col="disaster_drag_index")
    risk_for_target = risk_all[risk_all["year_month"].isin(TARGET_MONTHS)].copy()

    # ------------------------------------------------------------------
    # Step 2: Derive self-consistent recovery weights from disaster_drag_index
    # ------------------------------------------------------------------
    from src.models.sales.calibrate_damping import load_calibration
    LATE_DAMPING, DISASTER_DAMPING = load_calibration()
    log.info("loaded DISASTER_DAMPING=%.3f, LATE_DAMPING=%.3f",
             DISASTER_DAMPING, LATE_DAMPING)
    recovery_weights = derive_recovery_weights(risk_for_target, DISASTER_DAMPING)
    log.info("recovery_weights derived from disaster_drag_index — "
             "Feb mean=%.3f, Jul mean=%.3f",
             recovery_weights[recovery_weights["year_month"]==TARGET_MONTHS[0]]
                ["recovery_weight"].mean(),
             recovery_weights[recovery_weights["year_month"]==TARGET_MONTHS[-1]]
                ["recovery_weight"].mean())

    # ------------------------------------------------------------------
    # Step 3: Build M1 scenarios using the data-driven weights
    # ------------------------------------------------------------------
    log.info("===== building M1 scenarios: pessimistic / baseline / optimistic =====")
    m1_pess = m1_forecast_horizon(panel, cohort_a, scenario="pessimistic")
    m1_base = m1_forecast_horizon(panel, cohort_a, scenario="baseline",
                                    recovery_weights=recovery_weights)
    m1_opt  = m1_forecast_horizon(panel, cohort_a, scenario="optimistic")
    m1_all = pd.concat([m1_pess, m1_base, m1_opt], ignore_index=True)
    log.info("M1 future rows: %d (3 scenarios × %d products × %d months)",
             len(m1_all), len(cohort_a), HORIZON)

    # Save the derived recovery weights for transparency
    recovery_weights.to_parquet(FC_DIR / "m4_horizon_recovery_weights.parquet",
                                 index=False)

    m1_future = m1_base   # default risk frame uses baseline scenario

    # M4 assemble inputs for target months
    from src.models.sales.forecast import (
        assemble_inputs, forecast_frame,
    )
    baseline_price = baseline_price_horizon(panel, cohort_a)
    inputs = assemble_inputs(m1_future, risk_for_target, baseline_price,
                              slice_name="future")
    log.info("M4 inputs: %d rows", len(inputs))

    # Load M2 pooled elasticity
    m2_pool = pd.read_parquet(FC_DIR / "m2_elasticity_pool.parquet").iloc[0].to_dict()
    eps_mean = float(m2_pool["elasticity_own"])
    eps_se = float(m2_pool["elasticity_own_se"])
    log.info("M2 elasticity: %.3f ± %.3f", eps_mean, eps_se)

    # Run Monte Carlo for all 3 scenarios — same risk inputs, different M1.
    m4_scenarios = []
    for scen, m1_scen in [("pessimistic", m1_pess), ("baseline", m1_base),
                            ("optimistic", m1_opt)]:
        inputs_s = assemble_inputs(m1_scen, risk_for_target, baseline_price,
                                     slice_name="future")
        m4_s = forecast_frame(inputs_s, elasticity_mean=eps_mean,
                                elasticity_se=eps_se)
        m4_s["slice"] = "future"
        m4_s["scenario"] = scen
        m4_scenarios.append(m4_s)
        total = m4_s.groupby("year_month")["sales_q50"].sum()
        log.info("scenario=%s totals: %s", scen,
                 {str(k.date()): f"${v:,.0f}" for k, v in total.items()})
    m4 = pd.concat(m4_scenarios, ignore_index=True)

    # Save artifacts
    out_path = FC_DIR / "m4_horizon_6mo.parquet"
    m4.to_parquet(out_path, index=False)
    log.info("wrote %s (%d rows = 3 scenarios)", out_path, len(m4))
    m1_all.to_parquet(FC_DIR / "m1_horizon_6mo.parquet", index=False)
    forward_product.to_parquet(FC_DIR / "m3d_horizon_6mo_disaster.parquet", index=False)
    risk_for_target.to_parquet(FC_DIR / "m3_horizon_6mo_risk.parquet", index=False)

    # Portfolio summary per scenario
    print("\n===== M4 PORTFOLIO FORECAST — Feb-Jul 2018 by scenario =====")
    for scen in ("pessimistic", "baseline", "optimistic"):
        sub = m4[m4["scenario"] == scen]
        by_month = (sub.groupby("year_month")
                       .agg(q50=("sales_q50", "sum"),
                            q10=("sales_q10", "sum"),
                            q90=("sales_q90", "sum"))
                       .round(0))
        total = by_month["q50"].sum()
        print(f"\n--- {scen.upper()} (6-mo total q50 = ${total:,.0f}) ---")
        print(by_month.to_string())

    print("\n===== TOP 5 PRODUCTS BY 6-MO TOTAL FORECAST (baseline scenario) =====")
    base = m4[m4["scenario"] == "baseline"]
    top = (base.groupby("product_card_id")["sales_q50"].sum()
                .sort_values(ascending=False).head(5))
    for pid in top.index:
        name = meta.loc[meta["product_card_id"] == pid, "product_name"].iloc[0]
        print(f"  {int(pid)} {name[:40]:40s}  6-mo total q50 = ${top[pid]:>12,.0f}")

    print("\n===== DISASTER TAIL — Maria recovery into Feb-Jul 2018 =====")
    tail = (risk_for_target.groupby("year_month")
                            .agg(disaster_index=("disaster_index", "mean")))
    drag = (risk_all[risk_all["year_month"].isin(TARGET_MONTHS)]
              .groupby("year_month")
              .agg(disaster_drag_index=("disaster_drag_index", "mean")))
    print(tail.merge(drag, on="year_month").round(3).to_string())

    return {
        "m4": m4, "m1": m1_future, "forward_disaster": forward_product,
        "risk": risk_for_target, "by_month": by_month, "top": top,
    }


if __name__ == "__main__":
    run()
