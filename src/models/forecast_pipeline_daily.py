"""Daily-frequency forward forecast pipeline — Feb-Jul 2018.

Daily analogue of `forecast_pipeline.py`. Produces:

- forecasts/m3_pipeline_daily.parquet  — daily disaster_index + drag (history + 3-scenario forecast)
- forecasts/m1_pipeline_daily.parquet  — daily SARIMAX gross_qty forecast (history + 3 scenarios)
- forecasts/m2_pipeline_daily.parquet  — monthly elasticity broadcast to daily (M2 stays monthly per design)
- forecasts/m4_pipeline_daily.parquet  — daily MC sales revenue (history + 3 scenarios)

The three scenarios cascade end-to-end exactly as in the monthly pipeline:

| Scenario     | Daily revenue-lag kernel (from monthly) | Elasticity β |
|--------------|------------------------------------------|--------------|
| pessimistic  | 180-day (monthly 6-tail)                 | -0.39        |
| baseline     |  90-day (monthly [0.05,0.20,0.80])       | -0.805       |
| optimistic   |  90-day shorter                          | -1.22        |

M1 is **SARIMAX (m=7 weekly)** with each scenario's daily disaster_index as
exogenous regressor.
"""
from __future__ import annotations

# Force torch-first import order for Windows
import torch  # noqa: F401

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "daily_panel.parquet"
FC_DIR = ROOT / "forecasts"

AS_OF = pd.Timestamp("2018-01-31")
ORIGIN = pd.Timestamp("2018-01-31")
HORIZON_DAYS = 181
TARGET_DATES = pd.date_range("2018-02-01", "2018-07-31", freq="D")
SEASON_M = 7

# Pre-storm TimesFM origin — context cuts off before Maria PR landfall
# (2017-09-20). Forecasting from here gives wide, properly-calibrated
# quantile bands; post-Maria origin compressed q10–q90 by ~50x because
# the recent training tail had near-zero variance.
PRE_STORM_ORIGIN = pd.Timestamp("2017-08-31")
# Days from PRE_STORM_ORIGIN through end of TARGET_DATES, inclusive.
PRE_STORM_HORIZON = (TARGET_DATES.max() - PRE_STORM_ORIGIN).days  # = 334

# Single-forecast daily pipeline. Scenarios were dropped at user request —
# the daily pipeline now produces ONE forecast track per parquet (TimesFM
# for M1; baseline kernel for M3). The `scenario` schema column is kept
# (=='baseline' on forecast rows, None on actuals) so downstream consumers
# don't have to special-case daily vs monthly.
SCENARIOS = ("baseline",)

# Single-scenario config. β_recovery=1.0 applies the full data-driven
# recovery lift; set to 0 to disable it. The recovery factor itself is
# still per (product, day) and data-driven:
#
#   recovery_factor[pid, t] =
#       β_recovery
#       × (1 - disaster_drag_index[pid, t])^RECOVERY_SHAPE
#       × historical_recovery_strength[pid]
#
# Forecast adjustment per (product, day):
#   q_adjusted = q_timesfm + recovery_factor · pre_maria_baseline_qty
SCENARIO_CONFIG: dict[str, dict] = {
    "baseline": {
        "kernel_monthly": (0.05, 0.20, 0.80),
        "elasticity": -0.805,
        "beta_recovery": 1.0,
        "description": "TimesFM daily forecast + data-driven recovery lift + pooled-FE β.",
    },
}

# Pre-Maria baseline window: daily mean qty over this stretch is the qty
# floor the recovery factor phases back into the forecast.
PRE_MARIA_WINDOW = (pd.Timestamp("2017-01-01"), pd.Timestamp("2017-08-31"))
# Historical disaster-exposure window (used to derive per-product recovery
# strength from observed storm impact in the recent past).
HIST_EXPOSURE_WINDOW = (pd.Timestamp("2017-08-01"), pd.Timestamp("2018-01-31"))
# Recovery curvature: 1.0 = linear in (1 - drag); >1 = recover more slowly.
RECOVERY_SHAPE = 1.0


def _pre_maria_baseline_qty(panel: pd.DataFrame, product_ids: list) -> pd.Series:
    """Per-product daily mean gross_qty over the pre-Maria window. Falls back
    to lifetime mean (pre-origin) where the window has no activity."""
    pre = panel[(panel["date"] >= PRE_MARIA_WINDOW[0]) &
                (panel["date"] <= PRE_MARIA_WINDOW[1]) &
                (panel["product_card_id"].isin(product_ids))]
    by_pid = pre.groupby("product_card_id")["gross_qty"].mean()
    lifetime = (panel[panel["date"] <= ORIGIN]
                  .groupby("product_card_id")["gross_qty"].mean())
    by_pid = by_pid.reindex(product_ids).fillna(lifetime).fillna(0.0)
    return by_pid.rename("pre_maria_baseline")


def _historical_recovery_strength(m3_daily: pd.DataFrame, product_ids: list) -> pd.Series:
    """Per-product recovery strength derived from historical disaster_index
    exposure during the late-2017 storm window. Returns values in [0, 1]:

      strength[pid] = 1 - mean(disaster_index[pid]) over HIST_EXPOSURE_WINDOW

    A product whose customer-mix saw heavy storm exposure (PR-heavy etc.)
    gets a low strength → smaller recovery lift. A product whose customers
    were unaffected gets a high strength → near-full recovery.
    """
    hist = m3_daily[(m3_daily["data_type"] == "actual")
                     & (m3_daily["date"] >= HIST_EXPOSURE_WINDOW[0])
                     & (m3_daily["date"] <= HIST_EXPOSURE_WINDOW[1])
                     & (m3_daily["product_card_id"].isin(product_ids))]
    exposure = hist.groupby("product_card_id")["disaster_index"].mean()
    strength = (1.0 - exposure).clip(lower=0.0, upper=1.0)
    strength = strength.reindex(product_ids).fillna(1.0)
    return strength.rename("historical_recovery_strength")


# ---------------------------------------------------------------------------
# M3 daily disaster — reuses src.models.risk.disaster_daily
# ---------------------------------------------------------------------------
def run_m3() -> pd.DataFrame:
    """Build/load the daily disaster panel (history + 3 scenarios)."""
    log.info("===== M3 daily — disaster_index + drag, 3 scenarios =====")
    from src.models.risk.disaster_daily import build_daily_disaster
    m3 = build_daily_disaster()
    return m3


# ---------------------------------------------------------------------------
# M1 daily — pre-storm TimesFM context + multiplicative recovery damping
# ---------------------------------------------------------------------------
def run_m1(panel: pd.DataFrame, product_ids: list, m3_daily: pd.DataFrame) -> pd.DataFrame:
    """Daily M1 forecast using **pre-storm TimesFM context** + multiplicative
    recovery damping.

    Why pre-storm context: when TimesFM is fed history through 2018-01-31,
    the last 4 months are 99.7% zero-demand. TimesFM normalises by recent
    mean/std, so its q10–q90 bands collapse to a near-point (q10 ≈ q50 ≈
    q90 for many products). Pre-storm context (cut at 2017-08-31, before
    Maria PR landfall) preserves the normal demand variance and produces
    bands ~50× wider on the same products.

    Pipeline:
      1. TimesFM(context=pre-storm) forecasts 334 days forward from
         2017-08-31, covering Sep 2017 - Jul 2018.
      2. Slice the Feb 1 - Jul 31 2018 portion (`TARGET_DATES`).
      3. Multiplicatively damp by residual storm impact:

             rf[p, t]   = 1 - β_recovery × drag[p, t] × (1 - strength[p])
             q_adj[p,t] = q_pre_storm_TimesFM[p, t] × max(rf[p, t], 0)

         where `strength[p] = 1 - mean(disaster_index)` over the late-2017
         storm window (range 0.66 PR-heavy → 1.0 unaffected). PR-heavy
         products get a meaningful damp; US-only products see rf ≈ 1.

    Output schema matches `m4_pipeline_daily`:
        product_card_id, date, data_type, scenario, q10, q50, q90, actual_gross_qty
    `scenario` is None on actuals and 'baseline' on forecast rows.
    """
    log.info("===== M1 daily — pre-storm TimesFM + multiplicative damp =====")
    from src.models.demand.timesfm_model import (
        TimesFMForecaster, forecast_panel_timesfm_daily,
    )

    rows: list[pd.DataFrame] = []

    # History rows (actuals — scenario=None)
    hist = panel[panel["date"] <= ORIGIN][
        ["product_card_id", "date", "gross_qty"]].copy()
    hist = hist[hist["product_card_id"].isin(product_ids)]
    hist["data_type"] = "actual"
    hist["scenario"] = None
    hist["q10"] = np.nan
    hist["q50"] = np.nan
    hist["q90"] = np.nan
    hist["actual_gross_qty"] = hist["gross_qty"]
    hist = hist.drop(columns="gross_qty")
    rows.append(hist)

    # Per-product historical storm exposure (used by the multiplicative damp)
    hist_strength = _historical_recovery_strength(m3_daily, product_ids)
    log.info("historical recovery strength (1 - mean disaster_index over %s..%s): "
             "mean=%.3f, min=%.3f, max=%.3f",
             HIST_EXPOSURE_WINDOW[0].date(), HIST_EXPOSURE_WINDOW[1].date(),
             hist_strength.mean(), hist_strength.min(), hist_strength.max())

    # Pre-storm TimesFM forecast: 334-day horizon from 2017-08-31. Round
    # up to the nearest 128 patch boundary for TimesFM compilation (384).
    tfm_max_horizon = (((PRE_STORM_HORIZON + 127) // 128) * 128)
    tfm = TimesFMForecaster(max_context=1024, max_horizon=tfm_max_horizon)
    log.info("TimesFM batched: pre-storm origin=%s, horizon=%d (compiled=%d)",
             PRE_STORM_ORIGIN.date(), PRE_STORM_HORIZON, tfm_max_horizon)
    tfm_fc = forecast_panel_timesfm_daily(
        panel, tfm, PRE_STORM_ORIGIN, PRE_STORM_HORIZON, product_ids)
    # Keep only the target Feb-Jul 2018 window
    tfm_fc = tfm_fc[tfm_fc["date"].isin(TARGET_DATES)].copy()
    log.info("sliced to TARGET_DATES (%s..%s): %d rows",
             TARGET_DATES.min().date(), TARGET_DATES.max().date(), len(tfm_fc))

    # Multiplicative recovery damp
    scen = "baseline"
    beta = SCENARIO_CONFIG[scen]["beta_recovery"]
    log.info("recovery damp: β_recovery=%.2f (rf = 1 - β × drag × (1 - strength))",
             beta)
    scen_drag = (m3_daily[(m3_daily["data_type"] == "forecast")
                          & (m3_daily["scenario"] == scen)]
                   .set_index(["product_card_id", "date"])["disaster_drag_index"])

    damped_rows = []
    for pid in product_ids:
        sub = tfm_fc[tfm_fc["product_card_id"] == pid].sort_values("date").copy()
        try:
            drag = scen_drag.loc[pid].reindex(sub["date"]).fillna(0).to_numpy(dtype=float)
        except KeyError:
            drag = np.zeros(len(sub))
        susceptibility = 1.0 - float(hist_strength.loc[pid])
        rf = np.clip(1.0 - beta * drag * susceptibility, 0.0, 1.0)
        sub["q10"] = np.maximum(sub["q10"].to_numpy() * rf, 0.0)
        sub["q50"] = np.maximum(sub["q50"].to_numpy() * rf, 0.0)
        sub["q90"] = np.maximum(sub["q90"].to_numpy() * rf, 0.0)
        damped_rows.append(sub[["product_card_id", "date", "q10", "q50", "q90"]])
    fc = pd.concat(damped_rows, ignore_index=True)
    fc["data_type"] = "forecast"
    fc["scenario"] = scen
    fc["actual_gross_qty"] = np.nan
    rows.append(fc[["product_card_id", "date", "data_type", "scenario",
                      "q10", "q50", "q90", "actual_gross_qty"]])

    m1_full = pd.concat(rows, ignore_index=True)
    m1_full = m1_full[["product_card_id", "date", "data_type", "scenario",
                         "q10", "q50", "q90", "actual_gross_qty"]]
    return m1_full


# ---------------------------------------------------------------------------
# M2 elasticity (stays monthly per design — broadcast to daily for joins)
# ---------------------------------------------------------------------------
def run_m2() -> pd.DataFrame:
    """M2 stays a monthly model — one β per scenario. We emit one row per
    (date, data_type, scenario) by broadcasting the monthly β to every day
    so the daily M4 join is direct.
    """
    log.info("===== M2 (monthly β, broadcast daily for join convenience) =====")
    m2_pool = pd.read_parquet(FC_DIR / "m2_elasticity_pool.parquet").iloc[0].to_dict()
    beta_pool = float(m2_pool["elasticity_own"])
    se_pool = float(m2_pool["elasticity_own_se"])

    panel = pd.read_parquet(PANEL_PATH)
    hist_dates = pd.date_range(panel["date"].min(), ORIGIN, freq="D")

    rows = []
    for d in hist_dates:
        rows.append({
            "date": d,
            "data_type": "actual",
            "scenario": None,
            "elasticity_q10": beta_pool - 1.2816 * se_pool,
            "elasticity_q50": beta_pool,
            "elasticity_q90": beta_pool + 1.2816 * se_pool,
            "source_note": "Pooled-FE estimate from full panel (monthly fit)",
        })
    for scen in SCENARIOS:
        beta_scen = SCENARIO_CONFIG[scen]["elasticity"]
        for d in TARGET_DATES:
            rows.append({
                "date": d,
                "data_type": "forecast",
                "scenario": scen,
                "elasticity_q10": beta_scen - 0.5 * 1.2816 * se_pool,
                "elasticity_q50": beta_scen,
                "elasticity_q90": beta_scen + 0.5 * 1.2816 * se_pool,
                "source_note": SCENARIO_CONFIG[scen]["description"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# M4 daily MC
# ---------------------------------------------------------------------------
def run_m4(panel: pd.DataFrame, product_ids: list,
            m1_full: pd.DataFrame, m3_daily: pd.DataFrame) -> pd.DataFrame:
    log.info("===== M4 daily — Monte Carlo per (product, date), 3 scenarios =====")
    from src.models.sales.run_daily import baseline_price_daily, montecarlo_daily

    rows: list[pd.DataFrame] = []

    # History rows (actual revenue_realized)
    hist = panel[panel["date"] <= ORIGIN][
        ["product_card_id", "date", "revenue_realized"]].copy()
    hist = hist[hist["product_card_id"].isin(product_ids)]
    hist["data_type"] = "actual"
    hist["scenario"] = None
    hist["q10"] = np.nan
    hist["q50"] = np.nan
    hist["q90"] = np.nan
    hist["actual_revenue_realized"] = hist["revenue_realized"]
    hist = hist.drop(columns="revenue_realized")
    rows.append(hist)

    # M3 risk daily (forecast portion has trailing-90d rates)
    m3_risk = pd.read_parquet(FC_DIR / "m3_risk_drag_daily.parquet")
    baseline_price = baseline_price_daily(panel, ORIGIN)

    for scen in SCENARIOS:
        log.info("  scenario=%s", scen)
        m1_scen = m1_full[(m1_full["data_type"] == "forecast")
                           & (m1_full["scenario"] == scen)].copy()
        m1_scen = m1_scen.rename(columns={
            "q10": "q10_demand", "q50": "q50_demand", "q90": "q90_demand"})
        m1_scen["horizon"] = (m1_scen["date"] - ORIGIN).dt.days

        # Disaster forecast (scenario-specific)
        scen_dis = (m3_daily[(m3_daily["data_type"] == "forecast")
                              & (m3_daily["scenario"] == scen)]
                     [["product_card_id", "date",
                       "disaster_index", "disaster_drag_index"]])
        # Forward risk rates (already broadcast across horizon in m3_risk)
        risk_fwd = m3_risk[m3_risk["date"].isin(TARGET_DATES)][
            ["product_card_id", "date", "p_fraud", "p_cancel", "p_late"]]

        inputs = (m1_scen[["product_card_id", "date", "horizon",
                             "q10_demand", "q50_demand", "q90_demand"]]
                    .merge(risk_fwd, on=["product_card_id", "date"], how="left")
                    .merge(scen_dis, on=["product_card_id", "date"], how="left")
                    .merge(baseline_price.reset_index(),
                            on="product_card_id", how="left"))
        for c in ["p_fraud", "p_cancel", "p_late",
                    "disaster_index", "disaster_drag_index", "baseline_price"]:
            inputs[c] = inputs[c].fillna(0.0)

        beta_scen = SCENARIO_CONFIG[scen]["elasticity"]
        fc = montecarlo_daily(inputs, planned_price_factor=1.0,
                                elasticity_mean=beta_scen,
                                elasticity_se=0.0)  # scenario already shifts β
        fc = fc.rename(columns={"sales_q10": "q10",
                                  "sales_q50": "q50",
                                  "sales_q90": "q90"})
        fc["data_type"] = "forecast"
        fc["scenario"] = scen
        fc["actual_revenue_realized"] = np.nan
        rows.append(fc[["product_card_id", "date", "data_type", "scenario",
                          "q10", "q50", "q90", "actual_revenue_realized"]])

    m4_full = pd.concat(rows, ignore_index=True)
    return m4_full


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run():
    log.info("===== DAILY PIPELINE — as-of %s, horizon %d days =====",
             AS_OF.date(), HORIZON_DAYS)
    panel = pd.read_parquet(PANEL_PATH)
    product_ids = sorted(panel["product_card_id"].unique().tolist())
    log.info("forecasting all products: %d", len(product_ids))

    m3 = run_m3()
    m3.to_parquet(FC_DIR / "m3_pipeline_daily.parquet", index=False)
    log.info("wrote m3_pipeline_daily.parquet (%d rows)", len(m3))

    m1 = run_m1(panel, product_ids, m3)
    m1.to_parquet(FC_DIR / "m1_pipeline_daily.parquet", index=False)
    log.info("wrote m1_pipeline_daily.parquet (%d rows)", len(m1))

    m2 = run_m2()
    m2.to_parquet(FC_DIR / "m2_pipeline_daily.parquet", index=False)
    log.info("wrote m2_pipeline_daily.parquet (%d rows)", len(m2))

    m4 = run_m4(panel, product_ids, m1, m3)
    m4.to_parquet(FC_DIR / "m4_pipeline_daily.parquet", index=False)
    log.info("wrote m4_pipeline_daily.parquet (%d rows)", len(m4))

    # Summary
    print("\n===== DAILY PIPELINE SUMMARY =====")
    print("\n--- M1 portfolio q50 daily mean (forecast window) ---")
    s = (m1[m1["data_type"] == "forecast"]
          .groupby("scenario")["q50"].agg(["sum", "mean"]).round(2))
    print(s.to_string())

    print("\n--- M4 portfolio 6-month totals ($) ---")
    for scen in SCENARIOS:
        s = m4[(m4["data_type"] == "forecast") & (m4["scenario"] == scen)]
        print(f"  {scen:12s}: q10=${s['q10'].sum():>14,.0f}  "
              f"q50=${s['q50'].sum():>14,.0f}  q90=${s['q90'].sum():>14,.0f}")


if __name__ == "__main__":
    run()
