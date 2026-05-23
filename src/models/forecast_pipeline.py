"""Production forecast pipeline — 6-month horizon, full M3→M1→M2→M4 chain
with 3 scenarios at every stage.

Flow
----
    M3 (risk/disaster)  →  M1 (demand)  →  M2 (elasticity)  →  M4 (sales)

Each scenario carries scenario-specific parameters end-to-end. The
parameters were chosen to express coherent business cases:

| Scenario     | Storm tail (revenue lag) | Elasticity beta | Reading |
|--------------|--------------------------|--------------|---------|
| pessimistic  | 6-month kernel (Maria-like prolonged) | −0.39 (less price-responsive) | "slow recovery, less discount-driven growth" |
| baseline     | 3-month kernel (calibrated)           | −0.805 (calibrated)            | "model-consistent default" |
| optimistic   | 3-month shorter kernel                | −1.22 (more price-responsive)  | "fast recovery, customers grow with discounts" |

Pessimistic and optimistic beta = baseline ± 1·SE of the pooled-FE estimate.

Output files
------------
Each parquet contains BOTH historical actuals AND forward forecasts,
distinguished by `data_type ∈ {'actual', 'forecast'}`. Forecast rows
carry a `scenario ∈ {'pessimistic', 'baseline', 'optimistic'}`; history
rows have `scenario = None`. Quantile columns `q10 / q50 / q90` carry
the forecast bands; for history rows the actual value is in the
corresponding `actual_*` column.

- `forecasts/m3_pipeline.parquet`  — disaster_index + disaster_drag_index
- `forecasts/m1_pipeline.parquet`  — gross_qty demand forecast
- `forecasts/m2_pipeline.parquet`  — pooled elasticity beta
- `forecasts/m4_pipeline.parquet`  — sales revenue forecast
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

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "panel_meta.parquet"
FC_DIR = ROOT / "forecasts"

AS_OF = pd.Timestamp("2018-01-31")
ORIGIN = pd.Timestamp("2018-01-01")
HORIZON = 6
TARGET_MONTHS = pd.date_range("2018-02-01", "2018-07-01", freq="MS")

SCENARIOS = ("pessimistic", "baseline", "optimistic")

# Per-scenario configuration — coherent business priors
SCENARIO_CONFIG: dict[str, dict] = {
    "pessimistic": {
        # 6-month revenue-impact tail — Maria-like prolonged recovery
        "kernel": (0.05, 0.15, 0.40, 0.30, 0.20, 0.10),
        # Less price-responsive demand (closer to inelastic; less ability to
        # grow with discounts). beta + 1·SE of pooled estimate.
        "elasticity": -0.39,
        "description": (
            "Storm recovery is slow (6-month drag tail); customers are less "
            "price-responsive (beta=-0.39). Slowest recovery, smallest growth "
            "from price moves."
        ),
    },
    "baseline": {
        # 3-month kernel (currently calibrated default)
        "kernel": (0.05, 0.20, 0.80),
        # Pooled-FE beta from M2
        "elasticity": -0.805,
        "description": (
            "Calibrated defaults — 3-month drag kernel, pooled-FE elasticity. "
            "Internally consistent with M3 DAMPING calibration."
        ),
    },
    "optimistic": {
        # Shorter, milder tail
        "kernel": (0.05, 0.10, 0.30),
        # More price-responsive (more elastic). beta - 1·SE.
        "elasticity": -1.22,
        "description": (
            "Storm recovery is fast (impact decays in ~2 months); customers "
            "are price-responsive (beta=-1.22). Fastest recovery, largest "
            "growth opportunity from price moves."
        ),
    },
}


# ---------------------------------------------------------------------------
# M3 disaster: per-scenario disaster_index + drag
# ---------------------------------------------------------------------------
def _convolve_per_product(df: pd.DataFrame, src_col: str, dst_col: str,
                            kernel: tuple[float, ...]) -> pd.DataFrame:
    """Per-product convolution of src_col with kernel weights."""
    w = np.asarray(kernel, dtype=float)
    df = df.sort_values(["product_card_id", "year_month"]).reset_index(drop=True).copy()
    out = np.zeros(len(df), dtype=float)
    for pid, idx in df.groupby("product_card_id", sort=False).indices.items():
        d = df.loc[idx, src_col].to_numpy(dtype=float)
        c = np.zeros_like(d)
        for i in range(len(d)):
            for k, weight in enumerate(w):
                if i - k >= 0:
                    c[i] += weight * d[i - k]
        out[idx] = c
    df[dst_col] = np.clip(out, 0.0, 1.0)
    return df


def run_m3() -> pd.DataFrame:
    """Build M3 history + 3-scenario forecast. Returns long-form DataFrame:
        product_card_id, year_month, data_type, scenario,
        disaster_index, disaster_drag_index, actual_disaster_index
    """
    log.info("===== M3 (disaster + revenue-lag drag) — 3 scenarios =====")

    # Historical disaster_index (same across scenarios — fact)
    risk_hist = pd.read_parquet(FC_DIR / "m3_risk_drag.parquet")[
        ["product_card_id", "year_month", "disaster_index"]]

    # Forward disaster_index (Tier-2 + Tier-3, same across scenarios)
    from src.models.sales.forward_forecast import build_forward_disaster_for_backtest
    forward, _combined = build_forward_disaster_for_backtest(list(TARGET_MONTHS))
    forward = forward.rename(columns={"forward_disaster_index": "disaster_index"})[
        ["product_card_id", "year_month", "disaster_index"]]

    rows_out: list[pd.DataFrame] = []

    # History rows — same for every scenario; pick one representative scenario tag
    # (we'll de-duplicate later). Actually we want history rows to be
    # scenario=None (no scenario applies to actuals).
    # But disaster_drag_index requires convolution which needs history continuity.
    # Approach: for history rows, compute disaster_drag_index using the
    # BASELINE kernel (since history values are facts; the kernel choice
    # only affects forecast period).

    # Step 1: history rows with baseline kernel applied for drag
    hist_with_drag = _convolve_per_product(
        risk_hist.copy(), "disaster_index", "disaster_drag_index",
        SCENARIO_CONFIG["baseline"]["kernel"])
    hist_with_drag["data_type"] = "actual"
    hist_with_drag["scenario"] = None
    hist_with_drag["actual_disaster_index"] = hist_with_drag["disaster_index"]
    rows_out.append(hist_with_drag)

    # Step 2: forecast rows per scenario (different kernels)
    for scen in SCENARIOS:
        kernel = SCENARIO_CONFIG[scen]["kernel"]
        # Concatenate history + forecast for convolution context
        full = pd.concat([risk_hist, forward], ignore_index=True)
        full = _convolve_per_product(full, "disaster_index", "disaster_drag_index", kernel)
        # Keep only forecast rows
        fc = full[full["year_month"].isin(TARGET_MONTHS)].copy()
        fc["data_type"] = "forecast"
        fc["scenario"] = scen
        fc["actual_disaster_index"] = np.nan
        rows_out.append(fc)

    m3_full = pd.concat(rows_out, ignore_index=True)
    m3_full = m3_full[["product_card_id", "year_month", "data_type", "scenario",
                         "disaster_index", "disaster_drag_index",
                         "actual_disaster_index"]]
    return m3_full


# ---------------------------------------------------------------------------
# M1 demand: disaster-aware forecast using SARIMAX (with disaster_index exog)
# ---------------------------------------------------------------------------
def _sarimax_forecast_per_product(
    panel: pd.DataFrame, cohort_a: list,
    exog_history_by_pid: dict, exog_future_by_pid: dict,
) -> pd.DataFrame:
    """Run SARIMAX with disaster_index exog for each product. The exog
    series MUST be product-specific (different products see different
    disaster impact via their customer-country mix)."""
    from src.models.demand.baselines import SARIMAXForecaster, SARIMAForecaster
    sarimax = SARIMAXForecaster()
    sarima_fb = SARIMAForecaster()
    rows = []
    for pid in cohort_a:
        hist = panel[(panel["product_card_id"] == pid)
                      & (panel["year_month"] <= ORIGIN)].sort_values("year_month")
        y = hist["gross_qty"].fillna(0).to_numpy(dtype=float)
        ex_hist = exog_history_by_pid.get(pid)
        ex_fut = exog_future_by_pid.get(pid)
        if ex_hist is not None and ex_fut is not None:
            q = sarimax.forecast(y, HORIZON, exog_history=ex_hist, exog_future=ex_fut)
        else:
            q = sarima_fb.forecast(y, HORIZON)
        rows.append(pd.DataFrame({
            "product_card_id": pid,
            "year_month": TARGET_MONTHS,
            "q10": q["q10"], "q50": q["q50"], "q90": q["q90"],
            "horizon": np.arange(1, HORIZON + 1),
        }))
    return pd.concat(rows, ignore_index=True)


def run_m1(panel: pd.DataFrame, cohort_a: list, m3_full: pd.DataFrame) -> pd.DataFrame:
    """Build M1 history + 3-scenario disaster-aware demand forecast.

    The production demand model in the pipeline is **SARIMAX with
    disaster_index as exogenous regressor** — the model natively learns
    how demand responds to storm severity. Each scenario uses its own
    disaster_index series (from M3) so the demand forecast varies with
    the storm assumption.

    Schema: product_card_id, year_month, data_type, scenario,
            q10, q50, q90, actual_gross_qty
    """
    log.info("===== M1 (demand) — disaster-aware SARIMAX, 3 scenarios =====")

    rows_out: list[pd.DataFrame] = []

    # History rows (actuals — no scenario applies)
    hist = panel[panel["year_month"] <= ORIGIN][
        ["product_card_id", "year_month", "gross_qty"]].copy()
    hist = hist[hist["product_card_id"].isin(cohort_a)]
    hist["data_type"] = "actual"
    hist["scenario"] = None
    hist["q10"] = np.nan
    hist["q50"] = np.nan
    hist["q90"] = np.nan
    hist["actual_gross_qty"] = hist["gross_qty"]
    hist = hist.drop(columns="gross_qty")
    rows_out.append(hist)

    # Build exog_history_by_pid (same across scenarios — historical data
    # is fact; scenarios only diverge for the forecast window).
    hist_disaster = (m3_full[m3_full["data_type"] == "actual"]
                       .set_index(["product_card_id", "year_month"])["disaster_index"])

    exog_history_by_pid = {}
    for pid in cohort_a:
        try:
            series = hist_disaster.loc[pid].sort_index().to_numpy(dtype=float)
            exog_history_by_pid[pid] = series
        except KeyError:
            exog_history_by_pid[pid] = None

    # Per-scenario forecasts: each scenario's disaster_index for the
    # forecast horizon is its own exog future series.
    for scen in SCENARIOS:
        scen_fc = (m3_full[(m3_full["data_type"] == "forecast")
                            & (m3_full["scenario"] == scen)]
                     .set_index(["product_card_id", "year_month"])["disaster_index"])
        exog_future_by_pid = {}
        for pid in cohort_a:
            try:
                series = scen_fc.loc[pid].sort_index().to_numpy(dtype=float)
                exog_future_by_pid[pid] = series
            except KeyError:
                exog_future_by_pid[pid] = None

        log.info("  scenario=%s: running SARIMAX over %d products",
                 scen, len(cohort_a))
        fc = _sarimax_forecast_per_product(
            panel, cohort_a, exog_history_by_pid, exog_future_by_pid)
        fc["data_type"] = "forecast"
        fc["scenario"] = scen
        fc["actual_gross_qty"] = np.nan
        rows_out.append(fc[["product_card_id", "year_month", "data_type",
                              "scenario", "q10", "q50", "q90",
                              "actual_gross_qty"]])

    m1_full = pd.concat(rows_out, ignore_index=True)
    m1_full = m1_full[["product_card_id", "year_month", "data_type", "scenario",
                         "q10", "q50", "q90", "actual_gross_qty"]]
    return m1_full


# ---------------------------------------------------------------------------
# M2 elasticity: 3 scenario beta values
# ---------------------------------------------------------------------------
def run_m2() -> pd.DataFrame:
    """M2 has no time-series — just a single pooled beta per scenario.
    For convenience we output a row per (scenario, year_month) in the
    historical + forecast window so joins remain easy.

    Schema:
        year_month, data_type, scenario, elasticity_q10, elasticity_q50,
        elasticity_q90, source_note
    """
    log.info("===== M2 (price elasticity) — 3 scenarios =====")
    m2_pool = pd.read_parquet(FC_DIR / "m2_elasticity_pool.parquet").iloc[0].to_dict()
    beta_pool = float(m2_pool["elasticity_own"])
    se_pool = float(m2_pool["elasticity_own_se"])
    log.info("pooled beta = %.3f ± %.3f (SE)", beta_pool, se_pool)

    panel_history = pd.read_parquet(PANEL_PATH)
    hist_months = sorted(panel_history["year_month"].unique())

    rows_out: list[dict] = []

    # History rows — single estimate, no scenario
    for ym in hist_months:
        rows_out.append({
            "year_month": pd.Timestamp(ym),
            "data_type": "actual",
            "scenario": None,
            "elasticity_q10": beta_pool - 1.2816 * se_pool,
            "elasticity_q50": beta_pool,
            "elasticity_q90": beta_pool + 1.2816 * se_pool,
            "source_note": "Pooled-FE estimate from full panel",
        })

    # Forecast rows — 3 scenarios, beta varies per scenario
    for scen in SCENARIOS:
        beta_scen = SCENARIO_CONFIG[scen]["elasticity"]
        for ym in TARGET_MONTHS:
            rows_out.append({
                "year_month": pd.Timestamp(ym),
                "data_type": "forecast",
                "scenario": scen,
                # Forecast q10/q90 = scenario beta ± half-SE (smaller band since
                # the scenario already shifts the mean)
                "elasticity_q10": beta_scen - 0.5 * 1.2816 * se_pool,
                "elasticity_q50": beta_scen,
                "elasticity_q90": beta_scen + 0.5 * 1.2816 * se_pool,
                "source_note": SCENARIO_CONFIG[scen]["description"],
            })

    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# M4 sales: per-scenario sales forecast combining M1 + M2 + M3
# ---------------------------------------------------------------------------
def run_m4(panel: pd.DataFrame, cohort_a: list,
            m1_full: pd.DataFrame, m2_full: pd.DataFrame,
            m3_full: pd.DataFrame) -> pd.DataFrame:
    """Build M4 history + 3-scenario forecast. Returns long-form:
        product_card_id, year_month, data_type, scenario,
        q10, q50, q90, actual_revenue_realized
    """
    log.info("===== M4 (sales) — 3 scenarios =====")
    from src.models.sales.forecast import assemble_inputs, forecast_frame
    from src.models.sales.horizon_6mo import (
        baseline_price_horizon, estimate_forward_risk_rates,
    )
    from src.models.sales.calibrate_damping import load_calibration

    LATE_DAMPING, DISASTER_DAMPING = load_calibration()

    rows_out: list[pd.DataFrame] = []

    # History — actuals from panel
    hist = panel[panel["year_month"] <= ORIGIN][
        ["product_card_id", "year_month", "revenue_realized"]].copy()
    hist = hist[hist["product_card_id"].isin(cohort_a)]
    hist["data_type"] = "actual"
    hist["scenario"] = None
    hist["q10"] = np.nan
    hist["q50"] = np.nan
    hist["q90"] = np.nan
    hist["actual_revenue_realized"] = hist["revenue_realized"]
    hist = hist.drop(columns="revenue_realized")
    rows_out.append(hist)

    # Forecast — per-scenario M4
    baseline_price = baseline_price_horizon(panel, cohort_a)
    risk_hist = pd.read_parquet(FC_DIR / "m3_risk_drag.parquet")[
        ["product_card_id", "year_month", "p_fraud", "p_cancel", "p_late",
         "disaster_index"]]
    risk_future_base = estimate_forward_risk_rates(panel, risk_hist,
                                                     list(TARGET_MONTHS))

    for scen in SCENARIOS:
        log.info("M4 scenario: %s", scen)
        # M1 for this scenario
        m1_scen = m1_full[(m1_full["data_type"] == "forecast")
                           & (m1_full["scenario"] == scen)].copy()
        m1_scen = m1_scen[["product_card_id", "year_month", "q10", "q50", "q90"]]
        m1_scen["horizon"] = (
            (m1_scen["year_month"] - ORIGIN).dt.days // 30 + 1).astype(int)
        m1_scen["model"] = "timesfm"
        m1_scen["slice"] = "future"

        # M2 elasticity for this scenario
        beta_scen = SCENARIO_CONFIG[scen]["elasticity"]

        # Risk frame for this scenario — use the scenario's drag from M3
        scen_drag = (m3_full[(m3_full["data_type"] == "forecast")
                              & (m3_full["scenario"] == scen)]
                       [["product_card_id", "year_month",
                         "disaster_index", "disaster_drag_index"]])
        risk_scen = risk_future_base.merge(
            scen_drag, on=["product_card_id", "year_month"], how="left")
        risk_scen["disaster_index"] = risk_scen["disaster_index"].fillna(0.0)
        risk_scen["disaster_drag_index"] = risk_scen["disaster_drag_index"].fillna(0.0)

        # Assemble M4 inputs
        inputs = assemble_inputs(m1_scen, risk_scen, baseline_price,
                                   slice_name="future")
        m4_scen = forecast_frame(inputs, elasticity_mean=beta_scen,
                                   elasticity_se=0.0)  # scenario already encodes the beta shift

        fc = m4_scen[["product_card_id", "year_month",
                       "sales_q10", "sales_q50", "sales_q90"]].rename(
            columns={"sales_q10": "q10", "sales_q50": "q50", "sales_q90": "q90"})
        fc["data_type"] = "forecast"
        fc["scenario"] = scen
        fc["actual_revenue_realized"] = np.nan
        rows_out.append(fc)

    m4_full = pd.concat(rows_out, ignore_index=True)
    m4_full = m4_full[["product_card_id", "year_month", "data_type", "scenario",
                         "q10", "q50", "q90", "actual_revenue_realized"]]
    return m4_full


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run() -> dict:
    log.info("===== FORECAST PIPELINE — as-of %s, horizon %s =====",
             AS_OF.date(), [m.date() for m in TARGET_MONTHS])
    panel = pd.read_parquet(PANEL_PATH)
    meta = pd.read_parquet(META_PATH)
    cohort_a = meta.loc[meta["cohort"] == "A_active", "product_card_id"].tolist()
    log.info("cohort A: %d products", len(cohort_a))

    # M3 first
    m3_full = run_m3()
    m3_full.to_parquet(FC_DIR / "m3_pipeline.parquet", index=False)
    log.info("wrote m3_pipeline.parquet — %d rows", len(m3_full))

    # M1 second (uses M3 drag)
    m1_full = run_m1(panel, cohort_a, m3_full)
    m1_full.to_parquet(FC_DIR / "m1_pipeline.parquet", index=False)
    log.info("wrote m1_pipeline.parquet — %d rows", len(m1_full))

    # M2 third
    m2_full = run_m2()
    m2_full.to_parquet(FC_DIR / "m2_pipeline.parquet", index=False)
    log.info("wrote m2_pipeline.parquet — %d rows", len(m2_full))

    # M4 fourth (uses M1 + M2 + M3)
    m4_full = run_m4(panel, cohort_a, m1_full, m2_full, m3_full)
    m4_full.to_parquet(FC_DIR / "m4_pipeline.parquet", index=False)
    log.info("wrote m4_pipeline.parquet — %d rows", len(m4_full))

    # Print summary
    print("\n===== PIPELINE SUMMARY — Feb-Jul 2018 by scenario =====")
    print("\n--- M3 disaster_drag_index portfolio mean ---")
    m3_summary = (m3_full[m3_full["data_type"] == "forecast"]
                    .groupby(["year_month", "scenario"])["disaster_drag_index"]
                    .mean().unstack().round(3))
    print(m3_summary.to_string())

    print("\n--- M1 demand q50 (gross_qty units, portfolio sum) ---")
    m1_summary = (m1_full[m1_full["data_type"] == "forecast"]
                    .groupby(["year_month", "scenario"])["q50"]
                    .sum().unstack().round(0))
    print(m1_summary.to_string())

    print("\n--- M2 elasticity beta (constant per scenario) ---")
    for scen in SCENARIOS:
        beta = SCENARIO_CONFIG[scen]["elasticity"]
        print(f"  {scen:12s}: beta = {beta:.3f}")

    print("\n--- M4 sales q50 ($, portfolio sum by scenario) ---")
    m4_summary = (m4_full[m4_full["data_type"] == "forecast"]
                    .groupby(["year_month", "scenario"])["q50"]
                    .sum().unstack().round(0))
    print(m4_summary.to_string())
    print("\n6-month totals (q50):")
    for scen in SCENARIOS:
        total = (m4_full[(m4_full["data_type"] == "forecast")
                          & (m4_full["scenario"] == scen)]["q50"].sum())
        print(f"  {scen:12s}: ${total:>14,.0f}")

    return {
        "m1": m1_full, "m2": m2_full, "m3": m3_full, "m4": m4_full,
    }


if __name__ == "__main__":
    run()
