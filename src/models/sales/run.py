"""M4 — orchestrator. Produces sales forecasts and a backtest table.

Outputs
-------
- forecasts/m4_sales.parquet                 — Base scenario forecasts (val+test slices)
- forecasts/m4_sales_scenarios.parquet       — Base / Stress / Tail per (product, month)
- forecasts/m4_sales_backtest.parquet        — actual vs forecast on val+test slices
- forecasts/m4_sales_decomposition.parquet   — error decomposition (demand vs price vs risk)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.sales.forecast import (
    N_SAMPLES,
    assemble_inputs,
    baseline_price_per_product,
    forecast_frame,
)
from src.models.sales.scenarios import run_scenarios

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
FC_DIR = ROOT / "forecasts"

# Forecast origins reused from M1 backtest. Test extended to 7 months
# (2017-07 .. 2018-01) to cover the full hurricane season + recovery tail.
SLICES = [
    {"name": "val",  "origin": pd.Timestamp("2016-12-01"), "horizon": 6},
    {"name": "test", "origin": pd.Timestamp("2017-06-01"), "horizon": 7},
]


def _load_inputs() -> dict:
    panel = pd.read_parquet(ROOT / "data" / "processed" / "monthly_panel.parquet")
    meta = pd.read_parquet(ROOT / "data" / "processed" / "panel_meta.parquet")
    m1 = pd.read_parquet(FC_DIR / "m1_demand.parquet")
    m1 = m1[m1["model"] == "timesfm"]   # champion
    m2_pool = pd.read_parquet(FC_DIR / "m2_elasticity_pool.parquet").iloc[0].to_dict()
    risk = pd.read_parquet(FC_DIR / "m3_risk_drag.parquet")
    return {"panel": panel, "meta": meta, "m1": m1,
            "m2_pool": m2_pool, "risk": risk}


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    num = np.nansum(np.abs(y_pred - y_true))
    den = np.nansum(np.abs(y_true))
    return float(num / den) if den > 0 else float("nan")


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    out = np.where(denom == 0, 0.0, 2 * np.abs(y_pred - y_true) / denom)
    return float(np.nanmean(out))


def _coverage(y_true, q_lo, q_hi) -> float:
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= np.asarray(q_lo, float)) &
                          (y_true <= np.asarray(q_hi, float))))


# ---------------------------------------------------------------------------
def run() -> None:
    log.info("loading inputs")
    inputs = _load_inputs()
    panel, m1, m2_pool, risk = (inputs["panel"], inputs["m1"],
                                 inputs["m2_pool"], inputs["risk"])
    epsilon_mean = float(m2_pool["elasticity_own"])
    epsilon_se = float(m2_pool["elasticity_own_se"])
    log.info("M2 elasticity: %.3f ± %.3f", epsilon_mean, epsilon_se)

    # ---------------------------------------------------------------------
    # Tier-2+3 FORWARD disaster_index is the production source for the
    # forward-risk-adj view. It's what an operator would have plugged in at
    # forecast time (no hindsight). The Tier-1 historical disaster_index in
    # `risk` (which carries the realised anomaly proxy too) is kept around
    # for the legacy/reference comparison run later.
    # ---------------------------------------------------------------------
    from src.models.sales.forward_forecast import build_forward_disaster_for_backtest
    target_months_forward = sorted({pd.Timestamp(s["origin"] + pd.DateOffset(months=h))
                                     for s in SLICES for h in range(1, s["horizon"] + 1)})
    log.info("---- building Tier-2+3 forward disaster_index (%d target months) ----",
             len(target_months_forward))
    forward_product, forward_combined = build_forward_disaster_for_backtest(
        target_months_forward)
    forward_product = forward_product.rename(
        columns={"forward_disaster_index": "disaster_index_forward"})
    risk_with_forward = risk.merge(
        forward_product, on=["product_card_id", "year_month"], how="left")
    risk_with_forward["disaster_index_historical"] = risk_with_forward["disaster_index"]
    # For target months we have a forward value; for training months we
    # leave the historical Tier-1 value in place (training period doesn't
    # need forward substitution).
    have_fwd = risk_with_forward["disaster_index_forward"].notna()
    risk_with_forward.loc[have_fwd, "disaster_index"] = \
        risk_with_forward.loc[have_fwd, "disaster_index_forward"]
    log.info("forward disaster substituted into %d (product, month) rows",
             int(have_fwd.sum()))

    # Compute the revenue-lagged `disaster_drag_index` from a *clean*
    # severity series. The convolution kernel reaches back 2 months, and
    # without scrubbing it pulls in the M3 anomaly-proxy noise from
    # non-target months — which inflates val drag in months that are
    # actually calm.
    #
    # Clean rule: use the forward-substituted value where available; zero
    # everywhere else. We keep `disaster_index` in the saved risk frame
    # untouched (for reporting / dashboards), and only build a transient
    # "drag input" column to feed the convolution.
    from src.models.sales.revenue_lag import apply_revenue_lag_per_product
    risk_with_forward["disaster_index_for_drag"] = \
        risk_with_forward["disaster_index_forward"].fillna(0.0)
    risk_with_forward = apply_revenue_lag_per_product(
        risk_with_forward, src_col="disaster_index_for_drag",
        dst_col="disaster_drag_index")
    risk_with_forward = risk_with_forward.drop(columns=["disaster_index_for_drag"])
    log.info("computed disaster_drag_index (revenue lag, clean input) on %d rows",
             len(risk_with_forward))
    risk = risk_with_forward  # default risk used by the main M4 pass

    all_base: list[pd.DataFrame] = []
    all_scen: list[pd.DataFrame] = []
    decomp_rows: list[pd.DataFrame] = []
    backtest_rows: list[pd.DataFrame] = []

    for slc in SLICES:
        name, origin, horizon = slc["name"], slc["origin"], slc["horizon"]
        log.info("---- slice=%s origin=%s horizon=%d ----", name, origin.date(), horizon)

        baseline = baseline_price_per_product(panel, origin)
        inputs_df = assemble_inputs(m1, risk, baseline, slice_name=name)
        log.info("  inputs assembled: %d rows", len(inputs_df))

        # Base scenario
        fc_base = forecast_frame(
            inputs_df,
            planned_price_factor=1.0,
            elasticity_mean=epsilon_mean,
            elasticity_se=epsilon_se,
        )
        fc_base["slice"] = name
        all_base.append(fc_base)
        log.info("  base forecast done")

        # All scenarios
        sc_dict = run_scenarios(inputs_df, elasticity_mean=epsilon_mean,
                                elasticity_se=epsilon_se)
        for scn_name, scn_df in sc_dict.items():
            scn_df["slice"] = name
            all_scen.append(scn_df)

        # Backtest two views against their respective targets:
        # - sales_q50_pre_risk  vs gross_revenue       (M1 + M2 ceiling)
        # - sales_q50 (forward-risk-adj) vs revenue_realized (operating view)
        actuals = panel[(panel["data_quality"] == "ok")][
            ["product_card_id", "year_month",
             "qty", "gross_qty", "p_eff",
             "revenue_realized", "gross_revenue"]
        ]
        bt = fc_base.merge(actuals, on=["product_card_id", "year_month"], how="left")
        bt["err_gross"]   = bt["gross_revenue"]    - bt["sales_q50_pre_risk"]
        bt["err_forward"] = bt["revenue_realized"] - bt["sales_q50"]
        backtest_rows.append(bt)

        # Error decomposition (linearised):
        #   err_total ≈ Δdemand · baseline_price + qty · Δprice − qty · price · Δrisk
        # We don't have an "actual" risk_drag to compare against (M3 is calibrated
        # but actuals at product-month grain are noisy), so we just split into
        # demand and price components and report the residual.
        # Error decomposition on the GROSS side (apples-to-apples with M1).
        bt2 = bt.merge(inputs_df, on=["product_card_id", "year_month"], suffixes=("","_in"))
        delta_demand = bt2["gross_qty"] - bt2["q50_demand"]
        delta_price = bt2["p_eff"] - bt2["planned_price"]
        decomp = pd.DataFrame({
            "product_card_id": bt2["product_card_id"],
            "year_month": bt2["year_month"],
            "slice": name,
            "actual_gross_revenue": bt2["gross_revenue"],
            "actual_revenue_realized": bt2["revenue_realized"],
            "forecast_pre_risk_q50": bt2["sales_q50_pre_risk"],
            "forecast_risk_adj_q50": bt2["sales_q50"],
            "err_gross": bt2["err_gross"],
            "err_forward": bt2["err_forward"],
            "delta_demand_qty": delta_demand,
            "delta_price": delta_price,
            "demand_contrib_to_err": delta_demand * bt2["planned_price"],
            "price_contrib_to_err": bt2["q50_demand"] * delta_price,
            "residual_contrib_to_err":
                bt2["err_gross"]
                - delta_demand * bt2["planned_price"]
                - bt2["q50_demand"] * delta_price,
        })
        decomp_rows.append(decomp)

        log.info("  %s pre-risk    vs gross_revenue    : WAPE=%.4f cov80=%.4f",
                 name,
                 _wape(bt["gross_revenue"], bt["sales_q50_pre_risk"]),
                 _coverage(bt["gross_revenue"], bt["sales_q10_pre_risk"], bt["sales_q90_pre_risk"]))
        log.info("  %s forward-adj vs revenue_realized : WAPE=%.4f cov80=%.4f",
                 name,
                 _wape(bt["revenue_realized"], bt["sales_q50"]),
                 _coverage(bt["revenue_realized"], bt["sales_q10"], bt["sales_q90"]))

    FC_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_base, ignore_index=True).to_parquet(FC_DIR / "m4_sales.parquet", index=False)
    pd.concat(all_scen, ignore_index=True).to_parquet(FC_DIR / "m4_sales_scenarios.parquet", index=False)
    pd.concat(backtest_rows, ignore_index=True).to_parquet(FC_DIR / "m4_sales_backtest.parquet", index=False)
    pd.concat(decomp_rows, ignore_index=True).to_parquet(FC_DIR / "m4_sales_decomposition.parquet", index=False)
    log.info("wrote m4_sales / m4_sales_scenarios / m4_sales_backtest / m4_sales_decomposition")

    # ---------------------------------------------------------------------
    # Reference run: M4 using the *Tier-1 historical* disaster_index in
    # the forward-risk-adj view (the pre-Tier-2 behaviour). Lets us quote
    # the WAPE delta from adding the forward-disaster signal.
    # ---------------------------------------------------------------------
    log.info("---- reference pass with Tier-1 historical disaster_index ----")
    forward_combined.to_parquet(FC_DIR / "m4_forward_disaster_combined.parquet",
                                  index=False)
    forward_product.to_parquet(FC_DIR / "m4_forward_disaster_per_product.parquet",
                                index=False)

    # Reference pass: same revenue-lag convolution but on the raw
    # Tier-1 historical disaster_index (anomaly + known mixed). This is
    # the pre-Tier-2 baseline — useful to quantify how much WAPE was
    # being lost to the noisy anomaly proxy leaking through the lag.
    from src.models.sales.revenue_lag import apply_revenue_lag_per_product
    risk_hist = inputs["risk"].copy()  # original Tier-1 historical, unmodified
    risk_hist = apply_revenue_lag_per_product(
        risk_hist, src_col="disaster_index", dst_col="disaster_drag_index")
    all_base_hist: list[pd.DataFrame] = []
    for slc in SLICES:
        name = slc["name"]
        baseline = baseline_price_per_product(panel, slc["origin"])
        inputs_df_hist = assemble_inputs(m1, risk_hist, baseline, slice_name=name)
        fc_hist = forecast_frame(
            inputs_df_hist, planned_price_factor=1.0,
            elasticity_mean=epsilon_mean, elasticity_se=epsilon_se,
        )
        fc_hist["slice"] = name
        all_base_hist.append(fc_hist)
    bt_hist = pd.concat(all_base_hist, ignore_index=True).merge(
        panel[panel["data_quality"] == "ok"]
            [["product_card_id", "year_month", "revenue_realized", "gross_revenue"]],
        on=["product_card_id", "year_month"], how="left",
    )
    bt_hist.to_parquet(FC_DIR / "m4_sales_backtest_legacy_disaster.parquet",
                        index=False)

    print("\n===== HEAD-TO-HEAD: forward-risk-adj WAPE — Tier-1 historical vs Tier-2+3 forward =====")
    bt_all = pd.concat(backtest_rows, ignore_index=True)  # this uses forward disaster
    for slc in ("val", "test"):
        h = bt_hist[bt_hist["slice"].eq(slc)]
        f = bt_all[bt_all["slice"].eq(slc)]
        if h.empty or f.empty:
            continue
        wh = _wape(h["revenue_realized"], h["sales_q50"])
        wf = _wape(f["revenue_realized"], f["sales_q50"])
        print(f"  {slc:4s}: Tier-1 historical={wh:.4f}  Tier-2+3 forward={wf:.4f}  "
              f"delta={wh-wf:+.4f}  ({(wh-wf)/max(wh,1e-9):+.1%})")

    # Portfolio headlines — two views.
    bt_all = pd.concat(backtest_rows, ignore_index=True)
    print("\n===== M4 PRE-RISK vs GROSS_REVENUE (M1 + M2 ceiling) =====")
    for slc, sub in bt_all.groupby("slice"):
        print(f"  {slc:4s}: WAPE={_wape(sub['gross_revenue'], sub['sales_q50_pre_risk']):.4f}  "
              f"cov80={_coverage(sub['gross_revenue'], sub['sales_q10_pre_risk'], sub['sales_q90_pre_risk']):.4f}")
    print("\n===== M4 FORWARD-RISK-ADJ vs REVENUE_REALIZED (operating view) =====")
    for slc, sub in bt_all.groupby("slice"):
        print(f"  {slc:4s}: WAPE={_wape(sub['revenue_realized'], sub['sales_q50']):.4f}  "
              f"cov80={_coverage(sub['revenue_realized'], sub['sales_q10'], sub['sales_q90']):.4f}  "
              f"gap={(sub['revenue_realized'].sum() - sub['sales_q50'].sum())/sub['revenue_realized'].sum():+.2%}")

    # Error decomposition summary
    decomp = pd.concat(decomp_rows, ignore_index=True)
    print("\n===== ERROR DECOMPOSITION (mean |contribution|) =====")
    print(decomp[["demand_contrib_to_err","price_contrib_to_err","residual_contrib_to_err"]]
          .abs().mean().round(2).to_string())


if __name__ == "__main__":
    run()
