"""Daily M4 sales backtest — analogue of `src/models/sales/run.py`.

Reads daily M1 forecasts + daily M3 disaster/risk + monthly M2 elasticity,
runs Monte Carlo per (product, date), and emits:

- forecasts/m4_sales_daily.parquet           (base scenario, val+test)
- forecasts/m4_sales_backtest_daily.parquet  (joined with daily actuals for evaluation)
- forecasts/m4_sales_decomposition_daily.parquet (error decomposition)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.sales.forecast import (
    SalesInputs, montecarlo_sales, baseline_price_per_product,
    forecast_frame, LATE_DAMPING, DISASTER_DAMPING, N_SAMPLES,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "daily_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "daily_panel_meta.parquet"
FC_DIR = ROOT / "forecasts"

VAL_ORIGIN = pd.Timestamp("2016-12-31")
VAL_HORIZON = 181
TEST_ORIGIN = pd.Timestamp("2017-06-30")
TEST_HORIZON = 92


def baseline_price_daily(panel: pd.DataFrame, origin: pd.Timestamp,
                          window_days: int = 90) -> pd.Series:
    """Trailing 90-day qty-weighted p_eff per product as price baseline."""
    cutoff_lo = origin - pd.Timedelta(days=window_days)
    sub = panel[(panel["date"] > cutoff_lo) &
                (panel["date"] <= origin) &
                (panel["p_eff"].notna()) &
                (panel["qty"] > 0)].copy()
    sub["pxq"] = sub["p_eff"] * sub["qty"]
    agg = sub.groupby("product_card_id").agg(pxq=("pxq", "sum"), qty=("qty", "sum"))
    out = (agg["pxq"] / agg["qty"]).rename("baseline_price")
    list_price = panel.groupby("product_card_id")["p_list"].first()
    out = out.reindex(list_price.index)
    return out.fillna(list_price).rename("baseline_price")


def assemble_daily_inputs(m1_fc: pd.DataFrame,
                            m3_risk: pd.DataFrame,
                            m3_disaster: pd.DataFrame,
                            baseline_price: pd.Series,
                            model_name: str = "sarimax") -> pd.DataFrame:
    m1 = m1_fc[m1_fc["model"] == model_name][
        ["product_card_id", "date", "horizon", "slice", "q10", "q50", "q90"]].rename(
        columns={"q10": "q10_demand", "q50": "q50_demand", "q90": "q90_demand"})

    # M3 risk daily (long history; daily p_fraud/cancel/late)
    risk_cols = ["product_card_id", "date", "p_fraud", "p_cancel", "p_late"]
    df = m1.merge(m3_risk[risk_cols], on=["product_card_id", "date"], how="left")

    # M3 disaster daily (history rows have data_type='actual')
    dis_cols = ["product_card_id", "date", "disaster_index", "disaster_drag_index"]
    dis_hist = m3_disaster[m3_disaster["data_type"] == "actual"][dis_cols]
    df = df.merge(dis_hist, on=["product_card_id", "date"], how="left")
    for c in ["p_fraud", "p_cancel", "p_late", "disaster_index", "disaster_drag_index"]:
        df[c] = df[c].fillna(0.0)

    # Baseline price
    df = df.merge(baseline_price.reset_index(), on="product_card_id", how="left")
    df["baseline_price"] = df["baseline_price"].fillna(0.0)
    return df


def montecarlo_daily(inputs: pd.DataFrame,
                       planned_price_factor: float = 1.0,
                       elasticity_mean: float = -0.687,
                       elasticity_se: float = 0.389,
                       n_samples: int = N_SAMPLES,
                       seed: int = 12345) -> pd.DataFrame:
    """Daily MC — reuses SalesInputs / montecarlo_sales row-wise."""
    rng_master = np.random.default_rng(seed)
    rows: list[dict] = []
    for _, r in inputs.iterrows():
        inp = SalesInputs(
            product_card_id=r["product_card_id"],
            year_month=r["date"],          # SalesInputs.year_month is just a label
            q10_demand=r["q10_demand"],
            q50_demand=r["q50_demand"],
            q90_demand=r["q90_demand"],
            baseline_price=r["baseline_price"],
            p_fraud=r["p_fraud"],
            p_cancel=r["p_cancel"],
            p_late=r["p_late"],
            disaster_index=r["disaster_index"],
            disaster_drag_index=r.get("disaster_drag_index", 0.0),
        )
        planned_price = r["baseline_price"] * planned_price_factor
        out = montecarlo_sales(
            inp, planned_price, elasticity_mean, elasticity_se,
            n_samples=n_samples,
            seed=int(rng_master.integers(0, 2**31 - 1)),
        )
        rows.append({
            "product_card_id": r["product_card_id"],
            "date": r["date"],
            "horizon": r["horizon"],
            "planned_price": planned_price,
            **out,
        })
    return pd.DataFrame(rows)


def run():
    log.info("===== M4 daily MC backtest =====")
    panel = pd.read_parquet(PANEL_PATH)
    meta = pd.read_parquet(META_PATH)
    cohort_a = meta.loc[meta["cohort"] == "A_active", "product_card_id"].tolist()
    log.info("cohort A: %d products", len(cohort_a))

    m1_fc = pd.read_parquet(FC_DIR / "m1_demand_daily.parquet")
    m3_risk = pd.read_parquet(FC_DIR / "m3_risk_drag_daily.parquet")
    m3_disaster = pd.read_parquet(FC_DIR / "m3_disaster_daily.parquet")

    # Load M2 monthly pool elasticity (β stays monthly per user spec)
    m2_pool = pd.read_parquet(FC_DIR / "m2_elasticity_pool.parquet").iloc[0].to_dict()
    beta = float(m2_pool["elasticity_own"])
    se = float(m2_pool["elasticity_own_se"])
    log.info("M2 pooled β = %.3f ± %.3f", beta, se)

    all_scen, all_bt = [], []
    for slc_name, origin in (("val", VAL_ORIGIN), ("test", TEST_ORIGIN)):
        log.info("--- slice=%s ---", slc_name)
        baseline_price = baseline_price_daily(panel, origin)
        m1_slc = m1_fc[m1_fc["slice"] == slc_name].copy()
        inputs = assemble_daily_inputs(m1_slc, m3_risk, m3_disaster, baseline_price)

        log.info("  running base MC (%d rows × %d samples)", len(inputs), N_SAMPLES)
        fc_base = montecarlo_daily(inputs, planned_price_factor=1.0,
                                     elasticity_mean=beta, elasticity_se=se)
        fc_base["slice"] = slc_name
        all_scen.append(fc_base)

        actuals = panel[["product_card_id", "date", "qty", "gross_qty",
                          "p_eff", "revenue_realized", "gross_revenue"]]
        bt = fc_base.merge(actuals, on=["product_card_id", "date"], how="left")
        bt["err_gross"] = bt["gross_revenue"] - bt["sales_q50_pre_risk"]
        bt["err_forward"] = bt["revenue_realized"] - bt["sales_q50"]
        all_bt.append(bt)

    fc_all = pd.concat(all_scen, ignore_index=True)
    bt_all = pd.concat(all_bt, ignore_index=True)

    fc_all.to_parquet(FC_DIR / "m4_sales_daily.parquet", index=False)
    bt_all.to_parquet(FC_DIR / "m4_sales_backtest_daily.parquet", index=False)
    log.info("wrote m4_sales_daily.parquet (%d rows)", len(fc_all))
    log.info("wrote m4_sales_backtest_daily.parquet (%d rows)", len(bt_all))

    # Portfolio headlines
    def _wape(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.nansum(np.abs(a - b)) / max(np.nansum(np.abs(a)), 1e-9))
    def _cov(a, lo, hi):
        a, lo, hi = np.asarray(a, float), np.asarray(lo, float), np.asarray(hi, float)
        return float(np.mean((a >= lo) & (a <= hi)))

    print("\n===== DAILY PORTFOLIO HEADLINES =====")
    for slc, sub in bt_all.groupby("slice"):
        wape_pre = _wape(sub["gross_revenue"].fillna(0), sub["sales_q50_pre_risk"])
        cov_pre = _cov(sub["gross_revenue"].fillna(0),
                        sub["sales_q10_pre_risk"], sub["sales_q90_pre_risk"])
        wape_fwd = _wape(sub["revenue_realized"].fillna(0), sub["sales_q50"])
        cov_fwd = _cov(sub["revenue_realized"].fillna(0),
                        sub["sales_q10"], sub["sales_q90"])
        print(f"  {slc:4s}  pre-risk     WAPE={wape_pre:.4f}  cov80={cov_pre:.4f}")
        print(f"        forward-adj  WAPE={wape_fwd:.4f}  cov80={cov_fwd:.4f}")


if __name__ == "__main__":
    run()
