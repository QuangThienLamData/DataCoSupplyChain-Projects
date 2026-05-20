"""M4 — Sales forecast integration.

Combines M1 (demand) + M2 (elasticity) + M3 (risk) into a quantile sales
forecast per (product, month, scenario).

Math
----
For each (product p, future month t):

    qty_adjusted = qty_M1 · (planned_price / baseline_price) ^ ε
    sales        = qty_adjusted · planned_price · (1 − risk_drag)

`ε` is the pooled own-price elasticity from M2 (≈ -0.69). `baseline_price`
is the trailing-3-month qty-weighted effective price. `planned_price` is
the user's policy input (defaults to baseline).

Uncertainty propagation
-----------------------
Monte Carlo with N_SAMPLES draws:
- qty   ~ Lognormal fitted to M1 (q10, q50, q90) per (product, month)
- ε     ~ Normal(β_pool, SE_pool²)  bounded to [-3, 0]
- risk  ~ Beta(α, β) calibrated to mean = predicted_rate, fixed concentration

Output is the empirical P10 / P50 / P90 of `sales` across samples.

Note on risk_drag composition
-----------------------------
M1 is now trained on `panel.gross_qty` — demand expressed by customers
*before* fraud / cancel removal. The full 4-component risk_drag applies:

    risk_drag = 1 - (1 - p_fraud) · (1 - p_cancel)
                  · (1 - LATE_DAMPING·p_late)
                  · (1 - DISASTER_DAMPING·disaster_index)

This gives M4 two backtest targets:
- Pre-risk forecast (gross_qty × planned_price) → compared against
  `panel.gross_revenue` (gross customer spend before fraud/cancel removal).
- Risk-adjusted forecast (pre-risk × (1 − risk_drag)) → compared against
  `panel.revenue_realized` (post-fraud, post-cancel actual realised).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FC_DIR = ROOT / "forecasts"
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"

N_SAMPLES = 2000  # Monte Carlo draws per (product, month, scenario)
ELASTICITY_FLOOR = -3.0
ELASTICITY_CEIL = 0.0

# Damping factors: loaded from the data-driven calibration JSON written by
# `src/models/sales/calibrate_damping.py`. DISASTER is identified off known
# hurricane events (Maria dominates). LATE has no in-data signal and uses
# an industry-typical prior. See the calibration notebook for evidence.
def _load_damping_constants() -> tuple[float, float]:
    try:
        from src.models.sales.calibrate_damping import load_calibration
        return load_calibration()
    except Exception:
        return 0.10, 0.20  # safe defaults if calibration hasn't run

LATE_DAMPING, DISASTER_DAMPING = _load_damping_constants()

# Approximate quantile of N(0,1) at p=0.9 — used to back out lognormal σ from q10/q90
Z90 = 1.2815515655446004


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------
def _lognormal_params_from_quantiles(q10: float, q50: float, q90: float) -> tuple[float, float]:
    """Fit a lognormal to a (q10, q50, q90) triple in a closed form.
    For X ~ Lognormal(μ, σ),  log(median) = μ;  log(q90) - log(q10) = 2σ·Z90.
    Falls back to a tight band if q50 ≈ 0 or quantiles are degenerate.
    """
    q50 = max(q50, 1e-3)
    if q90 <= q10 or not np.isfinite(q10) or not np.isfinite(q90):
        # No spread — use a small synthetic σ
        return float(np.log(q50)), 0.10
    mu = np.log(q50)
    sigma = (np.log(max(q90, q50 + 1e-3)) - np.log(max(q10, 1e-3))) / (2 * Z90)
    sigma = float(np.clip(sigma, 0.02, 1.5))
    return float(mu), sigma


def _sample_qty(q10: float, q50: float, q90: float, n: int,
                rng: np.random.Generator) -> np.ndarray:
    mu, sigma = _lognormal_params_from_quantiles(q10, q50, q90)
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def _sample_risk_component(p: float, n: int, concentration: float,
                            rng: np.random.Generator) -> np.ndarray:
    """Beta sample around a calibrated rate. concentration is α + β (higher
    = tighter). Clipped to [1e-4, 1 − 1e-4] to avoid degenerate Beta."""
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    a = p * concentration
    b = (1 - p) * concentration
    return rng.beta(a, b, size=n)


# ---------------------------------------------------------------------------
# Core integration (no MC) — fast deterministic version
# ---------------------------------------------------------------------------
@dataclass
class SalesInputs:
    """Bundle of all model outputs needed for one (product, month) integration.

    `disaster_index` is the storm severity profile (peak at landfall month).
    `disaster_drag_index` is the revenue-lagged version (peak at M+2) and is
    what the drag math actually uses. See `revenue_lag.py` for the kernel.
    """
    product_card_id: float
    year_month: pd.Timestamp
    q10_demand: float
    q50_demand: float
    q90_demand: float
    baseline_price: float
    p_fraud: float
    p_cancel: float
    p_late: float
    disaster_index: float
    disaster_drag_index: float = 0.0


def deterministic_sales(inp: SalesInputs, planned_price: float,
                        elasticity: float) -> float:
    """Point estimate of expected sales — no uncertainty bands.
    Full 4-component risk_drag (M1 forecasts gross demand)."""
    baseline = max(inp.baseline_price, 1e-6)
    elasticity_term = (planned_price / baseline) ** elasticity
    qty_adj = inp.q50_demand * elasticity_term
    # Use the revenue-lagged drag index here, NOT the raw severity index.
    # See `src/models/sales/revenue_lag.py` for the kernel.
    risk_drag = 1 - (
        (1 - inp.p_fraud) *
        (1 - inp.p_cancel) *
        (1 - LATE_DAMPING * inp.p_late) *
        (1 - DISASTER_DAMPING * inp.disaster_drag_index)
    )
    return float(qty_adj * planned_price * (1 - risk_drag))


# ---------------------------------------------------------------------------
# Monte Carlo integration
# ---------------------------------------------------------------------------
def montecarlo_sales(
    inp: SalesInputs,
    planned_price: float,
    elasticity_mean: float,
    elasticity_se: float,
    risk_concentration: float = 80.0,
    n_samples: int = N_SAMPLES,
    seed: int | None = None,
) -> dict:
    """Return P10/P50/P90 of sales, with component sample arrays for
    decomposition / debugging."""
    rng = np.random.default_rng(seed)

    qty = _sample_qty(inp.q10_demand, inp.q50_demand, inp.q90_demand,
                      n_samples, rng)
    eps = np.clip(rng.normal(elasticity_mean, elasticity_se, n_samples),
                  ELASTICITY_FLOOR, ELASTICITY_CEIL)

    baseline = max(inp.baseline_price, 1e-6)
    elasticity_term = (planned_price / baseline) ** eps
    qty_adj = qty * elasticity_term

    # Full 4-component risk_drag — M1 forecasts gross demand, so fraud +
    # cancel + late + disaster all apply (no double-counting).
    #
    # We split the drag into two layers because only fraud + cancel are
    # observable in history (they map to order_status):
    #   - drag_historical = fraud + cancel    (validatable against revenue_realized)
    #   - drag_forward    = late + disaster   (hypothetical — no refund / disaster
    #                                          ledger in this dataset)
    p_fraud = _sample_risk_component(inp.p_fraud, n_samples, risk_concentration, rng)
    p_cancel = _sample_risk_component(inp.p_cancel, n_samples, risk_concentration, rng)
    p_late = _sample_risk_component(inp.p_late, n_samples, risk_concentration, rng)
    # The drag math uses the revenue-lagged disaster index (peak at M+2),
    # NOT the raw severity index (peak at landfall). Reporting / dashboards
    # continue to show the raw `disaster_index` separately.
    di = np.clip(inp.disaster_drag_index, 0.0, 1.0)

    drag_historical = 1 - (1 - p_fraud) * (1 - p_cancel)
    drag_forward    = 1 - (1 - LATE_DAMPING * p_late) * (1 - DISASTER_DAMPING * di)
    risk_drag       = 1 - (1 - drag_historical) * (1 - drag_forward)

    # Three views:
    #  - pre-risk           : gross expected sales (vs gross_revenue)
    #  - historical-risk    : after fraud + cancel (vs revenue_realized)
    #  - forward-risk-adj   : after all four risks (planning number; not backtestable)
    sales_pre_risk = np.maximum(qty_adj * planned_price, 0.0)
    sales_historical_risk = np.maximum(sales_pre_risk * (1 - drag_historical), 0.0)
    sales = np.maximum(sales_pre_risk * (1 - risk_drag), 0.0)

    return {
        # Forward risk-adjusted (planning view; used in scenarios)
        "sales_q10": float(np.quantile(sales, 0.10)),
        "sales_q50": float(np.quantile(sales, 0.50)),
        "sales_q90": float(np.quantile(sales, 0.90)),
        "sales_mean": float(sales.mean()),
        "sales_std": float(sales.std()),
        # Pre-risk (backtestable against gross_revenue)
        "sales_q10_pre_risk": float(np.quantile(sales_pre_risk, 0.10)),
        "sales_q50_pre_risk": float(np.quantile(sales_pre_risk, 0.50)),
        "sales_q90_pre_risk": float(np.quantile(sales_pre_risk, 0.90)),
        # Historical-risk (backtestable against revenue_realized)
        "sales_q10_historical": float(np.quantile(sales_historical_risk, 0.10)),
        "sales_q50_historical": float(np.quantile(sales_historical_risk, 0.50)),
        "sales_q90_historical": float(np.quantile(sales_historical_risk, 0.90)),
        # Components for decomposition
        "qty_q50_adj": float(np.quantile(qty_adj, 0.50)),
        "risk_drag_q50": float(np.quantile(risk_drag, 0.50)),
        "drag_historical_q50": float(np.quantile(drag_historical, 0.50)),
        "drag_forward_q50": float(np.quantile(drag_forward, 0.50)),
        "elasticity_term_q50": float(np.quantile(elasticity_term, 0.50)),
    }


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def baseline_price_per_product(panel: pd.DataFrame,
                                origin: pd.Timestamp,
                                window: int = 3) -> pd.Series:
    """Trailing-`window`-month qty-weighted p_eff per product as the price
    baseline at the forecast origin."""
    clean = panel[panel["data_quality"] == "ok"]
    cutoff_lo = origin - pd.DateOffset(months=window)
    sub = clean[(clean["year_month"] > cutoff_lo) &
                (clean["year_month"] <= origin) &
                (clean["p_eff"].notna()) &
                (clean["qty"] > 0)].copy()
    sub["pxq"] = sub["p_eff"] * sub["qty"]
    agg = sub.groupby("product_card_id").agg(
        pxq=("pxq", "sum"), qty=("qty", "sum"),
    )
    out = (agg["pxq"] / agg["qty"]).rename("baseline_price")
    # Products with no recent activity → fall back to list price
    list_price = panel.groupby("product_card_id")["p_list"].first()
    out = out.reindex(list_price.index)
    return out.fillna(list_price).rename("baseline_price")


def assemble_inputs(
    m1_forecasts: pd.DataFrame,
    risk_drag: pd.DataFrame,
    baseline_price: pd.Series,
    slice_name: str | None = None,
    model_name: str = "timesfm",
) -> pd.DataFrame:
    """Join M1 + M3 + baseline_price into one frame per (product, month)."""
    m1 = m1_forecasts.copy()
    if "model" in m1.columns:
        m1 = m1[m1["model"] == model_name]
    if slice_name and "slice" in m1.columns:
        m1 = m1[m1["slice"] == slice_name]

    df = m1[["product_card_id", "year_month", "horizon", "q10", "q50", "q90"]].copy()
    df = df.rename(columns={"q10": "q10_demand",
                            "q50": "q50_demand",
                            "q90": "q90_demand"})
    risk_cols = ["product_card_id", "year_month",
                 "p_fraud", "p_cancel", "p_late", "disaster_index"]
    if "disaster_drag_index" in risk_drag.columns:
        risk_cols.append("disaster_drag_index")
    df = df.merge(risk_drag[risk_cols],
                  on=["product_card_id", "year_month"], how="left")
    fill_cols = ["p_fraud", "p_cancel", "p_late", "disaster_index"]
    if "disaster_drag_index" in df.columns:
        fill_cols.append("disaster_drag_index")
    else:
        df["disaster_drag_index"] = 0.0
    for c in fill_cols:
        df[c] = df[c].fillna(0.0)
    df = df.merge(baseline_price.reset_index(),
                  on="product_card_id", how="left")
    df["baseline_price"] = df["baseline_price"].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Batch forecast a frame
# ---------------------------------------------------------------------------
def forecast_frame(
    inputs: pd.DataFrame,
    planned_price_factor: float = 1.0,
    elasticity_mean: float = -0.687,
    elasticity_se: float = 0.389,
    n_samples: int = N_SAMPLES,
    seed: int = 12345,
) -> pd.DataFrame:
    """Apply Monte Carlo integration row-wise. planned_price_factor scales the
    baseline price (1.0 = base case, 1.10 = +10% list, 0.90 = -10% discount-deepening)."""
    rng_master = np.random.default_rng(seed)
    rows: list[dict] = []
    for _, r in inputs.iterrows():
        inp = SalesInputs(
            product_card_id=r["product_card_id"],
            year_month=r["year_month"],
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
            "year_month": r["year_month"],
            "horizon": r["horizon"],
            "planned_price": planned_price,
            **out,
        })
    return pd.DataFrame(rows)
