"""Run the M1 backtest across all baseline + TimesFM models.

Conventions
-----------
- IMPORTANT: this script imports `torch` before anything else so that
  pandas 3.x / prophet's cmdstanpy DLLs don't poison the torch DLL load
  order on Windows.
- Cohort A (>= 12 active months): forecast each product individually.
- Cohort B (sparse): forecast at category level (seasonal-naive), then
  disaggregate to products by 24-mo historical share. Flagged in output.
- Origins: validation (2016-12, h=6) and test (2017-06, h=3). Each origin
  produces one frame, stored together with a `slice` column.

Outputs
-------
- forecasts/m1_demand.parquet      — long format, all models, all slices
- forecasts/m1_demand_metrics.parquet  — per-product + portfolio metrics
"""
from __future__ import annotations

# --- torch-first import order (critical on Windows) -------------------------
import torch  # noqa: F401  -- must precede pandas/prophet imports
# ---------------------------------------------------------------------------

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.demand.baselines import (
    ETSForecaster,
    ProphetForecaster,
    SARIMAForecaster,
    SeasonalNaiveForecaster,
    forecast_panel_baseline,
)
from src.models.demand.evaluate import (
    EvalSlice,
    compare_models,
    score_per_product,
    score_portfolio,
)
from src.models.demand.timesfm_model import (
    TimesFMForecaster,
    forecast_panel_timesfm,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "panel_meta.parquet"
OUT_FC = ROOT / "forecasts" / "m1_demand.parquet"
OUT_METRICS = ROOT / "forecasts" / "m1_demand_metrics.parquet"

# Slices defined by (origin, horizon, name).
# Test extended to 7 months (2017-07 .. 2018-01) — full hurricane season + recovery.
SLICES = [
    {"name": "val",  "origin": pd.Timestamp("2016-12-01"), "horizon": 6},
    {"name": "test", "origin": pd.Timestamp("2017-06-01"), "horizon": 7},
]


# ---------------------------------------------------------------------------
# Cohort B (sparse): category-level forecast → disaggregate to products
# ---------------------------------------------------------------------------
def forecast_cohort_b(
    panel: pd.DataFrame,
    cohort_b_ids: list,
    origin: pd.Timestamp,
    horizon: int,
    model_name: str = "seasonal_naive_category",
) -> pd.DataFrame:
    """Aggregate cohort B by category_id, forecast with seasonal-naive,
    then split back to products by their 24-month qty share."""
    sub = panel[panel["product_card_id"].isin(cohort_b_ids)].copy()
    sub = sub[sub["data_quality"] == "ok"]
    if sub.empty:
        return pd.DataFrame(columns=[
            "product_card_id", "year_month", "horizon",
            "q10", "q50", "q90", "model",
        ])

    months = pd.to_datetime(sorted(sub["year_month"].unique()))
    origin_idx = list(months).index(origin)
    forecast_months = months[origin_idx + 1: origin_idx + 1 + horizon]

    # category-level history
    cat_hist = (sub[sub["year_month"] <= origin]
                .groupby(["category_id", "year_month"])["qty"].sum()
                .reset_index())

    out_rows: list[pd.DataFrame] = []
    snaive = SeasonalNaiveForecaster()
    for cat, g in cat_hist.groupby("category_id"):
        g = g.sort_values("year_month")
        q = snaive.forecast(g["qty"].to_numpy(), horizon)

        # products in this category with their 24-mo share
        recent = sub[(sub["year_month"] <= origin) &
                     (sub["year_month"] > origin - pd.DateOffset(months=24)) &
                     (sub["category_id"] == cat)]
        if recent.empty:
            continue
        share = recent.groupby("product_card_id")["qty"].sum()
        share = share / share.sum() if share.sum() > 0 else share

        for pid, sh in share.items():
            out_rows.append(pd.DataFrame({
                "product_card_id": pid,
                "year_month": forecast_months,
                "horizon": np.arange(1, horizon + 1),
                "q10": q["q10"] * sh, "q50": q["q50"] * sh, "q90": q["q90"] * sh,
                "model": model_name,
            }))
    if not out_rows:
        return pd.DataFrame(columns=[
            "product_card_id", "year_month", "horizon",
            "q10", "q50", "q90", "model",
        ])
    return pd.concat(out_rows, ignore_index=True)


# ---------------------------------------------------------------------------
TARGET_COL = "gross_qty"   # M1 forecasts demand expressed (incl. fraud/cancel
                            # rows), so M4 can apply the full risk_drag without
                            # double-counting. The realised column stays as
                            # `qty_realized` for reference / monitoring only.


def _retarget_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Rename columns so the rest of the M1 pipeline (which reads `qty`)
    forecasts whatever TARGET_COL points to."""
    panel = panel.copy()
    panel = panel.rename(columns={"qty": "qty_realized"})
    panel["qty"] = panel[TARGET_COL]
    return panel


def run() -> None:
    log.info("loading panel + meta (M1 target = %s)", TARGET_COL)
    panel = _retarget_panel(pd.read_parquet(PANEL_PATH))
    meta = pd.read_parquet(META_PATH)
    cohort_a = meta.loc[meta["cohort"] == "A_active", "product_card_id"].tolist()
    cohort_b = meta.loc[meta["cohort"] == "B_sparse", "product_card_id"].tolist()
    log.info("cohort A = %d products, cohort B = %d products", len(cohort_a), len(cohort_b))

    # Init forecasters once; TimesFM model loads on first call.
    snaive = SeasonalNaiveForecaster()
    ets = ETSForecaster()
    prophet = ProphetForecaster(interval_width=0.8)
    sarima = SARIMAForecaster()
    tfm = TimesFMForecaster(max_context=24, max_horizon=7)

    all_forecasts: list[pd.DataFrame] = []
    metric_rows: list[pd.DataFrame] = []
    portfolio_rows: list[dict] = []

    for slc in SLICES:
        origin, horizon, slice_name = slc["origin"], slc["horizon"], slc["name"]
        log.info("---- slice=%s origin=%s horizon=%d ----", slice_name, origin.date(), horizon)

        # Cohort A: each model
        forecasts_a: dict[str, pd.DataFrame] = {}
        # Prophet skipped here for speed — it's 50+ minutes for the extended
        # 7-month test horizon and was the weakest baseline (val WAPE 0.21
        # vs 0.12 for TimesFM). To re-enable, uncomment the line below.
        baselines = [
            (forecast_panel_baseline(panel, snaive, origin, horizon, product_ids=cohort_a),  "seasonal_naive"),
            (forecast_panel_baseline(panel, ets,    origin, horizon, product_ids=cohort_a),  "ets"),
            (forecast_panel_baseline(panel, sarima, origin, horizon, product_ids=cohort_a),  "sarima"),
            # (forecast_panel_baseline(panel, prophet,origin, horizon, product_ids=cohort_a),  "prophet"),
            (forecast_panel_timesfm(panel, tfm,     origin, horizon, product_ids=cohort_a),  "timesfm"),
        ]
        for fc, name in baselines:
            forecasts_a[name] = fc
            log.info("  %s: %d rows", name, len(fc))

        # Cohort B: category fallback
        fc_b = forecast_cohort_b(panel, cohort_b, origin, horizon)
        log.info("  cohort_b fallback: %d rows", len(fc_b))

        # Score each model on cohort A
        actuals = panel[(panel["data_quality"] == "ok") &
                        (panel["product_card_id"].isin(cohort_a))][
            ["product_card_id", "year_month", "qty"]]
        history = panel[(panel["data_quality"] == "ok") &
                        (panel["product_card_id"].isin(cohort_a)) &
                        (panel["year_month"] <= origin)][
            ["product_card_id", "year_month", "qty"]]

        per_product: dict[str, pd.DataFrame] = {}
        for name, fc in forecasts_a.items():
            sl = EvalSlice(forecast=fc, actual=actuals, train_history=history)
            ppt = score_per_product(sl)
            per_product[name] = ppt
            port = score_portfolio(sl) | {"slice": slice_name, "model": name, "cohort": "A_active"}
            portfolio_rows.append(port)
            log.info("  [%s] portfolio: %s", name, {k: round(v, 4) if isinstance(v, float) else v for k, v in port.items()})

        # Cohort B portfolio
        if not fc_b.empty:
            actuals_b = panel[(panel["data_quality"] == "ok") &
                              (panel["product_card_id"].isin(cohort_b))][
                ["product_card_id", "year_month", "qty"]]
            history_b = panel[(panel["data_quality"] == "ok") &
                              (panel["product_card_id"].isin(cohort_b)) &
                              (panel["year_month"] <= origin)][
                ["product_card_id", "year_month", "qty"]]
            sl_b = EvalSlice(forecast=fc_b, actual=actuals_b, train_history=history_b)
            port_b = score_portfolio(sl_b) | {"slice": slice_name,
                                              "model": "seasonal_naive_category",
                                              "cohort": "B_sparse"}
            portfolio_rows.append(port_b)
            log.info("  [cohort_B] portfolio: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in port_b.items()})

        cmp_df = compare_models(per_product)
        cmp_df["slice"] = slice_name
        cmp_df["cohort"] = "A_active"
        metric_rows.append(cmp_df)

        # Stash forecasts (tag slice + cohort)
        for name, fc in forecasts_a.items():
            all_forecasts.append(fc.assign(slice=slice_name, cohort="A_active"))
        if not fc_b.empty:
            all_forecasts.append(fc_b.assign(slice=slice_name, cohort="B_sparse"))

    # Save
    OUT_FC.parent.mkdir(parents=True, exist_ok=True)
    fc_all = pd.concat(all_forecasts, ignore_index=True)
    fc_all.to_parquet(OUT_FC, index=False)
    log.info("wrote %s  shape=%s", OUT_FC, fc_all.shape)

    metrics_per_product = pd.concat(metric_rows, ignore_index=True)
    metrics_portfolio = pd.DataFrame(portfolio_rows)
    metrics = {"per_product": metrics_per_product, "portfolio": metrics_portfolio}
    metrics_per_product.to_parquet(OUT_METRICS, index=False)
    metrics_portfolio.to_parquet(
        OUT_METRICS.with_name("m1_demand_portfolio.parquet"), index=False)
    log.info("wrote %s + portfolio companion", OUT_METRICS)

    # Tidy summary
    print("\n===== PORTFOLIO SUMMARY =====")
    print(metrics_portfolio.to_string(index=False))


if __name__ == "__main__":
    run()
