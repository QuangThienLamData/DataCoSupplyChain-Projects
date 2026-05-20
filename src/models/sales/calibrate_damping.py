"""Empirically calibrate LATE_DAMPING and DISASTER_DAMPING.

Problem
-------
The damping factors in M4 convert M3's risk signals into revenue drag:

    drag_forward = 1 - (1 - LATE_DAMPING · p_late)(1 - DISASTER_DAMPING · disaster_index)

Hand-set values (LATE 0.10, DISASTER 0.20) produced a 22% gap between the
risk-adjusted forecast and realised history that lacked supporting data.

What we can calibrate
---------------------

**DISASTER_DAMPING** — yes. The truncated period 2017-10..2018-01 contains
Hurricane Maria evidence (PR products dropped 50–60% MoM relative to seasonal
expectation). We fit:

    implied_drag_{p,t} = 1 - gross_qty_actual_{p,t} / gross_qty_expected_{p,t}

where the expected is the same product's value 12 months earlier (seasonal
naive). We then regress `implied_drag ~ disaster_index_{p,t}` and report the
slope as DISASTER_DAMPING. Limited to (a) products with non-trivial PR
exposure, where Maria's effect dominates, OR (b) all products where
disaster_index > 0.2 to avoid identifying off zero-noise.

**LATE_DAMPING** — no in-data signal. The dataset has `late_delivery_risk`
(a flag) but no refund column. We report the upper bound from a regression
against the residual after disaster (essentially: how much *extra* drag
beyond disaster correlates with late rate), but treat 0.10 as a prior
informed by typical e-commerce refund rates on late deliveries (~5–15%).

The script writes the calibrated constants into a small JSON config
consumed by `forecast.py`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
DISASTER_PATH = ROOT / "forecasts" / "m3d_disaster_product.parquet"
CONFIG_PATH = ROOT / "forecasts" / "m4_damping_calibration.json"

# Prior on LATE_DAMPING; not changed unless we get evidence
LATE_PRIOR = 0.10

# Trimming bounds to keep regression robust to outliers
DRAG_LO, DRAG_HI = -0.9, 0.99
DI_THRESHOLD_FOR_FIT = 0.20  # only rows with non-trivial disaster signal


def _seasonal_baseline(panel: pd.DataFrame) -> pd.DataFrame:
    """Add `gross_qty_expected_seasonal` = same product, same month, 1 year earlier."""
    p = panel.sort_values(["product_card_id", "year_month"]).copy()
    p["gross_qty_lag12"] = p.groupby("product_card_id")["gross_qty"].shift(12)
    p = p.rename(columns={"gross_qty_lag12": "gross_qty_expected_seasonal"})
    return p


def _ols_slope_no_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope-only OLS y = β·x + noise. Returns (slope, R²)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return float("nan"), float("nan")
    slope = float(np.dot(x, y) / np.dot(x, x))
    ss_res = float(np.sum((y - slope * x) ** 2))
    ss_tot = float(np.sum(y ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2


def _project_known_severity_to_products(panel: pd.DataFrame) -> pd.Series:
    """For each (product, month), the maximum known-disaster severity that
    touches any of its destination markets. Used to filter the calibration
    sample to rows where a real event was occurring."""
    import sqlite3
    from src.models.risk.known_disasters_v2 import country_monthly_indicator

    # Country-level severity per month (US + PR)
    cty = country_monthly_indicator()
    # Project to products via the *customer*-country mix this month
    with sqlite3.connect(ROOT / "data" / "dataco.db") as conn:
        cmix = pd.read_sql(
            "SELECT order_date, product_card_id, customer_country FROM supply_chain",
            conn, parse_dates=["order_date"])
    cmix["year_month"] = cmix["order_date"].dt.to_period("M").dt.to_timestamp()
    mix = (cmix.assign(_one=1)
                .pivot_table(index=["product_card_id", "year_month"],
                             columns="customer_country", values="_one",
                             aggfunc="sum", fill_value=0))
    mix = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).reset_index()

    sev = cty.pivot(index="year_month", columns="customer_country",
                    values="known_severity").reset_index().fillna(0)
    out = mix.merge(sev, on="year_month", how="left", suffixes=("_mix", "_sev")).fillna(0)
    countries = [c for c in cty["customer_country"].unique()
                 if c in mix.columns and c in sev.columns]
    out["known_severity_product"] = sum(
        out[f"{c}_mix"] * out[f"{c}_sev"] for c in countries
    )
    return out.set_index(["product_card_id", "year_month"])["known_severity_product"]


def calibrate() -> dict:
    """Calibrate DISASTER_DAMPING against the **revenue-lagged** drag index,
    not the raw severity index. DAMPING multiplies the drag index in M4, so
    that's the regressor whose slope we want.
    """
    log.info("loading panel + disaster index")
    panel = pd.read_parquet(PANEL_PATH)
    disaster = pd.read_parquet(DISASTER_PATH)
    panel = panel.merge(
        disaster[["product_card_id", "year_month", "disaster_index"]],
        on=["product_card_id", "year_month"], how="left",
    )
    panel["disaster_index"] = panel["disaster_index"].fillna(0.0)

    # Derive the revenue-lagged drag index for the calibration fit. M4
    # uses `disaster_drag_index` against DISASTER_DAMPING, so we fit
    # implied_drag on disaster_drag_index for self-consistent calibration.
    from src.models.sales.revenue_lag import apply_revenue_lag_per_product
    panel = apply_revenue_lag_per_product(
        panel, src_col="disaster_index", dst_col="disaster_drag_index")
    # Use disaster_drag_index as the regressor for calibration
    panel["disaster_index"] = panel["disaster_drag_index"]

    log.info("projecting known-event severity to (product, month)")
    sev = _project_known_severity_to_products(panel)
    panel = panel.merge(sev.rename("known_severity"),
                        left_on=["product_card_id", "year_month"],
                        right_index=True, how="left")
    panel["known_severity"] = panel["known_severity"].fillna(0.0)

    log.info("computing seasonal-naive expectations")
    panel = _seasonal_baseline(panel)
    panel["implied_drag"] = 1 - (
        panel["gross_qty"] / panel["gross_qty_expected_seasonal"].replace(0, np.nan)
    )

    # --- Two slope estimates ---
    # (a) Broad: any (product, month) with elevated disaster_index. Low signal-
    #     to-noise because anomaly proxy fires on non-disaster events too.
    fit_broad = panel[
        panel["disaster_index"].between(DI_THRESHOLD_FOR_FIT, 1.0) &
        panel["gross_qty_expected_seasonal"].notna() &
        panel["gross_qty_expected_seasonal"].gt(0) &
        panel["implied_drag"].between(DRAG_LO, DRAG_HI)
    ].copy()
    slope_broad, r2_broad = _ols_slope_no_intercept(
        fit_broad["disaster_index"].to_numpy(),
        fit_broad["implied_drag"].to_numpy(),
    )
    log.info("BROAD slope=%.3f  R²=%.3f  n=%d", slope_broad, r2_broad, len(fit_broad))

    # (b) Known-events only: rows where the hurricane calendar fired (Maria
    #     dominates this sample). Highest causal identification.
    fit_known = panel[
        panel["known_severity"].gt(0) &
        panel["gross_qty_expected_seasonal"].notna() &
        panel["gross_qty_expected_seasonal"].gt(0) &
        panel["implied_drag"].between(DRAG_LO, DRAG_HI)
    ].copy()
    slope_known, r2_known = _ols_slope_no_intercept(
        fit_known["disaster_index"].to_numpy(),
        fit_known["implied_drag"].to_numpy(),
    )
    log.info("KNOWN slope=%.3f  R²=%.3f  n=%d", slope_known, r2_known, len(fit_known))

    # Use the known-events slope as the primary calibration. Falls back to
    # broad if known-events sample is too small.
    disaster_slope = slope_known if len(fit_known) >= 15 and np.isfinite(slope_known) \
        else slope_broad
    disaster_r2 = r2_known if len(fit_known) >= 15 and np.isfinite(slope_known) \
        else r2_broad
    fit_df = fit_known if len(fit_known) >= 15 and np.isfinite(slope_known) \
        else fit_broad

    # LATE: try to identify any residual drag after disaster is removed.
    panel["implied_drag_after_disaster"] = (
        panel["implied_drag"] - disaster_slope * panel["disaster_index"]
    )
    late_fit = panel[
        panel["gross_qty_expected_seasonal"].notna() &
        panel["gross_qty_expected_seasonal"].gt(0) &
        panel["implied_drag_after_disaster"].between(DRAG_LO, DRAG_HI) &
        panel["late_rate"].notna() &
        panel["late_rate"].gt(0)
    ].copy()
    late_slope, late_r2 = _ols_slope_no_intercept(
        late_fit["late_rate"].to_numpy(),
        late_fit["implied_drag_after_disaster"].to_numpy(),
    )
    log.info("LATE residual slope=%.3f  R²=%.3f  n=%d",
             late_slope, late_r2, len(late_fit))

    # Decide final values.
    # DISASTER: trust the slope, floor / ceiling for sanity.
    disaster_final = float(np.clip(disaster_slope, 0.05, 1.0))
    # LATE: only override prior if slope is positive AND R² > 0.05 (signal exists).
    if np.isfinite(late_slope) and late_slope > 0 and late_r2 > 0.05:
        late_final = float(np.clip(late_slope, 0.0, 0.30))
        late_source = "data"
    else:
        late_final = LATE_PRIOR
        late_source = "prior (no in-data signal)"

    out = {
        "DISASTER_DAMPING": disaster_final,
        "DISASTER_calibration": {
            "method": "OLS slope of implied_drag (1 − actual/expected_seasonal) "
                      "on disaster_index, restricted to known-event rows where "
                      "the hurricane calendar fired (primary) — falls back to "
                      "broad sample if known-events sample is too small.",
            "broad": {"slope": slope_broad, "r2": r2_broad, "n": int(len(fit_broad))},
            "known_events_only": {"slope": slope_known, "r2": r2_known,
                                   "n": int(len(fit_known))},
            "selected": "known_events_only" if len(fit_known) >= 15 and np.isfinite(slope_known)
                        else "broad",
        },
        "LATE_DAMPING": late_final,
        "LATE_calibration": {
            "slope": late_slope, "r2": late_r2,
            "n_rows": int(len(late_fit)),
            "source": late_source,
            "prior": LATE_PRIOR,
        },
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(out, indent=2))
    log.info("wrote calibration to %s", CONFIG_PATH)
    return out


def load_calibration() -> tuple[float, float]:
    """Return (LATE_DAMPING, DISASTER_DAMPING). Falls back to prior values if
    the calibration file is missing (e.g., first run)."""
    if not CONFIG_PATH.exists():
        return LATE_PRIOR, 0.20
    cfg = json.loads(CONFIG_PATH.read_text())
    return float(cfg["LATE_DAMPING"]), float(cfg["DISASTER_DAMPING"])


if __name__ == "__main__":
    out = calibrate()
    print(json.dumps(out, indent=2))
