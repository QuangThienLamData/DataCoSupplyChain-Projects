"""Tier-2 forward forecast — M4 sales projection using NHC active-storms
input instead of the historical Tier-1 disaster index.

This is the operational mode: stand at as_of_dt, look at the active storms
NHC is watching today (or, for backtest, the HURDAT2 storms in flight
±12h of as_of_dt), project their impact onto the next 1-3 months for each
product via the customer-country mix, then run M4 Monte Carlo with the
forward-disaster substituted in.

The standard M4 historical run uses the *realised* disaster_index from M3
(known after the fact). This module produces a parallel forecast that
would have been issued *before* the events occurred. Comparing the two
on archived storms (Irma, Maria, Harvey) shows the value of Tier 2
forecast skill.

Usage
-----
    from src.models.sales.forward_forecast import forward_m4_forecast
    out = forward_m4_forecast(
        as_of_dt=pd.Timestamp("2017-09-05"),
        target_months=("2017-09-01", "2017-10-01"),
    )
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.nhc_active_storms import (
    fetch_active_storms_live, simulate_from_hurdat,
)
from src.features.forward_exposure import (
    compute_forward_exposure, lift_to_monthly,
)
from src.models.sales.forecast import (
    assemble_inputs, forecast_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
FC_DIR = ROOT / "forecasts"
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"


def project_forward_to_products(
    forward_monthly: pd.DataFrame,
    target_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Per (product, target_month), forward_disaster_index = customer-country
    mix-weighted sum of the per-country forward severities.

    Uses customer-country mix (consistent with Tier-1 disaster.py's
    customer-side projection).
    """
    if forward_monthly.empty:
        # No active storms — every product gets 0 forward disaster
        with sqlite3.connect(ROOT / "data" / "dataco.db") as conn:
            pids = pd.read_sql("SELECT DISTINCT product_card_id FROM supply_chain",
                                conn)
        out = pd.MultiIndex.from_product(
            [pids["product_card_id"], target_months],
            names=["product_card_id", "year_month"]).to_frame(index=False)
        out["forward_disaster_index"] = 0.0
        return out

    # Mix by product-month: use 12 months of order history up to the latest
    # available target_month as a stable customer-country mix
    cutoff = max(target_months)
    with sqlite3.connect(ROOT / "data" / "dataco.db") as conn:
        orders = pd.read_sql(
            "SELECT order_date, product_card_id, customer_country FROM supply_chain",
            conn, parse_dates=["order_date"])
    mix_window_start = cutoff - pd.DateOffset(months=12)
    mix_orders = orders[(orders["order_date"] >= mix_window_start)
                         & (orders["order_date"] <= cutoff)]
    mix = (mix_orders.assign(_one=1)
                       .pivot_table(index="product_card_id",
                                    columns="customer_country",
                                    values="_one", aggfunc="sum",
                                    fill_value=0))
    mix = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Use country-level forward index (state-level would need state mix)
    cty = (forward_monthly[forward_monthly["customer_state"].isna()]
            .copy())
    # Also fold PR (which is country=Puerto Rico, state=PR) into country
    pr = forward_monthly[forward_monthly["customer_country"].eq("Puerto Rico")]
    if not pr.empty:
        cty = pd.concat([
            cty,
            pr.assign(customer_state=None)
        ], ignore_index=True)
    cty = (cty.groupby(["customer_country", "year_month"], dropna=False)
              ["forward_disaster_index"].max().reset_index())

    cty_wide = cty.pivot(index="year_month", columns="customer_country",
                          values="forward_disaster_index").fillna(0)

    rows: list[dict] = []
    for tm in target_months:
        tm = pd.Timestamp(tm)
        country_row = cty_wide.loc[tm] if tm in cty_wide.index else None
        for pid in mix.index:
            if country_row is None:
                disaster = 0.0
            else:
                # Σ mix[pid, country] × country_row[country]
                shared = list(set(mix.columns) & set(country_row.index))
                disaster = float(np.sum(
                    mix.loc[pid, shared].to_numpy() *
                    country_row[shared].to_numpy()
                ))
            rows.append({
                "product_card_id": pid,
                "year_month": tm,
                "forward_disaster_index": disaster,
            })
    return pd.DataFrame(rows)


def forward_m4_forecast(
    as_of_dt: pd.Timestamp,
    target_months: list | tuple,
    use_live: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run Tier-2 forward M4: predict next 1-3 months using current storms.

    Args:
        as_of_dt: "now" — controls which storms are considered active
        target_months: months to forecast (e.g., next month, +2 months)
        use_live: if True, fetch from NHC live feed; if False, simulate from
                  HURDAT2 (for backtesting)

    Returns dict with keys:
        active_storms     — what was on the radar at as_of_dt
        forward_daily     — daily forward exposure by region
        forward_monthly   — monthly roll-up by region
        forward_product   — per-(product, month) forward_disaster_index
        m4_forward        — M4 sales forecast using the forward disaster_index
    """
    target_months = [pd.Timestamp(m) for m in target_months]
    log.info("Tier-2 M4 forward — as_of=%s, targets=%s", as_of_dt,
             [t.date() for t in target_months])

    # Step 1: get active storms
    if use_live:
        active = fetch_active_storms_live()
    else:
        active = simulate_from_hurdat(as_of_dt)
    log.info("active forecast points: %d (%d storms)",
             len(active), 0 if active.empty else active["name"].nunique())

    # Step 2: forward exposure → daily → monthly
    forward_daily = compute_forward_exposure(active)
    forward_monthly = lift_to_monthly(forward_daily)
    log.info("forward exposure rows: daily=%d, monthly=%d",
             len(forward_daily), len(forward_monthly))

    # Step 3: project to products
    forward_product = project_forward_to_products(forward_monthly, target_months)
    log.info("forward_product rows: %d", len(forward_product))

    # Step 4: rebuild M4 inputs with the forward disaster_index swapped in
    panel = pd.read_parquet(PANEL_PATH)
    m1 = pd.read_parquet(FC_DIR / "m1_demand.parquet")
    risk = pd.read_parquet(FC_DIR / "m3_risk_drag.parquet")
    baseline_price = (panel.dropna(subset=["p_eff"])
                            .sort_values(["product_card_id", "year_month"])
                            .groupby("product_card_id")["p_eff"]
                            .apply(lambda s: s.tail(3).mean())
                            .rename("baseline_price"))

    # Limit M1 to target months
    m1_target = m1[m1["year_month"].isin(target_months)]
    if m1_target.empty:
        log.warning("M1 has no rows for target months — using val/test rows instead")
        m1_target = m1[m1["slice"].isin(["val", "test"])]

    inputs = assemble_inputs(m1_target, risk, baseline_price)
    # Override disaster_index with the forward value
    inputs = inputs.merge(forward_product,
                           on=["product_card_id", "year_month"], how="left")
    inputs["disaster_index_historical"] = inputs["disaster_index"]
    inputs["disaster_index"] = inputs["forward_disaster_index"].fillna(
        inputs["disaster_index"])

    # Step 5: M4 Monte Carlo
    m4 = forecast_frame(inputs)
    m4 = m4.merge(inputs[["product_card_id", "year_month",
                           "disaster_index_historical",
                           "forward_disaster_index"]],
                   on=["product_card_id", "year_month"], how="left")
    log.info("M4 forward forecast rows: %d", len(m4))

    return {
        "active_storms": active,
        "forward_daily": forward_daily,
        "forward_monthly": forward_monthly,
        "forward_product": forward_product,
        "m4_forward": m4,
    }


def build_forward_disaster_for_backtest(
    target_months: list,
    walk_every_h: int = 24,
    lookback_days: int = 7,
    lookforward_days: int = 35,
    seasonal_as_of: str = "aug",
    tier3_value_col: str = "known_severity",
) -> pd.DataFrame:
    """Build per-(product, month) `disaster_index` we *would have predicted*
    for each target month under daily Tier-2 refresh + Tier-3 outlook.

    For each target month:
        1. Walk every `walk_every_h` from (month_start - lookback_days) to
           (month_end + lookforward_days), calling simulate_from_hurdat at
           each timestamp.
        2. Take the MAX forward_disaster_index per region across all as_ofs
           whose forecast date falls within the target month. (Operationally:
           "the most severe forward signal we ever issued for this month.")
        3. Combine with Tier-3 baseline (climatology × seasonal multiplier).
        4. Project to per-product via customer-country mix.

    The result is what an operator running this stack daily during 2017
    would have plugged into M4 for each month, *without* hindsight.

    Returns DataFrame: product_card_id, year_month, forward_disaster_index.
    """
    from src.data.nhc_active_storms import simulate_from_hurdat
    from src.features.forward_exposure import compute_forward_exposure
    from src.models.risk.seasonal_outlook import (
        apply_outlook, climatology_baseline, combine_with_tier2,
    )

    target_months = sorted({pd.Timestamp(m).normalize() for m in target_months})
    span_start = (min(target_months) - pd.Timedelta(days=lookback_days)).normalize()
    span_end = ((max(target_months) + pd.DateOffset(months=1) - pd.Timedelta(days=1))
                + pd.Timedelta(days=lookforward_days)).normalize()

    log.info("walking Tier-2 sim from %s to %s every %dh",
             span_start.date(), span_end.date(), walk_every_h)
    daily_predictions: list[pd.DataFrame] = []
    cur = span_start
    while cur <= span_end:
        active = simulate_from_hurdat(cur)
        if not active.empty:
            fwd = compute_forward_exposure(active)
            if not fwd.empty:
                daily_predictions.append(fwd)
        cur += pd.Timedelta(hours=walk_every_h)

    if daily_predictions:
        daily = pd.concat(daily_predictions, ignore_index=True)
        # First: aggregate to (region, *landfall month*) — the month the
        # storm actually impacts each region (taken as the forecast_date's
        # calendar month for the daily cone forecast).
        daily["landfall_month"] = daily["forecast_date"].dt.to_period("M").dt.to_timestamp()
        landfall_monthly = (daily.groupby(["customer_country", "customer_state",
                                           "landfall_month"], dropna=False)
                                  ["forward_disaster_index"].max().reset_index())
        # Apply the LAG_PROFILE: revenue impact peaks at M+2, not landfall.
        # See src/features/storm_exposure.py:LAG_PROFILE for the curve.
        from src.features.storm_exposure import LAG_PROFILE
        lagged_rows: list[dict] = []
        for _, r in landfall_monthly.iterrows():
            for k, factor in LAG_PROFILE.items():
                target = (r["landfall_month"] + pd.DateOffset(months=k)).normalize()
                lagged_rows.append({
                    "customer_country": r["customer_country"],
                    "customer_state": r["customer_state"],
                    "year_month": target,
                    "forward_disaster_index": float(r["forward_disaster_index"]) * factor,
                })
        lagged_df = pd.DataFrame(lagged_rows)
        tier2_monthly = (lagged_df[lagged_df["year_month"].isin(target_months)]
                          .groupby(["customer_country", "customer_state",
                                    "year_month"], dropna=False)
                          ["forward_disaster_index"].max().reset_index())
        log.info("Tier-2 rolling-max + LAG monthly rows: %d", len(tier2_monthly))
    else:
        tier2_monthly = pd.DataFrame(columns=[
            "customer_country", "customer_state", "year_month",
            "forward_disaster_index",
        ])
        log.info("no Tier-2 hits across the walk window")

    # Tier 3: climatology × seasonal multiplier
    cty_hist = pd.read_parquet(FC_DIR / "m3d_disaster_country.parquet")
    tier3 = apply_outlook(list(target_months), cty_hist,
                           as_of=seasonal_as_of)
    log.info("Tier-3 baseline rows: %d", len(tier3))

    # Combine to per-country forward index
    if tier2_monthly.empty:
        combined = tier3.copy()
        combined["tier2_forward"] = 0.0
        combined["forward_disaster_combined"] = combined["tier3_baseline"]
    else:
        combined = combine_with_tier2(tier3, tier2_monthly)
    log.info("combined Tier-2+3 rows: %d", len(combined))

    # Project to (product, month) via customer-country mix
    cty_idx = combined.rename(
        columns={"forward_disaster_combined": "forward_disaster_index"})[
        ["customer_country", "year_month", "forward_disaster_index"]]
    forward_product = project_forward_to_products(
        cty_idx.assign(customer_state=None), list(target_months))
    return forward_product, combined


def m4_with_forward_disaster(
    target_months: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run M4 backtest using Tier-2+3 forward disaster_index.

    Returns (m4_forward_df, comparison_df). The first is the new M4 output;
    the second is a side-by-side of disaster_index values (historical vs
    forward) per (product, month).
    """
    log.info("computing per-product forward disaster_index for %d target months",
             len(target_months))
    forward_product, combined = build_forward_disaster_for_backtest(target_months)

    # Load M4 standard inputs
    m1 = pd.read_parquet(FC_DIR / "m1_demand.parquet")
    m1 = m1[m1["model"] == "timesfm"]
    risk = pd.read_parquet(FC_DIR / "m3_risk_drag.parquet")
    panel = pd.read_parquet(PANEL_PATH)
    baseline_price = (panel.dropna(subset=["p_eff"])
                            .sort_values(["product_card_id", "year_month"])
                            .groupby("product_card_id")["p_eff"]
                            .apply(lambda s: s.tail(3).mean())
                            .rename("baseline_price"))

    m1_target = m1[m1["year_month"].isin(target_months)]
    if m1_target.empty:
        m1_target = m1[m1["slice"].isin(["val", "test"])]
        log.info("using val+test M1 rows: %d", len(m1_target))

    inputs = assemble_inputs(m1_target, risk, baseline_price)
    inputs = inputs.merge(forward_product, on=["product_card_id", "year_month"],
                           how="left")
    inputs["disaster_index_historical"] = inputs["disaster_index"]
    inputs["disaster_index"] = inputs["forward_disaster_index"].fillna(0.0)

    log.info("re-running M4 forecast_frame with forward disaster_index")
    m4_forward = forecast_frame(inputs)
    m4_forward = m4_forward.merge(
        inputs[["product_card_id", "year_month",
                "disaster_index_historical", "forward_disaster_index",
                "p_fraud", "p_cancel", "p_late"]],
        on=["product_card_id", "year_month"], how="left",
    )
    return m4_forward, combined


if __name__ == "__main__":
    # Demo: stand at Sep 1, 2017 — 5 days before Irma hits PR.
    out = forward_m4_forecast(
        as_of_dt=pd.Timestamp("2017-09-01"),
        target_months=[pd.Timestamp("2017-09-01")],
    )

    print("\n=== Active storms on Sep 1, 2017 ===")
    if not out["active_storms"].empty:
        print(out["active_storms"][["name", "lead_h", "lat", "lon",
                                      "max_wind_kt", "cone_radius_km"]]
                .round(2).to_string(index=False))

    print("\n=== Forward exposure (monthly roll-up) — top 10 ===")
    print(out["forward_monthly"]
            .sort_values("forward_disaster_index", ascending=False).head(10)
            .round(3).to_string(index=False))

    print("\n=== M4 forward vs historical disaster_index — top 10 products ===")
    cmp = out["m4_forward"].sort_values("forward_disaster_index",
                                          ascending=False).head(10)
    print(cmp[["product_card_id", "year_month",
                "disaster_index_historical", "forward_disaster_index",
                "sales_q50", "sales_q50_historical", "sales_q50_pre_risk"]]
            .round(3).to_string(index=False))
