"""Tier-2 forward exposure: convert NHC active-storm forecasts (5-day
cone, 12h-spaced) into per-(region, day) forward `disaster_index`.

Logic
-----
For each forecast point (storm, lead_h, lat, lon, wind, cone_radius_km):

1. **Effective influence radius** widens with the cone:
       eff_radius = INFLUENCE_RADIUS_KM + cone_radius_km
   This captures "we don't know exactly where the storm will be at +120h,
   so any region within the wider envelope is at risk."

2. **Effective wind at region**: same decay as Tier 1, but with eff_radius:
       wind_at_region = wind * clamp((eff_radius - distance) / (eff_radius - TS_FORCE), 0, 1)

3. **Severity score**: Tier-1 formula on the effective wind, capped [0, 1].
   No duration component (forecast points are instantaneous), no direct-hit
   bump (cone covers uncertainty).

4. **Lead-time confidence**: scores get scaled by a confidence factor that
   reflects forecast skill. At +12h confidence ≈ 0.95; at +120h ≈ 0.5.
   Beyond that, the forecast is roughly informationless.

5. **Per-region per-day aggregation**: take the MAX severity across all
   storms and all forecast points landing on a given day. We aggregate to
   daily granularity for operational dashboards; a monthly roll-up is the
   `lift_to_monthly()` helper for plugging into M4's product-month forecast.

Schema out (long-form):
  customer_country, customer_state (None for country-level), forecast_date,
  forward_disaster_index, contributing_storms, peak_lead_h
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.region_centroids import (
    COUNTRY_CENTROIDS, DATACO_US_STATES, US_STATE_CENTROIDS,
)
from src.features.storm_exposure import (
    INFLUENCE_RADIUS_KM, TS_FORCE_RADIUS_KM, haversine_km,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed"


def _confidence_at_lead(lead_h: float) -> float:
    """NHC-skill-inspired confidence factor.

    At lead_h=0, perfect knowledge → 1.0. By +120h, position error is
    ~325km and intensity error doubles — we discount to ~0.5. Linear
    interpolation between, beyond 120h decay to 0 by 168h.
    """
    if lead_h <= 0:
        return 1.0
    if lead_h <= 120:
        return float(1.0 - (lead_h / 120.0) * 0.5)
    if lead_h <= 168:
        return float(0.5 - (lead_h - 120) / 48 * 0.5)
    return 0.0


def _score_forward_severity(eff_wind_kt: float) -> float:
    if not np.isfinite(eff_wind_kt) or eff_wind_kt < 35:
        return 0.0
    return float(np.clip(((eff_wind_kt - 35) / 120) ** 2, 0.0, 1.0))


def _score_one_storm_point(
    fc_row: pd.Series, region_lat: float, region_lon: float,
) -> float:
    d = haversine_km(np.array([region_lat]), np.array([region_lon]),
                     float(fc_row["lat"]), float(fc_row["lon"]))[0]
    eff_radius = INFLUENCE_RADIUS_KM + float(fc_row["cone_radius_km"])
    if d > eff_radius:
        return 0.0
    if d <= TS_FORCE_RADIUS_KM:
        decay = 1.0
    else:
        decay = (eff_radius - d) / max(eff_radius - TS_FORCE_RADIUS_KM, 1.0)
        decay = float(np.clip(decay, 0.0, 1.0))
    eff_wind = float(fc_row["max_wind_kt"]) * decay
    severity = _score_forward_severity(eff_wind)
    confidence = _confidence_at_lead(float(fc_row["lead_h"]))
    return severity * confidence


def compute_forward_exposure(
    active_storms: pd.DataFrame,
) -> pd.DataFrame:
    """Score forward `disaster_index` per (region, forecast_date).

    Args:
        active_storms: DataFrame from
            `src.data.nhc_active_storms.fetch_active_storms_live()` or
            `simulate_from_hurdat(as_of_dt)`.

    Returns:
        long-form DataFrame:
            customer_country, customer_state, forecast_date,
            forward_disaster_index, contributing_storms, peak_lead_h
    """
    if active_storms.empty:
        return pd.DataFrame(columns=[
            "customer_country", "customer_state", "forecast_date",
            "forward_disaster_index", "contributing_storms", "peak_lead_h",
        ])

    as_of = pd.Timestamp(active_storms["advisory_dt"].iloc[0])
    df = active_storms.copy()
    df["forecast_dt"] = df["advisory_dt"] + pd.to_timedelta(df["lead_h"], unit="h")
    df["forecast_date"] = df["forecast_dt"].dt.normalize()

    regions: list[tuple[str, str | None, float, float]] = []
    for state in DATACO_US_STATES:
        if state in US_STATE_CENTROIDS:
            lat, lon = US_STATE_CENTROIDS[state]
            regions.append(("EE. UU.", state, lat, lon))
    for country, (lat, lon) in COUNTRY_CENTROIDS.items():
        state = "PR" if country == "Puerto Rico" else None
        regions.append((country, state, lat, lon))

    rows: list[dict] = []
    for country, state, lat, lon in regions:
        for date, day_df in df.groupby("forecast_date"):
            best_severity = 0.0
            contributing: list[str] = []
            peak_lead = None
            for _, fc in day_df.iterrows():
                sev = _score_one_storm_point(fc, lat, lon)
                if sev > 0:
                    if fc["name"] not in contributing:
                        contributing.append(fc["name"])
                    if sev > best_severity:
                        best_severity = sev
                        peak_lead = int(fc["lead_h"])
            if best_severity > 0:
                rows.append({
                    "customer_country": country,
                    "customer_state": state,
                    "forecast_date": pd.Timestamp(date),
                    "as_of_dt": as_of,
                    "forward_disaster_index": best_severity,
                    "contributing_storms": " + ".join(sorted(contributing)),
                    "peak_lead_h": peak_lead,
                })
    return pd.DataFrame(rows)


def lift_to_monthly(forward: pd.DataFrame,
                     apply_lag: bool = True) -> pd.DataFrame:
    """Roll daily forward exposure up to year_month for M4 integration.

    Takes the MAX forward_disaster_index across all days in the same
    month (per region) — operationally we care about the worst day in
    the month, since a single bad day saturates many of M4's downstream
    risks (cancellations spike, late deliveries spike).

    When `apply_lag=True` (default), the LAG_PROFILE from
    `storm_exposure.LAG_PROFILE` is applied: peak severity gets
    redistributed across landfall month + M+1..M+3. This mirrors the
    Tier-1 lag and is the right thing for M4 forward forecasting —
    revenue impact lags landfall by ~1 month.
    """
    if forward.empty:
        return pd.DataFrame(columns=[
            "customer_country", "customer_state", "year_month",
            "forward_disaster_index", "contributing_storms",
        ])
    f = forward.copy()
    f["year_month"] = f["forecast_date"].dt.to_period("M").dt.to_timestamp()
    g = (f.sort_values("forward_disaster_index", ascending=False)
           .groupby(["customer_country", "customer_state", "year_month"], dropna=False)
           .agg(
               forward_disaster_index=("forward_disaster_index", "max"),
               contributing_storms=("contributing_storms",
                                     lambda s: " + ".join(sorted(
                                         {x for line in s
                                          for x in line.split(" + ")}))),
           )
           .reset_index())
    if not apply_lag:
        return g
    # LAG: propagate the peak forward severity into M, M+1, M+2, M+3 per
    # LAG_PROFILE. Take MAX across overlapping contributions so multiple
    # storms feeding into the same month don't double-count.
    from src.features.storm_exposure import LAG_PROFILE
    rows: list[dict] = []
    for _, r in g.iterrows():
        for k, factor in LAG_PROFILE.items():
            target = (r["year_month"] + pd.DateOffset(months=k)).normalize()
            rows.append({
                "customer_country": r["customer_country"],
                "customer_state": r["customer_state"],
                "year_month": target,
                "forward_disaster_index": float(r["forward_disaster_index"]) * factor,
                "contributing_storms": r["contributing_storms"],
            })
    lagged = pd.DataFrame(rows)
    out = (lagged.sort_values("forward_disaster_index", ascending=False)
                  .groupby(["customer_country", "customer_state", "year_month"],
                            dropna=False)
                  .agg(forward_disaster_index=("forward_disaster_index", "max"),
                       contributing_storms=("contributing_storms",
                                             lambda s: " + ".join(sorted(
                                                 {x for line in s
                                                  for x in line.split(" + ")}))))
                  .reset_index())
    return out


def backtest_window(
    start: pd.Timestamp, end: pd.Timestamp, every_h: int = 24,
) -> pd.DataFrame:
    """Walk through a date window calling simulate_from_hurdat at every_h
    intervals and stacking the forward predictions.

    Useful for plotting "what would Tier 2 have said on each day of the
    2017 hurricane season?" without re-running the simulator manually.
    """
    from src.data.nhc_active_storms import simulate_from_hurdat
    out_frames: list[pd.DataFrame] = []
    cur = pd.Timestamp(start)
    end = pd.Timestamp(end)
    while cur <= end:
        active = simulate_from_hurdat(cur)
        if not active.empty:
            fwd = compute_forward_exposure(active)
            if not fwd.empty:
                out_frames.append(fwd)
        cur += pd.Timedelta(hours=every_h)
    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)


def run_demo() -> None:
    """Backtest demo: Tier 2's view of 2017 hurricane season (Aug–Sep)."""
    from src.data.nhc_active_storms import simulate_from_hurdat

    # Anchor demo: Sep 5, 2017 — 1 day before Irma hit PR
    log.info("Tier-2 backtest demo: as_of = 2017-09-05")
    active = simulate_from_hurdat(pd.Timestamp("2017-09-05 12:00"))
    fwd = compute_forward_exposure(active)
    log.info("forward_disaster_index rows: %d", len(fwd))
    if not fwd.empty:
        print("\n=== Top 20 forward predictions made on Sep 5, 2017 ===")
        cols = ["customer_country", "customer_state", "forecast_date",
                "forward_disaster_index", "contributing_storms", "peak_lead_h"]
        print(fwd.sort_values("forward_disaster_index", ascending=False)
                  .head(20)[cols].round(3).to_string(index=False))

    # Full Atlantic season walk: Aug 1 – Sep 25, 2017
    log.info("Full Aug–Sep 2017 backtest walk (every 24h)")
    big = backtest_window(pd.Timestamp("2017-08-01"),
                           pd.Timestamp("2017-09-25"), every_h=24)
    monthly = lift_to_monthly(big) if not big.empty else big
    out_daily = OUT_DIR / "forward_disaster_daily.parquet"
    out_monthly = OUT_DIR / "forward_disaster_monthly.parquet"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    big.to_parquet(out_daily, index=False)
    monthly.to_parquet(out_monthly, index=False)
    log.info("wrote %s (%d rows) and %s (%d rows)",
             out_daily, len(big), out_monthly, len(monthly))

    if not big.empty:
        print("\n=== Aug–Sep 2017 monthly roll-up (top 10 regions) ===")
        print(monthly.sort_values("forward_disaster_index", ascending=False)
                       .head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    run_demo()
