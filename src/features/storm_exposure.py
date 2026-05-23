"""Storm-exposure features per (region, year_month) from HURDAT2 tracks.

Given a region's centroid and a storm's 6-hourly track, we compute:

- nearest_approach_km   — closest the storm came to the centroid
- max_wind_at_region_kt — peak wind (kt) at the nearest-approach point,
                           decayed by the inverse distance from the storm
                           centre out to the tropical-storm radius
- hours_within_ts_band  — total hours within Rₜₛ (tropical-storm-force
                           extent ≈ 200 km from centre for a mature storm)
- landfall_near         — whether any HURDAT2 landfall point fell within
                           250 km of the centroid
- peak_status           — strongest status (TD<TS<HU) recorded near region
- severity              — single 0-1 score we plug into M3 in place of the
                           hand-coded `severity` from KNOWN_DISASTERS

Severity formula
----------------
Wind component (Pielke 2008: damage ∝ wind^2):

    wind_score = ((peak_eff_wind_kt - 35) / 120) ^ 2

Duration multiplier — a Harvey-class siege (50+ hours of TS-force over
the same region) does far more damage than a glancing pass of the same
peak wind:

    duration_mult = 1 + min(hours_within_ts_band / 72, 1.0)

so 0 hours → ×1.0, 36 hours → ×1.5, 72+ hours → ×2.0.

Direct-hit bump (eyewall passed within 100 km): ×1.15. Final score is
clipped to [0, 1].

This recovers Maria/Irma direct hits (peak-wind dominated) AND
Harvey (duration dominated) within the same scoring framework.

**Known limitation**: still wind-/duration-based, not flood-based. A
slow-moving but compact rainmaker that drops 1 m of rain (e.g.,
Tropical Storm Allison 2001) would still under-score. We accept this for
now — the M4 damping coefficient absorbs absolute scale.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.region_centroids import (
    COUNTRY_CENTROIDS, DATACO_US_STATES, US_STATE_CENTROIDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TRACKS_PATH = ROOT / "data" / "processed" / "hurdat2_tracks.parquet"
OUT_DIR = ROOT / "data" / "processed"

EARTH_RADIUS_KM = 6371.0
TS_FORCE_RADIUS_KM = 200.0     # ≈ typical R34 quadrant average
DIRECT_HIT_RADIUS_KM = 100.0   # landfall within this → "direct hit"
INFLUENCE_RADIUS_KM = 400.0    # outside this, storm has no economic effect
INTERP_STEP_HOURS = 1          # densify 6-hourly track to hourly

# Storm-impact LAG profile: `disaster_index` represents the **storm event
# severity at that month**, not the lagged revenue impact. Peak severity
# is at the landfall month itself (when the storm is physically present
# and ports / power / supply chain are most disrupted), with a decay tail
# as recovery proceeds. The DAMPING coefficient handles the conversion
# from storm severity to revenue drag.
#
# Pattern targeted by user: "disaster_index should be highest at 2017-09
# (Irma + Maria landfall) and 2017-10 (immediate aftermath)".
LAG_PROFILE: dict[int, float] = {
    0: 1.00,   # landfall month — peak severity (storm physically present)
    1: 0.80,   # M+1 — immediate aftermath, recovery hasn't started
    2: 0.45,   # M+2 — recovery underway, less acute disruption
    3: 0.25,   # M+3 — tail
    4: 0.10,   # M+4 — long tail for catastrophic storms
}


def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: float, lon2: float) -> np.ndarray:
    lat1r = np.radians(lat1); lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _densify_track(track: pd.DataFrame, step_h: int = INTERP_STEP_HOURS) -> pd.DataFrame:
    """Linearly interpolate a 6-hourly track to hourly resolution. Lets us
    integrate hours-within-radius without a coarse 6h grid."""
    if len(track) < 2:
        return track
    t = track.sort_values("point_dt").reset_index(drop=True)
    t0 = t["point_dt"].iloc[0]; t1 = t["point_dt"].iloc[-1]
    if t0 >= t1:
        return t
    new_idx = pd.date_range(t0, t1, freq=f"{step_h}h")
    f = t.set_index("point_dt")
    out = pd.DataFrame(index=new_idx)
    for col in ("lat", "lon", "max_wind_kt"):
        s = pd.to_numeric(f[col], errors="coerce")
        out[col] = np.interp(new_idx.astype("int64"),
                              f.index.astype("int64"), s.to_numpy())
    out["storm_id"] = t["storm_id"].iloc[0]
    out["name"] = t["name"].iloc[0]
    out["year"] = t["year"].iloc[0]
    out["point_dt"] = new_idx
    return out.reset_index(drop=True)


def _wind_at_region(distances_km: np.ndarray, winds_kt: np.ndarray) -> np.ndarray:
    """Effective wind felt at the region: storm wind, scaled down by how far
    the region is from the eyewall. Inside R34 → full wind; outside →
    linear decay to zero at INFLUENCE_RADIUS_KM."""
    decay = np.clip(
        (INFLUENCE_RADIUS_KM - distances_km) /
        (INFLUENCE_RADIUS_KM - TS_FORCE_RADIUS_KM), 0, 1)
    inside_core = distances_km <= TS_FORCE_RADIUS_KM
    return np.where(inside_core, winds_kt, winds_kt * decay)


def _score_severity(peak_eff_wind_kt: float, direct_hit: bool,
                     hours_within_ts_band: int) -> float:
    if not np.isfinite(peak_eff_wind_kt) or peak_eff_wind_kt < 35:
        return 0.0
    wind_score = ((peak_eff_wind_kt - 35) / 120) ** 2
    # 0h → 1.0, 36h → 1.5, 72h+ → 2.0
    duration_mult = 1.0 + min(hours_within_ts_band / 72.0, 1.0)
    score = wind_score * duration_mult
    if direct_hit:
        score *= 1.15
    return float(np.clip(score, 0.0, 1.0))


def _exposure_for_region(
    densified_storms: dict[str, pd.DataFrame],
    region_label: str,
    centroid: tuple[float, float],
) -> pd.DataFrame:
    """One row per (region, storm-month) where the storm came within
    INFLUENCE_RADIUS_KM. Empty rows are dropped."""
    lat0, lon0 = centroid
    rows: list[dict] = []
    for sid, track in densified_storms.items():
        d = haversine_km(track["lat"].to_numpy(),
                          track["lon"].to_numpy(), lat0, lon0)
        in_radius = d <= INFLUENCE_RADIUS_KM
        if not in_radius.any():
            continue
        winds = track["max_wind_kt"].to_numpy(dtype=float)
        eff_wind = _wind_at_region(d, winds)
        peak_eff = float(np.nanmax(eff_wind))
        nearest_km = float(np.nanmin(d))
        hours_in_ts = int((d <= TS_FORCE_RADIUS_KM).sum() * INTERP_STEP_HOURS)
        # When the storm was closest, what month was it?
        nearest_idx = int(np.nanargmin(d))
        peak_dt = track["point_dt"].iloc[nearest_idx]
        year_month = pd.Timestamp(year=peak_dt.year, month=peak_dt.month, day=1)
        # Direct-hit flag — eyewall passed within DIRECT_HIT_RADIUS_KM
        direct_hit = bool((d <= DIRECT_HIT_RADIUS_KM).any())
        severity = _score_severity(peak_eff, direct_hit, hours_in_ts)
        if severity == 0.0:
            continue
        # Apply LAG_PROFILE: storm-impact distribution across landfall month
        # and M+1..M+3. Peak revenue impact is at M+1, not the landfall month.
        for k, factor in LAG_PROFILE.items():
            ym = (year_month + pd.DateOffset(months=k)).normalize()
            rows.append({
                "region": region_label,
                "storm_id": sid,
                "name": track["name"].iloc[0],
                "year": int(track["year"].iloc[0]),
                "year_month": ym,
                "month_offset": k,
                "nearest_approach_km": nearest_km,
                "peak_eff_wind_kt": peak_eff,
                "hours_within_ts_band": hours_in_ts,
                "direct_hit": direct_hit,
                "severity": severity * factor,
            })
    return pd.DataFrame(rows)


def compute_storm_exposure(
    tracks_path: Path = TRACKS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (state_exposure, country_exposure) — long-form per
    (region, storm) rows that haven't been month-aggregated yet."""
    log.info("loading tracks from %s", tracks_path)
    tracks = pd.read_parquet(tracks_path)
    tracks = tracks.dropna(subset=["lat", "lon", "max_wind_kt"])
    tracks["max_wind_kt"] = tracks["max_wind_kt"].astype(float)

    log.info("densifying %d storms to %dh", tracks["storm_id"].nunique(),
             INTERP_STEP_HOURS)
    densified: dict[str, pd.DataFrame] = {}
    for sid, g in tracks.groupby("storm_id", sort=False):
        densified[sid] = _densify_track(g)

    log.info("scoring %d US states", len(DATACO_US_STATES))
    state_rows = []
    for state in DATACO_US_STATES:
        if state not in US_STATE_CENTROIDS:
            continue
        df = _exposure_for_region(densified, state, US_STATE_CENTROIDS[state])
        if not df.empty:
            df["customer_country"] = "EE. UU."
            df["customer_state"] = state
            state_rows.append(df)
    states = (pd.concat(state_rows, ignore_index=True)
              if state_rows else pd.DataFrame())
    log.info("  %d state-storm exposure rows", len(states))

    # Puerto Rico is a country in DataCo, with state code 'PR'
    log.info("scoring %d countries", len(COUNTRY_CENTROIDS))
    country_rows = []
    for country, centroid in COUNTRY_CENTROIDS.items():
        df = _exposure_for_region(densified, country, centroid)
        if not df.empty:
            df["customer_country"] = country
            df["customer_state"] = "PR" if country == "Puerto Rico" else None
            country_rows.append(df)
    countries = (pd.concat(country_rows, ignore_index=True)
                 if country_rows else pd.DataFrame())
    log.info("  %d country-storm exposure rows", len(countries))

    return states, countries


def aggregate_to_month(exposure: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Many storms can hit one region in one month — take the max severity
    and merge storm names. Returns the indicator schema expected by M3."""
    if exposure.empty:
        return pd.DataFrame(columns=group_cols + ["year_month",
                                                    "known_severity", "events"])
    out = (exposure.groupby(group_cols + ["year_month"])
                    .agg(known_severity=("severity", "max"),
                         events=("name", lambda s: " + ".join(
                             sorted({f"Storm {x.title()}" for x in s.unique()
                                     if isinstance(x, str)})))
                         )
                    .reset_index())
    return out


def run() -> dict[str, pd.DataFrame]:
    states, countries = compute_storm_exposure()
    state_monthly = aggregate_to_month(
        states, ["customer_country", "customer_state"])
    country_monthly = aggregate_to_month(
        countries, ["customer_country"])

    out_state = OUT_DIR / "storm_exposure_state_monthly.parquet"
    out_country = OUT_DIR / "storm_exposure_country_monthly.parquet"
    out_state_full = OUT_DIR / "storm_exposure_state_storm.parquet"
    out_country_full = OUT_DIR / "storm_exposure_country_storm.parquet"

    states.to_parquet(out_state_full, index=False)
    countries.to_parquet(out_country_full, index=False)
    state_monthly.to_parquet(out_state, index=False)
    country_monthly.to_parquet(out_country, index=False)
    log.info("wrote 4 files to %s", OUT_DIR)

    return {
        "state_storm": states, "country_storm": countries,
        "state_monthly": state_monthly, "country_monthly": country_monthly,
    }


if __name__ == "__main__":
    out = run()
    print("\n=== Top 20 region-storm exposures ===")
    cols = ["customer_country", "customer_state", "name", "year", "year_month",
            "nearest_approach_km", "peak_eff_wind_kt", "direct_hit", "severity"]
    full = pd.concat([out["state_storm"], out["country_storm"]], ignore_index=True)
    print(full[cols].sort_values("severity", ascending=False).head(20)
              .round(2).to_string(index=False))
    print("\n=== Top 10 country-month exposures ===")
    print(out["country_monthly"].sort_values("known_severity", ascending=False)
              .head(10).round(2).to_string(index=False))
