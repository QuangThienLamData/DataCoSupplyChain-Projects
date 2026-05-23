"""NHC active-storms client — Tier 2 of the storm-prediction roadmap.

Two operational modes, same output schema:

  fetch_active_storms_live()         — calls NHC CurrentStorms.json. Returns
                                         the live forecast track + intensity for
                                         each storm NHC is currently watching.
  simulate_from_hurdat(as_of_dt)     — for backtesting Tier 2 on history: given
                                         a date in the past, builds the same
                                         schema by reading the *next* 5 days of
                                         the actual HURDAT2 track. Lets us
                                         ask "what would Tier 2 have predicted
                                         5 days before Maria hit PR?"

Output schema (long-form DataFrame):
  storm_id, name, basin, advisory_dt, lead_h, lat, lon, max_wind_kt,
  cone_radius_km

`advisory_dt` is the issue time of the forecast (i.e., "today"); `lead_h` is
hours into the future from `advisory_dt`.

Cone radii are the NHC operational track-error climatology (≈67% confidence
envelope). They widen with lead time because forecasts are physically less
certain further out.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TRACKS_PATH = ROOT / "data" / "processed" / "hurdat2_tracks.parquet"
NHC_CURRENT_URL = "https://www.nhc.noaa.gov/CurrentStorms.json"

# NHC operational ~67% confidence track-error envelope (NHC verification
# program, recent 5-year window). Lead time hours → radius km.
NHC_CONE_RADII_KM: dict[int, float] = {
    0:    0.0,
    12:   48.0,
    24:   74.0,
    36:  102.0,
    48:  130.0,
    72:  185.0,
    96:  232.0,
    120: 324.0,
}

LEAD_TIMES_H: tuple[int, ...] = (12, 24, 36, 48, 72, 96, 120)


def _interp_cone_km(lead_h: float) -> float:
    keys = sorted(NHC_CONE_RADII_KM.keys())
    return float(np.interp(lead_h, keys, [NHC_CONE_RADII_KM[k] for k in keys]))


def fetch_active_storms_live(timeout_s: int = 15) -> pd.DataFrame:
    """Call NHC CurrentStorms.json and parse into the standard schema.

    Returns empty DataFrame when no storms are active. Raises on network or
    JSON errors so the caller can decide whether to swallow them."""
    log.info("fetching live active-storm list from NHC")
    with urllib.request.urlopen(NHC_CURRENT_URL, timeout=timeout_s) as r:
        payload = json.loads(r.read().decode("utf-8"))

    storms = payload.get("activeStorms") or []
    if not storms:
        log.info("no active storms")
        return pd.DataFrame(columns=[
            "storm_id", "name", "basin", "advisory_dt", "lead_h",
            "lat", "lon", "max_wind_kt", "cone_radius_km",
        ])

    rows: list[dict] = []
    for s in storms:
        sid = s.get("id") or s.get("binNumber") or s.get("name", "UNKNOWN")
        name = s.get("name", "UNKNOWN")
        basin = s.get("basin", "")[:2].upper() or "AL"
        advisory_dt = pd.Timestamp(
            s.get("lastUpdate") or s.get("issuanceTime") or s.get("dateString"),
            tz="UTC").tz_convert(None)
        # NHC forecast track lives at s["forecastTrack"]["points"] in 2025+
        track = (s.get("forecastTrack", {}).get("points")
                 or s.get("forecast", []) or [])
        for p in track:
            try:
                lead = int(p.get("fcstPeriod") or p.get("tauHours") or p["lead_h"])
                lat = float(p["latitude"]); lon = float(p["longitude"])
                wind = float(p.get("maxWindSpeedMph", 0)) * 0.868976  # mph→kt
                if wind == 0:
                    wind = float(p.get("maxWindSpeedKnots", 0))
            except (KeyError, TypeError, ValueError):
                continue
            rows.append({
                "storm_id": sid, "name": name, "basin": basin,
                "advisory_dt": advisory_dt, "lead_h": lead,
                "lat": lat, "lon": lon, "max_wind_kt": wind,
                "cone_radius_km": _interp_cone_km(lead),
            })
    return pd.DataFrame(rows)


def simulate_from_hurdat(
    as_of_dt: pd.Timestamp,
    lookahead_h: int = 120,
    active_window_h: int = 12,
) -> pd.DataFrame:
    """Build a Tier-2-shaped forecast from HURDAT2 history.

    Args:
        as_of_dt: the "now" pretend-time. Storms that had a HURDAT2 track
            point within ±active_window_h of this time are considered
            active.
        lookahead_h: how many hours ahead to publish forecast points for.
        active_window_h: half-width of the "is this storm currently
            active" window around as_of_dt.

    The output mimics what NHC would have published at as_of_dt: forecast
    track positions at the standard lead times, with the NHC climatological
    cone radii bolted on.
    """
    if not TRACKS_PATH.exists():
        raise FileNotFoundError("HURDAT2 tracks not built — run src.data.hurdat2_ingest")
    tracks = pd.read_parquet(TRACKS_PATH)
    tracks = tracks.dropna(subset=["lat", "lon", "max_wind_kt"]).copy()
    tracks["max_wind_kt"] = tracks["max_wind_kt"].astype(float)

    as_of = pd.Timestamp(as_of_dt)
    win_lo = as_of - pd.Timedelta(hours=active_window_h)
    win_hi = as_of + pd.Timedelta(hours=active_window_h)

    rows: list[dict] = []
    for sid, g in tracks.groupby("storm_id", sort=False):
        g = g.sort_values("point_dt")
        # Active = had a track point within the ±window
        active = g[(g["point_dt"] >= win_lo) & (g["point_dt"] <= win_hi)]
        if active.empty:
            continue
        name = g["name"].iloc[0]
        basin = g["basin"].iloc[0]
        # For each requested lead time, find the closest future track point
        for lead in LEAD_TIMES_H:
            if lead > lookahead_h:
                break
            target_dt = as_of + pd.Timedelta(hours=lead)
            future = g[g["point_dt"] >= as_of]
            if future.empty:
                # Storm dissipated before our as_of — skip future leads
                continue
            # Linear interpolation to the requested lead time
            idx = future["point_dt"].searchsorted(target_dt)
            if idx == 0:
                # target_dt is before the first remaining future point
                pt = future.iloc[0]
                lat, lon, wind = pt["lat"], pt["lon"], pt["max_wind_kt"]
            elif idx >= len(future):
                # target_dt past the storm's lifetime
                pt = future.iloc[-1]
                lat, lon, wind = pt["lat"], pt["lon"], pt["max_wind_kt"]
            else:
                a = future.iloc[idx - 1]; b = future.iloc[idx]
                w = ((target_dt - a["point_dt"]).total_seconds()
                     / max((b["point_dt"] - a["point_dt"]).total_seconds(), 1))
                lat = a["lat"] + w * (b["lat"] - a["lat"])
                lon = a["lon"] + w * (b["lon"] - a["lon"])
                wind = a["max_wind_kt"] + w * (b["max_wind_kt"] - a["max_wind_kt"])
            rows.append({
                "storm_id": sid, "name": name, "basin": basin,
                "advisory_dt": as_of, "lead_h": lead,
                "lat": float(lat), "lon": float(lon),
                "max_wind_kt": float(wind),
                "cone_radius_km": _interp_cone_km(lead),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Live mode (no storms = empty) ===")
    try:
        live = fetch_active_storms_live()
        print(f"{len(live)} rows")
        if not live.empty:
            print(live.head(20).to_string(index=False))
    except Exception as e:
        print(f"live fetch failed: {e}")
    print()
    print("=== Backtest mode: as of Sep 5, 2017 (1 day before Irma hit PR) ===")
    sim = simulate_from_hurdat(pd.Timestamp("2017-09-05 12:00"))
    print(f"{len(sim)} forecast points across {sim['name'].nunique() if not sim.empty else 0} storms")
    if not sim.empty:
        print(sim[["name", "lead_h", "lat", "lon", "max_wind_kt", "cone_radius_km"]]
                .round(2).to_string(index=False))
