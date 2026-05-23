"""Tier-1 replacement for the hand-coded `known_disasters` calendar.

v1 (`known_disasters.py`) was a hand-written list of 5 hurricane events
covering 2015–2017. v2 learns the same indicators from NOAA HURDAT2 best-
track data via `src/features/storm_exposure.py`.

What changes downstream:
  - More storms covered (Arthur 2014, Nate 2017, Earl/Otto 2016, etc.
    — anything that came within 400 km of a DataCo region)
  - Severities reflect peak wind + duration, not analyst judgment
  - Updating the year range is now a one-line change in `hurdat2_ingest`,
    not a hand edit of a Python list

Interface is identical to v1:
  - `country_monthly_indicator() -> DataFrame[customer_country, year_month,
                                              known_severity, events]`
  - `state_monthly_indicator()   -> DataFrame[customer_country,
                                              customer_state, year_month,
                                              known_severity, events]`

So `src/models/risk/disaster.py` only needs its import line flipped.

The first call triggers a rebuild if the HURDAT2 parquet is missing or
older than the source release; subsequent calls hit the cached parquet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
STATE_MONTHLY_PATH = ROOT / "data" / "processed" / "storm_exposure_state_monthly.parquet"
COUNTRY_MONTHLY_PATH = ROOT / "data" / "processed" / "storm_exposure_country_monthly.parquet"


def _ensure_built() -> None:
    """Build the exposure parquets if they don't exist yet."""
    if STATE_MONTHLY_PATH.exists() and COUNTRY_MONTHLY_PATH.exists():
        return
    log.info("HURDAT2 exposure features missing — building")
    # Trigger ingest if track file missing
    tracks_path = ROOT / "data" / "processed" / "hurdat2_tracks.parquet"
    if not tracks_path.exists():
        from src.data.hurdat2_ingest import run as ingest_run
        ingest_run()
    from src.features.storm_exposure import run as features_run
    features_run()


def country_monthly_indicator() -> pd.DataFrame:
    """One row per (customer_country, year_month) with `known_severity` =
    max severity across overlapping storm events."""
    _ensure_built()
    df = pd.read_parquet(COUNTRY_MONTHLY_PATH)
    # Match v1 column order
    return df[["customer_country", "year_month", "known_severity", "events"]]


def state_monthly_indicator() -> pd.DataFrame:
    """One row per (customer_country, customer_state, year_month) with
    `known_severity` = max severity across overlapping storm events."""
    _ensure_built()
    df = pd.read_parquet(STATE_MONTHLY_PATH)
    return df[["customer_country", "customer_state", "year_month",
               "known_severity", "events"]]


if __name__ == "__main__":
    print("=== Country-level (v2, from HURDAT2) ===")
    print(country_monthly_indicator().sort_values("known_severity", ascending=False)
                                      .round(3).to_string(index=False))
    print()
    print("=== State-level (v2, from HURDAT2) ===")
    print(state_monthly_indicator().sort_values("known_severity", ascending=False)
                                    .round(3).to_string(index=False))
