"""Download + parse NOAA HURDAT2 (Atlantic + East-Pacific basins).

HURDAT2 is NOAA NHC's official best-track archive: 6-hourly position, max
wind, pressure for every named tropical/subtropical cyclone. It powers Tier
1 of our disaster-prediction pipeline — replaces the 5-event hand-coded
calendar with a learned exposure feature over every storm 2014–2017
(extensible to any year).

Source page  : https://www.nhc.noaa.gov/data/
File format  : https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atl-1851-2021.pdf

Each storm starts with a 3-field header line:
    AL092017,                 IRMA,     67,
followed by N data lines of 21 comma-separated fields:
    yyyymmdd, hhmm, record_id, status, lat, lon, wind_kt, pressure, ...

We keep the columns needed for downstream exposure scoring:
  storm_id, basin, year, name, point_dt, status, lat, lon, max_wind_kt,
  min_pressure_mb, landfall_flag.
"""
from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "hurdat2"
PROCESSED_PATH = ROOT / "data" / "processed" / "hurdat2_tracks.parquet"

# Pinned release (Feb 2026); covers 1851/1949 → 2025.
ATLANTIC_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"
NEPAC_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2025-02272026.txt"

# Lines starting with these 2 chars are storm-header rows
HEADER_RE = re.compile(r"^(AL|EP|CP)\d{6},")


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log.info("cached %s (%d KB)", dest.name, dest.stat().st_size // 1024)
        return dest
    log.info("downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    log.info("  → %s (%d KB)", dest, dest.stat().st_size // 1024)
    return dest


def _parse_latlon(s: str) -> float:
    """'16.1N' -> 16.1; '31.4W' -> -31.4."""
    s = s.strip()
    sign = -1.0 if s[-1] in "SW" else 1.0
    return sign * float(s[:-1])


def parse_file(path: Path) -> pd.DataFrame:
    """Parse one HURDAT2 file into a long-form DataFrame of track points."""
    rows: list[dict] = []
    cur_id: str | None = None
    cur_name: str | None = None
    cur_basin: str | None = None
    cur_year: int | None = None

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if HEADER_RE.match(line):
                fields = [f.strip() for f in line.split(",")]
                cur_id = fields[0]
                cur_name = fields[1]
                cur_basin = cur_id[:2]   # AL / EP / CP
                cur_year = int(cur_id[4:8])
                continue
            # Data line
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < 8:
                continue
            try:
                yyyymmdd, hhmm = fields[0], fields[1]
                record_id, status = fields[2], fields[3]
                lat, lon = _parse_latlon(fields[4]), _parse_latlon(fields[5])
                wind = int(fields[6]) if fields[6] not in ("-999", "") else None
                pres = int(fields[7]) if fields[7] not in ("-999", "") else None
            except (ValueError, IndexError):
                continue
            rows.append({
                "storm_id": cur_id,
                "basin": cur_basin,
                "year": cur_year,
                "name": cur_name,
                "point_dt": pd.Timestamp(
                    f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]} "
                    f"{hhmm[0:2]}:{hhmm[2:4]}"),
                "record_id": record_id,
                "status": status,
                "lat": lat,
                "lon": lon,
                "max_wind_kt": wind,
                "min_pressure_mb": pres,
                "landfall_flag": record_id == "L",
            })
    return pd.DataFrame(rows)


def run(years: tuple[int, int] = (2014, 2017),
        keep_all_years: bool = False) -> pd.DataFrame:
    """Download + parse Atlantic + EP HURDAT2; trim to year window unless
    keep_all_years=True. Writes a clean parquet to data/processed/."""
    atl = _download(ATLANTIC_URL, RAW_DIR / Path(ATLANTIC_URL).name)
    epac = _download(NEPAC_URL, RAW_DIR / Path(NEPAC_URL).name)

    log.info("parsing Atlantic basin")
    df_atl = parse_file(atl)
    log.info("  %d points, %d storms", len(df_atl), df_atl["storm_id"].nunique())

    log.info("parsing E-Pacific basin")
    df_ep = parse_file(epac)
    log.info("  %d points, %d storms", len(df_ep), df_ep["storm_id"].nunique())

    df = pd.concat([df_atl, df_ep], ignore_index=True)
    if not keep_all_years:
        df = df[(df["year"] >= years[0]) & (df["year"] <= years[1])]
        log.info("trimmed to %d–%d → %d points, %d storms",
                 years[0], years[1], len(df), df["storm_id"].nunique())

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    log.info("wrote %s (%d rows)", PROCESSED_PATH, len(df))
    return df


if __name__ == "__main__":
    df = run()
    print("\nStorms in window (top 20 by peak wind):")
    peak = (df.groupby(["storm_id", "year", "name"])["max_wind_kt"].max()
              .reset_index().sort_values("max_wind_kt", ascending=False).head(20))
    print(peak.to_string(index=False))
