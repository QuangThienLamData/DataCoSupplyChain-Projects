"""Tier-3 NOAA CPC seasonal hurricane outlook → per-region baseline multiplier.

Lead time: **1–6 months** (issued May & August, covers Atlantic season
June-November). Compared to Tier 1 (post-event, weeks-lagged) and Tier 2
(5-7 day pre-event), Tier 3 is the longest lead but lowest specificity:
it tells you "this season will be more active than average" but not
"hurricane X will hit Florida on date Y." Useful for capacity planning,
inventory pre-positioning, staffing buffers — not for daily operations.

Mechanism
---------
NOAA classifies each season as **Below Normal / Near Normal / Above Normal**
(occasionally "Extremely Active"). We convert each category into a
**multiplier** on the climatological-baseline disaster_index for the
hurricane-season months (May-Nov):

    multiplier:
        Below Normal      → 0.7
        Near Normal       → 1.0
        Above Normal      → 1.3
        Extremely Active  → 1.5

Outside hurricane season (Dec-Apr) the multiplier is 1.0.

The multiplier is applied to the **climatology baseline**, not to Tier 2.
Tier 2 (specific named storms) takes over once it fires:

    forward_disaster_index =
        max(tier3_multiplier × climatology_baseline,
            tier2_specific_storms)

Sources
-------
Outlooks for 2014-2017 are hardcoded from the NOAA CPC May & August
bulletins (cross-referenced against Wikipedia season pages). To extend to
a new year, just add the dict entry — no other code changes.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]

# Category → multiplier on climatological baseline disaster_index
CATEGORY_MULTIPLIERS: dict[str, float] = {
    "Below Normal": 0.70,
    "Near Normal": 1.00,
    "Above Normal": 1.30,
    "Extremely Active": 1.50,
}

# Hurricane season window (multiplier applies only inside this)
SEASON_MONTHS: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11)

# NOAA CPC Atlantic hurricane season outlooks 2014-2017
# (sources: NOAA CPC May & August bulletins; cross-checked against
# Wikipedia year pages — categorical interpretation by author of this module)
SEASONAL_OUTLOOKS: dict[int, dict] = {
    2014: {
        "may_category": "Below Normal",
        "may_predicted": {"named": (8, 13), "hurricanes": (3, 6), "major": (1, 2)},
        "aug_category": "Below Normal",
        "aug_predicted": {"named": (7, 12), "hurricanes": (3, 6), "major": (0, 2)},
        "actual": {"named": 8, "hurricanes": 6, "major": 2},
    },
    2015: {
        "may_category": "Below Normal",
        "may_predicted": {"named": (6, 11), "hurricanes": (3, 6), "major": (0, 2)},
        "aug_category": "Below Normal",
        "aug_predicted": {"named": (6, 10), "hurricanes": (1, 4), "major": (0, 1)},
        "actual": {"named": 11, "hurricanes": 4, "major": 2},
    },
    2016: {
        "may_category": "Near Normal",
        "may_predicted": {"named": (10, 16), "hurricanes": (4, 8), "major": (1, 4)},
        "aug_category": "Above Normal",
        "aug_predicted": {"named": (12, 17), "hurricanes": (5, 8), "major": (2, 4)},
        "actual": {"named": 15, "hurricanes": 7, "major": 4},
    },
    2017: {
        "may_category": "Above Normal",
        "may_predicted": {"named": (11, 17), "hurricanes": (5, 9), "major": (2, 4)},
        "aug_category": "Extremely Active",
        "aug_predicted": {"named": (14, 19), "hurricanes": (5, 9), "major": (2, 5)},
        "actual": {"named": 17, "hurricanes": 10, "major": 6},
    },
    # 2018: NOAA CPC issues the first outlook in May. Standing at Jan 2018
    # we don't yet have one — default to Near Normal until May. (NOAA's
    # actual May 2018 call was "Near or Above Normal"; we use Near Normal
    # as the pre-outlook prior.)
    2018: {
        "may_category": "Near Normal",
        "may_predicted": {"named": (10, 16), "hurricanes": (5, 9), "major": (1, 4)},
        "aug_category": "Near Normal",
        "aug_predicted": {"named": (9, 13), "hurricanes": (4, 7), "major": (0, 2)},
        "actual": {"named": 15, "hurricanes": 8, "major": 2},  # actual 2018 season
        "note": "Default pre-outlook stub — May/Aug 2018 entries not yet released as of as_of=2018-01-31",
    },
}


def get_multiplier(year: int, month: int, as_of: str = "aug") -> float:
    """Return Tier-3 multiplier for (year, month).

    Args:
        year, month: target month
        as_of: 'may' or 'aug' — which outlook update to use. 'aug' is
               the more accurate / refined one; 'may' for the very-early
               planning view.

    Returns 1.0 outside hurricane season or if year isn't in our table.
    """
    if month not in SEASON_MONTHS:
        return 1.0
    info = SEASONAL_OUTLOOKS.get(year)
    if not info:
        return 1.0
    cat_key = "aug_category" if as_of == "aug" else "may_category"
    cat = info.get(cat_key, "Near Normal")
    return CATEGORY_MULTIPLIERS.get(cat, 1.0)


def climatology_baseline(
    historical_disaster: pd.DataFrame,
    value_col: str = "known_severity",
) -> pd.DataFrame:
    """Per (customer_country, calendar_month) mean of `value_col` across
    the years in the historical input.

    Args:
        historical_disaster: Tier-1 country index DataFrame with cols
            customer_country, year_month, and one of disaster_index or
            known_severity.
        value_col: column to average. Defaults to "known_severity" which
            is the HURDAT2-only signal — off-season baseline is ~0,
            hurricane-season baseline reflects real storm climatology.
            Switch to "disaster_index" only if you also want the anomaly
            proxy in the baseline (it fires on non-storm events, so the
            baseline becomes ~0.5 year-round — usually not what you want
            for Tier-3 planning).

    Returns DataFrame with cols customer_country, month, climatology_baseline.
    """
    if value_col not in historical_disaster.columns:
        value_col = "disaster_index"
    df = historical_disaster.copy()
    df["month"] = pd.to_datetime(df["year_month"]).dt.month
    out = (df.groupby(["customer_country", "month"])[value_col]
              .mean().reset_index()
              .rename(columns={value_col: "climatology_baseline"}))
    return out


def apply_outlook(
    target_months: list,
    historical_disaster: pd.DataFrame,
    as_of: str = "aug",
) -> pd.DataFrame:
    """For each (country, target_month), multiply climatology baseline by
    the Tier-3 seasonal multiplier.

    Returns long-form DataFrame:
        customer_country, year_month, multiplier, climatology_baseline,
        tier3_baseline (= mult × climatology), season_category.
    """
    climo = climatology_baseline(historical_disaster)
    rows: list[dict] = []
    for ym in target_months:
        ym = pd.Timestamp(ym)
        mult = get_multiplier(ym.year, ym.month, as_of=as_of)
        cat_key = "aug_category" if as_of == "aug" else "may_category"
        cat = SEASONAL_OUTLOOKS.get(ym.year, {}).get(cat_key, "Near Normal")
        sub = climo[climo["month"].eq(ym.month)]
        for _, r in sub.iterrows():
            rows.append({
                "customer_country": r["customer_country"],
                "year_month": ym,
                "multiplier": mult,
                "climatology_baseline": float(r["climatology_baseline"]),
                "tier3_baseline": float(r["climatology_baseline"] * mult),
                "season_category": cat,
            })
    return pd.DataFrame(rows)


def combine_with_tier2(tier3: pd.DataFrame, tier2: pd.DataFrame) -> pd.DataFrame:
    """For each (country, year_month), forward_disaster_index =
    max(tier3_baseline, tier2_forward_disaster_index).

    Tier 2 wins when specific storms are forecast; Tier 3 fills in
    capacity-planning baseline when Tier 2 is silent (no active storms).
    """
    if tier2.empty:
        out = tier3.copy()
        out["tier2_forward"] = 0.0
        out["forward_disaster_combined"] = out["tier3_baseline"]
        return out
    t2 = tier2[tier2["customer_state"].isna()].rename(
        columns={"forward_disaster_index": "tier2_forward"})[
        ["customer_country", "year_month", "tier2_forward"]]
    pr_rows = tier2[tier2["customer_country"].eq("Puerto Rico")].rename(
        columns={"forward_disaster_index": "tier2_forward"})[
        ["customer_country", "year_month", "tier2_forward"]]
    t2_all = pd.concat([t2, pr_rows], ignore_index=True).groupby(
        ["customer_country", "year_month"], as_index=False)["tier2_forward"].max()
    out = tier3.merge(t2_all, on=["customer_country", "year_month"], how="left")
    out["tier2_forward"] = out["tier2_forward"].fillna(0.0)
    out["forward_disaster_combined"] = np.maximum(
        out["tier3_baseline"], out["tier2_forward"])
    return out


def run_demo() -> None:
    """Demo: 2017 hurricane season seen from May vs August outlook."""
    # Build climatology from current Tier-1 country index
    cty = pd.read_parquet(ROOT / "forecasts" / "m3d_disaster_country.parquet")
    climo = climatology_baseline(cty)
    print("=== Climatology baseline (mean disaster_index by country × month) ===")
    pivot = climo.pivot(index="customer_country", columns="month",
                         values="climatology_baseline").round(3)
    print(pivot.to_string())

    target = pd.date_range("2017-05-01", "2017-12-01", freq="MS")
    print("\n=== Tier-3 forward baseline (May outlook) — 2017 ===")
    may = apply_outlook(list(target), cty, as_of="may")
    print(may[may["customer_country"].isin(["EE. UU.", "Puerto Rico", "Cuba"])]
            .round(3).to_string(index=False))
    print("\n=== Tier-3 forward baseline (August outlook) — 2017 ===")
    aug = apply_outlook(list(target), cty, as_of="aug")
    print(aug[aug["customer_country"].isin(["EE. UU.", "Puerto Rico", "Cuba"])]
            .round(3).to_string(index=False))

    print("\n=== Outlook summary across 2014-2017 ===")
    for y, info in SEASONAL_OUTLOOKS.items():
        print(f"  {y}: May='{info['may_category']}' "
              f"(mult={CATEGORY_MULTIPLIERS[info['may_category']]}), "
              f"Aug='{info['aug_category']}' "
              f"(mult={CATEGORY_MULTIPLIERS[info['aug_category']]}), "
              f"actual = {info['actual']['named']} named storms, "
              f"{info['actual']['hurricanes']} hurricanes")


if __name__ == "__main__":
    run_demo()
