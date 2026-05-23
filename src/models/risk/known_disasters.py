"""Known disaster events affecting the dataset's customer base (2015–2017).

The DataCo dataset's customers are concentrated in two `customer_country`
labels: 'EE. UU.' (USA) and 'Puerto Rico'. The major economic-disruption
events in this window are the 2017 Atlantic hurricane season, plus
Hurricane Matthew (2016). We encode these as a monthly indicator per
customer_country and per customer_state.

Source: NOAA NHC reports; cross-referenced against state-level damage
tallies. Severity is on a 0-1 scale calibrated to estimated revenue
disruption (1.0 = total monthly disruption, 0.5 = ~50% of activity lost).

This calendar is **input** to the disaster index — it gets fused with the
data-driven anomaly z-scores so we don't double-count.
"""
from __future__ import annotations

import pandas as pd

# Customer-country labels in the DataCo dataset
COUNTRY_US = "EE. UU."
COUNTRY_PR = "Puerto Rico"

# (event_name, year, month, country, state_codes [or '*' for whole country],
#  severity ∈ [0, 1])
KNOWN_DISASTERS: list[dict] = [
    # 2015
    {"event": "Hurricane Joaquin", "year": 2015, "month": 10,
     "country": COUNTRY_US,
     "states": ["NC", "SC", "VA"], "severity": 0.30},

    # 2016 — Hurricane Matthew, made landfall Oct 7-10, 2016
    {"event": "Hurricane Matthew", "year": 2016, "month": 10,
     "country": COUNTRY_US,
     "states": ["FL", "GA", "SC", "NC", "VA"], "severity": 0.45},
    {"event": "Hurricane Matthew", "year": 2016, "month": 10,
     "country": COUNTRY_PR,
     "states": ["PR"], "severity": 0.20},

    # 2017 — Hurricane Harvey (Aug 25 – Sep 2, 2017), Texas / Louisiana
    {"event": "Hurricane Harvey", "year": 2017, "month": 8,
     "country": COUNTRY_US,
     "states": ["TX", "LA"], "severity": 0.55},
    {"event": "Hurricane Harvey", "year": 2017, "month": 9,
     "country": COUNTRY_US,
     "states": ["TX", "LA"], "severity": 0.40},

    # 2017 — Hurricane Irma (Sep 6–12, 2017), landfall on PR Sep 6, FL Sep 10
    #   PR took a direct hit; US East Coast lost power for days. The Oct 2017
    #   volume drop in this dataset largely reflects this + Maria below.
    {"event": "Hurricane Irma", "year": 2017, "month": 9,
     "country": COUNTRY_PR,
     "states": ["PR"], "severity": 0.55},
    {"event": "Hurricane Irma", "year": 2017, "month": 10,
     "country": COUNTRY_PR,
     "states": ["PR"], "severity": 0.65},
    {"event": "Hurricane Irma", "year": 2017, "month": 9,
     "country": COUNTRY_US,
     "states": ["FL", "GA", "SC", "NC", "VA"], "severity": 0.45},
    {"event": "Hurricane Irma", "year": 2017, "month": 10,
     "country": COUNTRY_US,
     "states": ["FL", "GA"], "severity": 0.40},

    # 2017 — Hurricane Maria (Sep 19–20, 2017), catastrophic for PR
    {"event": "Hurricane Maria", "year": 2017, "month": 9,
     "country": COUNTRY_PR,
     "states": ["PR"], "severity": 0.75},
    {"event": "Hurricane Maria", "year": 2017, "month": 10,
     "country": COUNTRY_PR,
     "states": ["PR"], "severity": 0.90},
    {"event": "Hurricane Maria", "year": 2017, "month": 11,
     "country": COUNTRY_PR,
     "states": ["PR"], "severity": 0.80},
    {"event": "Hurricane Maria", "year": 2017, "month": 12,
     "country": COUNTRY_PR,
     "states": ["PR"], "severity": 0.60},
]


def country_monthly_indicator() -> pd.DataFrame:
    """One row per (customer_country, year_month) with `known_severity` =
    maximum severity across overlapping events. Useful when state info is
    missing or coarse aggregation is desired.
    """
    rows: list[dict] = []
    for d in KNOWN_DISASTERS:
        rows.append({
            "customer_country": d["country"],
            "year_month": pd.Timestamp(year=d["year"], month=d["month"], day=1),
            "severity": d["severity"],
            "event": d["event"],
        })
    df = pd.DataFrame(rows)
    # If multiple events overlap, take the *worst*
    agg = (df.groupby(["customer_country", "year_month"])
             .agg(known_severity=("severity", "max"),
                  events=("event", lambda s: " + ".join(sorted(set(s)))))
             .reset_index())
    return agg


def state_monthly_indicator() -> pd.DataFrame:
    """One row per (customer_country, customer_state, year_month) with
    `known_severity` = max across overlapping events."""
    rows: list[dict] = []
    for d in KNOWN_DISASTERS:
        for state in d["states"]:
            rows.append({
                "customer_country": d["country"],
                "customer_state": state,
                "year_month": pd.Timestamp(year=d["year"], month=d["month"], day=1),
                "severity": d["severity"],
                "event": d["event"],
            })
    df = pd.DataFrame(rows)
    agg = (df.groupby(["customer_country", "customer_state", "year_month"])
             .agg(known_severity=("severity", "max"),
                  events=("event", lambda s: " + ".join(sorted(set(s)))))
             .reset_index())
    return agg


if __name__ == "__main__":
    print("=== Country-level known disasters ===")
    print(country_monthly_indicator().to_string(index=False))
    print()
    print("=== State-level known disasters ===")
    print(state_monthly_indicator().to_string(index=False))
