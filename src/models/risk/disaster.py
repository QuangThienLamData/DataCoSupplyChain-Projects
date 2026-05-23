"""M3d — Disaster index from regional anomalies + known-event calendar.

Two views of disaster, fused:

1. **Synthetic anomaly index** (per `order_region` AND per `customer_country`)
   derived from volume z-scores + late-rate z-scores on a 12-month rolling
   window. Captures operational disruption that shows up in data.

2. **Known-events indicator** (per `customer_country` / `customer_state` /
   month) — hand-curated Atlantic hurricane calendar 2015-2017 covering
   Hurricane Irma (Sep 2017, PR + US East Coast), Hurricane Harvey (Aug-Sep
   2017, TX/LA), Hurricane Maria (Sep-Dec 2017, PR), Matthew, Joaquin. See
   `src/models/risk/known_disasters.py`.

The two signals are fused as `max(anomaly, known)` per geo-month — gives
credit to whichever is stronger. The customer-side index is mapped to
products via the product's historical customer-country mix (computed on the
fly from raw orders, since the panel only has destination-market mix).

Final per-product disaster_index = max(order-region projection, customer-geo
projection) — whichever channel is more disrupted dominates.

**Honest caveats**
- The synthetic index can fire on non-disaster demand shocks. The known-
  events calendar adds causal grounding for the major US/PR events.
- For other regions (LATAM, EMEA, APAC) we still rely on the synthetic
  proxy only — no calendar coverage. Future work: EM-DAT / GDACS feed.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.risk.known_disasters_v2 import (
    country_monthly_indicator,
    state_monthly_indicator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"

ROLLING_WINDOW = 12  # months
EPS = 1e-6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rolling_anomaly_index(g: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute volume-anomaly + late-rate composite for one geo group."""
    g = g.sort_values("year_month").copy()
    for col in ("orders", "late_rate"):
        mean = g[col].rolling(window=window, min_periods=3).mean()
        std = g[col].rolling(window=window, min_periods=3).std().replace(0, EPS)
        g[f"{col}_z"] = (g[col] - mean) / std
    vol = _sigmoid(-g["orders_z"].fillna(0))
    late = _sigmoid(g["late_rate_z"].fillna(0))
    g["anomaly_index"] = 0.5 * vol + 0.5 * late
    return g


def regional_monthly_signals() -> pd.DataFrame:
    """Order-destination region (`order_region`) aggregates."""
    with sqlite3.connect(ROOT / "data" / "dataco.db") as conn:
        df = pd.read_sql(
            "SELECT order_date, order_region, order_item_quantity, "
            "late_delivery_risk FROM supply_chain",
            conn, parse_dates=["order_date"],
        )
    df["year_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    agg = (df.groupby(["order_region", "year_month"])
             .agg(orders=("order_item_quantity", "size"),
                  late_rate=("late_delivery_risk", "mean"))
             .reset_index())
    return agg


def customer_country_monthly_signals() -> pd.DataFrame:
    """Customer-side aggregates: where the demand originates."""
    with sqlite3.connect(ROOT / "data" / "dataco.db") as conn:
        df = pd.read_sql(
            "SELECT order_date, customer_country, order_item_quantity, "
            "late_delivery_risk FROM supply_chain",
            conn, parse_dates=["order_date"],
        )
    df["year_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    agg = (df.groupby(["customer_country", "year_month"])
             .agg(orders=("order_item_quantity", "size"),
                  late_rate=("late_delivery_risk", "mean"))
             .reset_index())
    return agg


def customer_state_monthly_signals() -> pd.DataFrame:
    """Finer-grained customer-state aggregates (some states sparse → fewer
    months pass min_periods)."""
    with sqlite3.connect(ROOT / "data" / "dataco.db") as conn:
        df = pd.read_sql(
            "SELECT order_date, customer_country, customer_state, "
            "order_item_quantity, late_delivery_risk FROM supply_chain",
            conn, parse_dates=["order_date"],
        )
    df["year_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    agg = (df.groupby(["customer_country", "customer_state", "year_month"])
             .agg(orders=("order_item_quantity", "size"),
                  late_rate=("late_delivery_risk", "mean"))
             .reset_index())
    return agg


def compute_region_index(signals: pd.DataFrame,
                          window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Anomaly-only index per order_region-month (no known-events overlay)."""
    out = [_rolling_anomaly_index(g, window) for _, g in
           signals.groupby("order_region", sort=True)]
    df = pd.concat(out, ignore_index=True)
    df["disaster_index"] = df["anomaly_index"]
    return df


def compute_customer_country_index(
    signals: pd.DataFrame, window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Anomaly + known-events fused index per customer_country-month."""
    out = [_rolling_anomaly_index(g, window) for _, g in
           signals.groupby("customer_country", sort=True)]
    df = pd.concat(out, ignore_index=True)
    known = country_monthly_indicator()
    df = df.merge(known[["customer_country", "year_month", "known_severity"]],
                  on=["customer_country", "year_month"], how="left")
    df["known_severity"] = df["known_severity"].fillna(0.0)
    df["disaster_index"] = df[["anomaly_index", "known_severity"]].max(axis=1)
    return df


def compute_customer_state_index(
    signals: pd.DataFrame, window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Anomaly + known-events fused index per customer_state-month.
    States with <3 months of data fall back to anomaly_index = NaN, so the
    known-events overlay is the only signal there."""
    out = []
    for _, g in signals.groupby(["customer_country", "customer_state"], sort=True):
        out.append(_rolling_anomaly_index(g, window))
    df = pd.concat(out, ignore_index=True)
    known = state_monthly_indicator()
    df = df.merge(
        known[["customer_country", "customer_state", "year_month", "known_severity"]],
        on=["customer_country", "customer_state", "year_month"], how="left",
    )
    df["known_severity"] = df["known_severity"].fillna(0.0)
    df["anomaly_index"] = df["anomaly_index"].fillna(0.0)
    df["disaster_index"] = df[["anomaly_index", "known_severity"]].max(axis=1)
    return df


def region_to_market(region: str) -> str:
    """Many-to-one mapping from order_region to market label used in the panel."""
    region = (region or "").strip()
    if region in ("Western Europe", "Northern Europe", "Southern Europe",
                  "Eastern Europe", "Europe"):
        return "europe"
    if region in ("West Africa", "Central Africa", "North Africa",
                  "East Africa", "Southern Africa"):
        return "africa"
    if region in ("South America", "Central America", "Caribbean"):
        return "latam"
    if region in ("Southeast Asia", "South Asia", "Oceania",
                  "Eastern Asia", "West Asia", "Central Asia"):
        return "pacific_asia"
    if region in ("West of USA", "US Center", "East of USA", "South of  USA",
                  "South of USA", "Canada"):
        return "usca"
    return "unknown"


def aggregate_to_market(region_idx: pd.DataFrame) -> pd.DataFrame:
    """Roll regional index up to 5 markets (qty-weighted by region orders)."""
    df = region_idx.copy()
    df["market"] = df["order_region"].map(region_to_market)
    df = df[df["market"] != "unknown"]
    # Weight by region order count so big regions dominate their market
    df["weighted"] = df["disaster_index"] * df["orders"]
    out = (df.groupby(["market", "year_month"])
             .agg(orders=("orders", "sum"),
                  disaster_w=("weighted", "sum"))
             .reset_index())
    out["market_disaster_index"] = out["disaster_w"] / out["orders"]
    return out[["market", "year_month", "market_disaster_index"]]


def map_to_products(market_idx: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """For each (product, month) in the clean panel, project the market-level
    disaster index using the product's market_mix that same month."""
    clean = panel[panel["data_quality"] == "ok"].copy()
    mkt_cols = ["mkt_africa", "mkt_europe", "mkt_latam",
                "mkt_pacific_asia", "mkt_usca"]
    # Pivot market_idx wide
    wide = market_idx.pivot(index="year_month", columns="market",
                            values="market_disaster_index").reset_index()
    wide.columns = ["year_month"] + [f"di_{c}" for c in wide.columns[1:]]
    out = clean[["product_card_id", "year_month"] + mkt_cols].merge(
        wide, on="year_month", how="left",
    )
    # Weighted sum: Σ_market mkt_share × disaster_index_market
    for c in mkt_cols:
        di_col = f"di_{c.replace('mkt_', '')}"
        if di_col not in out.columns:
            out[di_col] = 0.0
    out["disaster_index"] = sum(
        out[c].fillna(0) * out[f"di_{c.replace('mkt_', '')}"].fillna(0)
        for c in mkt_cols
    )
    return out[["product_card_id", "year_month", "disaster_index"]]


def product_customer_country_mix() -> pd.DataFrame:
    """For each (product, month), share of orders by customer_country.
    Wide format: rows = (product_card_id, year_month), cols = cust_<country>.
    """
    with sqlite3.connect(ROOT / "data" / "dataco.db") as conn:
        df = pd.read_sql(
            "SELECT order_date, product_card_id, customer_country FROM supply_chain",
            conn, parse_dates=["order_date"],
        )
    df["year_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    mix = (df.assign(_one=1)
             .pivot_table(index=["product_card_id", "year_month"],
                          columns="customer_country", values="_one",
                          aggfunc="sum", fill_value=0))
    mix = mix.div(mix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    mix.columns = [f"cust_{c}" for c in mix.columns]
    return mix.reset_index()


def map_customer_country_to_products(country_idx: pd.DataFrame) -> pd.DataFrame:
    """Project per-country disaster_index to (product, month) via the
    product's actual customer-country mix that month."""
    mix = product_customer_country_mix()
    wide = country_idx.pivot(index="year_month", columns="customer_country",
                              values="disaster_index").reset_index()
    wide.columns = ["year_month"] + [f"di_cust_{c}" for c in wide.columns[1:]]
    out = mix.merge(wide, on="year_month", how="left")
    di_cols = [c for c in out.columns if c.startswith("di_cust_")]
    cust_cols = [c.replace("di_", "") for c in di_cols]
    out["disaster_customer"] = sum(
        out[c].fillna(0) * out[d].fillna(0)
        for c, d in zip(cust_cols, di_cols)
    )
    return out[["product_card_id", "year_month", "disaster_customer"]]


def fuse_product_disaster(
    market_proj: pd.DataFrame, customer_proj: pd.DataFrame,
) -> pd.DataFrame:
    """Fuse order-destination (market) and customer-country disaster signals
    per product-month — take the max so the worse-disrupted channel wins."""
    out = market_proj.merge(customer_proj, on=["product_card_id", "year_month"],
                            how="outer")
    out["disaster_index_market"] = out["disaster_index"].fillna(0)
    out["disaster_index_customer"] = out["disaster_customer"].fillna(0)
    out["disaster_index"] = out[["disaster_index_market",
                                 "disaster_index_customer"]].max(axis=1)
    return out[["product_card_id", "year_month",
                "disaster_index_market", "disaster_index_customer",
                "disaster_index"]]


def run() -> dict[str, pd.DataFrame]:
    log.info("loading panel + raw signals")
    panel = pd.read_parquet(PANEL_PATH)

    log.info("computing region-side (order destination) disaster index")
    region_idx = compute_region_index(regional_monthly_signals())
    market_idx = aggregate_to_market(region_idx)
    market_proj = map_to_products(market_idx, panel).rename(
        columns={"disaster_index": "disaster_index"}
    )

    log.info("computing customer-country (origin) disaster index + known events")
    country_idx = compute_customer_country_index(customer_country_monthly_signals())
    customer_proj = map_customer_country_to_products(country_idx)

    log.info("computing customer-state index (state-level known events)")
    state_idx = compute_customer_state_index(customer_state_monthly_signals())

    log.info("fusing market-side and customer-side disaster signals")
    product_idx = fuse_product_disaster(market_proj, customer_proj)

    return {
        "region_index": region_idx,
        "market_index": market_idx,
        "country_index": country_idx,
        "state_index": state_idx,
        "product_index": product_idx,
    }


if __name__ == "__main__":
    out = run()
    print("country_index (sorted by disaster_index):")
    print(out["country_index"]
          .sort_values("disaster_index", ascending=False).head(10)
          .round(3).to_string(index=False))
    print("\nstate_index — top 10 disaster months:")
    print(out["state_index"]
          .sort_values("disaster_index", ascending=False).head(10)
          .round(3).to_string(index=False))
    print("\nproduct_index summary:")
    print(out["product_index"]
          [["disaster_index_market", "disaster_index_customer", "disaster_index"]]
          .describe().round(4))
