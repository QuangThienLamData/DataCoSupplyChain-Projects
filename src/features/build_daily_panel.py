"""Build the daily product panel for the daily-frequency forecasting stack.

Reads `data/dataco.db::supply_chain` and aggregates to one row per
(product_card_id, date) with quantity, price, risk, and exposure features.
Mirrors `build_panel.py` but at daily granularity. M2 elasticity stays
monthly; everything downstream of M1/M3/M4 can read this daily panel.

Output: data/processed/daily_panel.parquet
Usage:  .venv312/Scripts/python.exe -m src.features.build_daily_panel
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "dataco.db"
OUT_PATH = ROOT / "data" / "processed" / "daily_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "daily_panel_meta.parquet"

EXCLUDED_STATUS = ("CANCELED", "SUSPECTED_FRAUD")
MARKETS = ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"]

# Daily slice boundaries (mirror the monthly setup):
#   train: 2015-01-01 .. 2016-12-31  (731 days, ~2 yrs)
#   val:   2017-01-01 .. 2017-06-30  (181 days, calm window)
#   test:  2017-07-01 .. 2017-09-30  ( 92 days, hurricane window)
#   future: anything after 2017-09-30 — kept but flagged data_quality="ok"
#           (post-Maria economic disruption is real, not corruption).
SPLIT_TRAIN_END = "2016-12-31"
SPLIT_VAL_END = "2017-06-30"
SPLIT_TEST_END = "2017-09-30"


def load_orders() -> pd.DataFrame:
    cols = [
        "order_date",
        "product_card_id",
        "product_name",
        "category_id",
        "category_name",
        "department_id",
        "department_name",
        "product_price",
        "order_item_product_price",
        "order_item_discount_rate",
        "order_item_quantity",
        "sales",
        "order_item_total",
        "order_status",
        "late_delivery_risk",
        "shipping_delay_days",
        "market",
        "order_region",
    ]
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"SELECT {', '.join(cols)} FROM supply_chain",
            conn,
            parse_dates=["order_date"],
        )
    return df


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["order_date"].dt.normalize()
    df["effective_price"] = df["order_item_product_price"] * (
        1.0 - df["order_item_discount_rate"].fillna(0.0)
    )
    df["is_realized"] = ~df["order_status"].isin(EXCLUDED_STATUS)
    df["is_fraud"] = (df["order_status"] == "SUSPECTED_FRAUD").astype("int8")
    df["is_cancel"] = (df["order_status"] == "CANCELED").astype("int8")
    df["is_late"] = df["late_delivery_risk"].fillna(0).astype("int8")
    return df


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(0).to_numpy()
    v = values.to_numpy()
    mask = (w > 0) & np.isfinite(v)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def aggregate(orders: pd.DataFrame) -> pd.DataFrame:
    realized = orders[orders["is_realized"]]
    grp_all = orders.groupby(["product_card_id", "date"], sort=True)
    grp_realized = realized.groupby(["product_card_id", "date"], sort=True)

    realized_agg = grp_realized.apply(
        lambda g: pd.Series({
            "qty": g["order_item_quantity"].sum(),
            "revenue_realized": g["sales"].sum(),
            "p_eff": _weighted_mean(g["effective_price"], g["order_item_quantity"]),
            "discount_rate_avg": _weighted_mean(
                g["order_item_discount_rate"], g["order_item_quantity"]
            ),
            "n_orders_realized": int(len(g)),
        }),
        include_groups=False,
    )

    all_agg = grp_all.apply(
        lambda g: pd.Series({
            "gross_qty": g["order_item_quantity"].sum(),
            "gross_revenue": g["sales"].sum(),
            "n_orders_total": int(len(g)),
            "fraud_rate": float(g["is_fraud"].mean()),
            "cancel_rate": float(g["is_cancel"].mean()),
            "late_rate": float(g["is_late"].mean()),
            "shipping_delay_mean": float(g["shipping_delay_days"].mean(skipna=True)),
            "p_list": float(g["product_price"].iloc[0]),
        }),
        include_groups=False,
    )

    # Market mix daily (share of orders per market per product×date)
    market_mix = (
        orders.assign(_one=1)
        .pivot_table(
            index=["product_card_id", "date"],
            columns="market",
            values="_one",
            aggfunc="sum",
            fill_value=0,
        )
    )
    market_mix = market_mix.div(market_mix.sum(axis=1).replace(0, np.nan), axis=0)
    for m in MARKETS:
        if m not in market_mix.columns:
            market_mix[m] = 0.0
    market_mix = market_mix[MARKETS].rename(
        columns=lambda c: f"mkt_{c.lower().replace(' ', '_')}"
    )

    panel = all_agg.join(realized_agg, how="left").join(market_mix, how="left")
    return panel.reset_index()


def reindex_dense(panel: pd.DataFrame) -> pd.DataFrame:
    days = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    products = panel["product_card_id"].unique()
    full_idx = pd.MultiIndex.from_product(
        [products, days], names=["product_card_id", "date"]
    )
    dense = (
        panel.set_index(["product_card_id", "date"])
        .reindex(full_idx)
        .reset_index()
    )
    zero_fill = [
        "qty", "gross_qty", "revenue_realized", "gross_revenue",
        "n_orders_realized", "n_orders_total",
    ]
    dense[zero_fill] = dense[zero_fill].fillna(0)
    # Market mix: zero-fill (no orders → no exposure that day)
    for c in dense.columns:
        if c.startswith("mkt_"):
            dense[c] = dense[c].fillna(0.0)
    return dense


def add_product_meta(panel: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    meta = (
        orders.dropna(subset=["product_card_id"])
        .sort_values("order_date")
        .groupby("product_card_id")
        .agg(
            product_name=("product_name", "first"),
            category_id=("category_id", "first"),
            category_name=("category_name", "first"),
            department_id=("department_id", "first"),
            department_name=("department_name", "first"),
            p_list_const=("product_price", "first"),
        )
        .reset_index()
    )
    panel = panel.merge(meta, on="product_card_id", how="left")
    panel["p_list"] = panel["p_list"].fillna(panel["p_list_const"])
    panel = panel.drop(columns=["p_list_const"])
    return panel


def add_calendar_and_split(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    d = panel["date"]
    panel["year"] = d.dt.year.astype("int16")
    panel["month"] = d.dt.month.astype("int8")
    panel["day_of_week"] = d.dt.dayofweek.astype("int8")  # Mon=0
    panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("int8")
    panel["quarter"] = d.dt.quarter.astype("int8")
    panel["is_q4"] = (panel["quarter"] == 4).astype("int8")

    label = d.dt.strftime("%Y-%m-%d")
    panel["split"] = np.select(
        [
            label <= SPLIT_TRAIN_END,
            label <= SPLIT_VAL_END,
            label <= SPLIT_TEST_END,
        ],
        ["train", "val", "test"],
        default="future",
    )
    panel["data_quality"] = "ok"
    # Convenience month key for joining the monthly M2 elasticity
    panel["year_month"] = panel["date"].dt.to_period("M").dt.to_timestamp()
    return panel


def build() -> pd.DataFrame:
    print("[1/5] loading orders from SQLite...")
    orders = load_orders()
    print(f"      {len(orders):,} rows")

    print("[2/5] adding daily keys + derived fields...")
    orders = add_keys(orders)

    print("[3/5] aggregating to (product, date)...")
    panel = aggregate(orders)
    print(f"      {len(panel):,} populated rows before reindex")

    print("[4/5] reindexing to dense daily panel + product metadata...")
    panel = reindex_dense(panel)
    panel = add_product_meta(panel, orders)

    print("[5/5] calendar features + train/val/test split...")
    panel = add_calendar_and_split(panel)

    panel = panel.sort_values(["product_card_id", "date"]).reset_index(drop=True)
    return panel


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel = build()
    panel.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  shape={panel.shape}")
    print("columns:", list(panel.columns))

    days_active = (
        panel[panel["qty"] > 0]
        .groupby("product_card_id")["date"]
        .nunique()
        .rename("days_active")
    )
    meta = (
        panel.groupby("product_card_id")
        .agg(
            product_name=("product_name", "first"),
            category_name=("category_name", "first"),
            department_name=("department_name", "first"),
            p_list=("p_list", "first"),
            total_qty=("qty", "sum"),
            total_revenue=("revenue_realized", "sum"),
        )
        .join(days_active)
        .fillna({"days_active": 0})
        .reset_index()
    )
    # Cohort threshold: a product needs at least ~180 active days (≈ 6 months
    # spread of activity) to be a credible candidate for daily SARIMAX. Sparser
    # products fall into B_sparse and get a category-level fallback.
    meta["cohort"] = np.where(meta["days_active"] >= 180, "A_active", "B_sparse")
    meta.to_parquet(META_PATH, index=False)
    print(f"wrote {META_PATH}  cohorts: {meta['cohort'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
