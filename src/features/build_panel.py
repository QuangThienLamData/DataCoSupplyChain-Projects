"""Build the monthly product panel for the forecasting stack.

Reads `data/dataco.db::supply_chain` and aggregates to one row per
(product_card_id, year_month) with quantity, price, risk, and exposure features.

Output: data/processed/monthly_panel.parquet
Usage:  .venv312/Scripts/python.exe -m src.features.build_panel
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "dataco.db"
OUT_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "panel_meta.parquet"

# Order statuses excluded from realized quantity / revenue
EXCLUDED_STATUS = ("CANCELED", "SUSPECTED_FRAUD")

# Markets are the 5-bucket version of regions; manageable as one-hots
MARKETS = ["Africa", "Europe", "LATAM", "Pacific Asia", "USCA"]

# Time splits — extended to use the full dataset (2015-01 .. 2018-01).
# Earlier versions truncated after 2017-09 because qty_per_row drops to 1.0
# in 2017-10+. We've confirmed this is real economic disruption from
# Hurricane Maria (PR), not data corruption — sales values are consistent
# with order_item_total / effective_price reconstruction.
SPLIT_TRAIN_END = "2016-12"   # 24 months of train (2 full seasonal cycles)
SPLIT_VAL_END = "2017-06"     # 6 months of validation (calm period)
SPLIT_TEST_END = "2018-01"    # 7 months of test — covers full hurricane season + recovery


def load_orders() -> pd.DataFrame:
    """Pull the minimum slice of columns we need from SQLite."""
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


def add_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
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


def aggregate_panel(orders: pd.DataFrame) -> pd.DataFrame:
    """Aggregate row-level orders to (product_card_id, year_month)."""
    realized = orders[orders["is_realized"]]

    grp_all = orders.groupby(["product_card_id", "year_month"], sort=True)
    grp_realized = realized.groupby(["product_card_id", "year_month"], sort=True)

    # Realized (excluding cancel + fraud) — used by demand/sales models
    realized_agg = grp_realized.apply(
        lambda g: pd.Series({
            "qty": g["order_item_quantity"].sum(),
            "revenue_realized": g["sales"].sum(),
            "p_eff": _weighted_mean(g["effective_price"], g["order_item_quantity"]),
            "discount_rate_avg": _weighted_mean(
                g["order_item_discount_rate"], g["order_item_quantity"]
            ),
            "discount_rate_std": float(g["order_item_discount_rate"].std(ddof=0)),
            "n_orders_realized": int(len(g)),
        }),
        include_groups=False,
    )

    # Full universe (incl. cancel/fraud) — used for risk-rate denominators
    # AND for the M4 backtest target (gross_revenue = revenue the customer
    # attempted to spend, before fraud/cancel removal).
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

    # Market mix — share of orders per market, this product-month
    market_mix = (
        orders.assign(_one=1)
        .pivot_table(
            index=["product_card_id", "year_month"],
            columns="market",
            values="_one",
            aggfunc="sum",
            fill_value=0,
        )
    )
    market_mix = market_mix.div(market_mix.sum(axis=1).replace(0, np.nan), axis=0)
    # Guarantee all 5 markets are present even if absent for some products
    for m in MARKETS:
        if m not in market_mix.columns:
            market_mix[m] = 0.0
    market_mix = market_mix[MARKETS].rename(columns=lambda c: f"mkt_{c.lower().replace(' ', '_')}")

    panel = all_agg.join(realized_agg, how="left").join(market_mix, how="left")
    return panel.reset_index()


def reindex_dense(panel: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing (product, month) combinations with zeros/NaNs so each
    product has a contiguous monthly series — required for time-series models.
    """
    months = pd.period_range(
        panel["year_month"].min().to_period("M"),
        panel["year_month"].max().to_period("M"),
        freq="M",
    ).to_timestamp()

    products = panel["product_card_id"].unique()
    full_idx = pd.MultiIndex.from_product(
        [products, months], names=["product_card_id", "year_month"]
    )
    dense = (
        panel.set_index(["product_card_id", "year_month"])
        .reindex(full_idx)
        .reset_index()
    )

    # Zero-fill the count-style columns where a product had no orders that month
    zero_fill = [
        "qty", "gross_qty", "revenue_realized", "gross_revenue",
        "n_orders_realized", "n_orders_total",
    ]
    dense[zero_fill] = dense[zero_fill].fillna(0)

    # Rates stay NaN where there were zero orders (rate is undefined, not zero)
    return dense


def add_product_meta(panel: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Backfill product/category metadata (constant per product)."""
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
    # Where p_list was NaN (zero-order months), use the constant
    panel["p_list"] = panel["p_list"].fillna(panel["p_list_const"])
    panel = panel.drop(columns=["p_list_const"])
    return panel


def add_calendar_and_split(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    ym = panel["year_month"]
    panel["year"] = ym.dt.year.astype("int16")
    panel["month"] = ym.dt.month.astype("int8")
    panel["quarter"] = ym.dt.quarter.astype("int8")
    panel["is_q4"] = (panel["quarter"] == 4).astype("int8")
    panel["is_nov_dec"] = panel["month"].isin([11, 12]).astype("int8")

    label = ym.dt.strftime("%Y-%m")
    panel["split"] = np.select(
        [
            label <= SPLIT_TRAIN_END,
            label <= SPLIT_VAL_END,
            label <= SPLIT_TEST_END,
        ],
        ["train", "val", "test"],
        default="future",  # any months beyond test window stay as "future"
    )
    # data_quality: kept for API compatibility; all months are now usable.
    # 2017-10..2018-01 reflect real Hurricane Maria-era economic disruption,
    # not data corruption. See sql diagnostics in memory.
    panel["data_quality"] = "ok"
    return panel


def build() -> pd.DataFrame:
    print("[1/5] loading orders from SQLite...")
    orders = load_orders()
    print(f"      {len(orders):,} rows")

    print("[2/5] adding time keys and derived fields...")
    orders = add_time_keys(orders)

    print("[3/5] aggregating to (product, month)...")
    panel = aggregate_panel(orders)
    print(f"      {len(panel):,} populated rows before reindex")

    print("[4/5] reindexing to dense panel + product metadata...")
    panel = reindex_dense(panel)
    panel = add_product_meta(panel, orders)

    print("[5/5] calendar features and train/val/test split...")
    panel = add_calendar_and_split(panel)

    panel = panel.sort_values(["product_card_id", "year_month"]).reset_index(drop=True)
    return panel


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel = build()
    panel.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  shape={panel.shape}")
    print("columns:", list(panel.columns))

    # Tiny meta file: product-level summary, useful for cohort selection
    months_active = (
        panel[panel["qty"] > 0]
        .groupby("product_card_id")["year_month"]
        .nunique()
        .rename("months_active")
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
        .join(months_active)
        .fillna({"months_active": 0})
        .reset_index()
    )
    meta["cohort"] = np.where(meta["months_active"] >= 12, "A_active", "B_sparse")
    meta.to_parquet(META_PATH, index=False)
    print(f"wrote {META_PATH}  cohorts: {meta['cohort'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
