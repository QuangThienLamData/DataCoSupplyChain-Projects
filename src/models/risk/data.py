"""Order-level data loader for the M3 risk sub-models.

Loads from `data/dataco.db::supply_chain` and exposes the features that are
known *at order time*. The shipping-outcome columns
(`days_shipping_real`, `delivery_status`, `shipping_delay_days`) are deliberately
withheld to avoid label leakage in fraud / cancel / late-delivery models.

Split policy (same time boundaries as M1/M2):
    train : order_date <  2017-01-01
    val   : 2017-01-01 ≤ order_date <  2017-07-01
    test  : 2017-07-01 ≤ order_date <  2017-10-01
    (truncated period 2017-10-01 → 2018-01-31 is dropped here too)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "dataco.db"

TRAIN_END = pd.Timestamp("2017-01-01")
VAL_END = pd.Timestamp("2017-07-01")
TEST_END = pd.Timestamp("2018-02-01")  # test now runs through 2018-01 inclusive

# Columns safe to use at order time (no shipping leakage)
ORDER_TIME_COLUMNS = [
    "order_id",
    "order_item_id",
    "order_date",
    # — product
    "product_card_id",
    "category_id",
    "department_id",
    "product_price",
    "order_item_product_price",
    "order_item_discount_rate",
    "order_item_quantity",
    "order_item_total",
    "sales",
    # — payment / channel
    "payment_type",
    # — customer
    "customer_id",
    "customer_segment",
    "customer_country",
    "customer_state",
    "customer_city",
    # — destination
    "market",
    "order_region",
    "order_country",
    # — shipping (planned, known at order time)
    "shipping_mode",
    "days_shipping_scheduled",
    # — labels (decided post-hoc; we use them as TARGETS, not features)
    "order_status",
    "late_delivery_risk",
]

# These three are explicitly EXCLUDED from features (leakage):
# - days_shipping_real, delivery_status, shipping_delay_days

CATEGORICAL_COLS = [
    "payment_type", "customer_segment", "customer_country", "customer_state",
    "market", "order_region", "order_country",
    "shipping_mode", "category_id", "department_id",
    "product_card_id",
]


def load_orders() -> pd.DataFrame:
    cols = ", ".join(ORDER_TIME_COLUMNS)
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"SELECT {cols} FROM supply_chain",
            conn,
            parse_dates=["order_date"],
        )
    return df


def add_split_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time split + minor derived features. Drops the truncated tail."""
    df = df.copy()
    df = df[df["order_date"] < TEST_END]  # keep through 2018-01 inclusive

    df["split"] = np.where(
        df["order_date"] < TRAIN_END, "train",
        np.where(df["order_date"] < VAL_END, "val", "test"),
    )

    # Calendar features
    df["order_year"] = df["order_date"].dt.year.astype("int16")
    df["order_month"] = df["order_date"].dt.month.astype("int8")
    df["order_dow"] = df["order_date"].dt.dayofweek.astype("int8")
    df["order_doy"] = df["order_date"].dt.dayofyear.astype("int16")
    df["is_q4"] = (df["order_month"].isin([10, 11, 12])).astype("int8")

    # Pricing features
    df["effective_price"] = df["order_item_product_price"] * (
        1 - df["order_item_discount_rate"].fillna(0)
    )
    df["discount_amount"] = df["order_item_product_price"] * \
        df["order_item_discount_rate"].fillna(0)

    # Targets (binary)
    df["is_fraud"] = (df["order_status"] == "SUSPECTED_FRAUD").astype("int8")
    df["is_cancel"] = (df["order_status"] == "CANCELED").astype("int8")
    df["is_late"] = df["late_delivery_risk"].fillna(0).astype("int8")

    return df


def split_xy(df: pd.DataFrame, target_col: str,
             feature_cols: list | None = None) -> dict:
    """Partition into {train,val,test}[X,y] with categorical dtypes set."""
    if feature_cols is None:
        # default = everything except identifiers, targets, leakage
        drop = {"order_id", "order_item_id", "order_date", "split",
                "order_status", "late_delivery_risk",
                "is_fraud", "is_cancel", "is_late"}
        feature_cols = [c for c in df.columns if c not in drop]

    out: dict = {}
    for name in ("train", "val", "test"):
        sub = df[df["split"] == name].copy()
        X = sub[feature_cols].copy()
        # All listed categoricals + any remaining string columns → category dtype
        for c in X.columns:
            if c in CATEGORICAL_COLS or X[c].dtype == object or X[c].dtype == "string":
                X[c] = X[c].astype("category")
        y = sub[target_col].astype("int8").to_numpy()
        out[name] = {"X": X, "y": y, "order_date": sub["order_date"],
                     "product_card_id": sub["product_card_id"]}
    out["feature_cols"] = feature_cols
    return out


def load_for_target(target: str) -> dict:
    """Convenience: load + split for a given binary target column."""
    df = add_split_and_features(load_orders())
    return split_xy(df, target_col=target)
