"""Feature engineering for the M2 price-elasticity panel.

From `monthly_panel.parquet`, build a regression frame at (product, month) grain
with all variables needed for the log-log elasticity specification:

    log_qty = α_p + β_p · log_p_eff + γ · log_p_eff_cat_excl_self
              + ρ · log_qty_lag1 + δ · month dummies + ε

Where:
- `p_eff` is the qty-weighted effective price (already on the panel).
- `p_eff_cat_excl_self` is the category-level effective price *excluding* the
  current product — captures substitution (cross-price elasticity).
- `log_qty_lag1` controls for demand persistence (partial adjustment),
  reducing simultaneity bias from price reacting to demand.
- Month dummies absorb seasonality.

Rows where qty == 0 or p_eff is missing are dropped (log is undefined). Rows
from the truncated window (`data_quality != 'ok'`) are dropped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "panel_meta.parquet"


def _category_price_excl_self(clean: pd.DataFrame) -> pd.Series:
    """For each (product, month), compute the qty-weighted average effective
    price across OTHER products in the same category, same month.

    Implementation: total = sum over category of (qty * p_eff); self = own
    qty * own p_eff; weight_total = sum of qty; subtract self contribution.
    """
    df = clean[["product_card_id", "category_id", "year_month", "qty", "p_eff"]].copy()
    df["px_qty"] = df["p_eff"] * df["qty"]

    cat_tot = (df.groupby(["category_id", "year_month"])
                 .agg(cat_pxq=("px_qty", "sum"),
                      cat_qty=("qty", "sum"))
                 .reset_index())
    out = df.merge(cat_tot, on=["category_id", "year_month"], how="left")
    out["cat_pxq_excl"] = out["cat_pxq"] - out["px_qty"]
    out["cat_qty_excl"] = out["cat_qty"] - out["qty"]
    # Where this is the only product in its category-month, fall back to own price
    out["p_eff_cat_excl_self"] = np.where(
        out["cat_qty_excl"] > 0,
        out["cat_pxq_excl"] / out["cat_qty_excl"],
        out["p_eff"],
    )
    return out.set_index(["product_card_id", "year_month"])["p_eff_cat_excl_self"]


def build_elasticity_frame(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return regression-ready frame with logged & lagged features."""
    if panel is None:
        panel = pd.read_parquet(PANEL_PATH)
    clean = panel[(panel["data_quality"] == "ok") &
                  (panel["qty"] > 0) &
                  (panel["p_eff"].notna()) &
                  (panel["p_eff"] > 0)].copy()

    cat_price = _category_price_excl_self(clean)
    clean = clean.merge(
        cat_price.rename("p_eff_cat_excl_self"),
        left_on=["product_card_id", "year_month"], right_index=True, how="left",
    )

    clean["log_qty"] = np.log(clean["qty"].astype(float))
    clean["log_p_eff"] = np.log(clean["p_eff"].astype(float))
    clean["log_p_eff_cat"] = np.log(clean["p_eff_cat_excl_self"].astype(float))

    # 1-month lag of log_qty per product (NaN at the first observation)
    clean = clean.sort_values(["product_card_id", "year_month"])
    clean["log_qty_lag1"] = clean.groupby("product_card_id")["log_qty"].shift(1)

    # 12-month lag for an alternative IV / specification check
    clean["log_qty_lag12"] = clean.groupby("product_card_id")["log_qty"].shift(12)

    # Keep what regressions need
    cols = [
        "product_card_id", "category_id", "category_name", "year_month",
        "qty", "p_eff", "p_eff_cat_excl_self",
        "log_qty", "log_p_eff", "log_p_eff_cat", "log_qty_lag1", "log_qty_lag12",
        "discount_rate_avg", "month", "year", "split",
    ]
    return clean[cols].reset_index(drop=True)


def cohort_products(cohort: str = "A_active") -> list:
    meta = pd.read_parquet(META_PATH)
    return meta.loc[meta["cohort"] == cohort, "product_card_id"].tolist()


if __name__ == "__main__":
    df = build_elasticity_frame()
    print("frame shape:", df.shape)
    print("rows per product (cohort A):")
    a = cohort_products("A_active")
    print(df[df["product_card_id"].isin(a)]
          .groupby("product_card_id").size().describe())
