"""Aggregate the 4 sub-model outputs into a single product-month risk_drag.

For each (product, month):
    risk_drag = 1 - (1 - p_fraud) · (1 - p_cancel) · (1 - λ · p_late) · (1 - μ · disaster_index)

Where:
- p_fraud, p_cancel ∈ [0, 1] — calibrated probabilities (fraction of orders lost).
- p_late ∈ [0, 1] — late-delivery rate; not all late deliveries become revenue
  loss, so we apply a *damping factor* λ = 0.10 (assumption: ~10% of late
  deliveries trigger refunds / cancellations / lost-customer revenue).
- disaster_index ∈ [0, 1] — composite proxy; damped by μ = 0.20 (max 20%
  revenue drag in a "full disaster" month).

The multiplicative form means independent risks; the alternative (additive
sum) over-counts when several rates are simultaneously high.

`risk_drag` is then ready for M4 as:
    expected_revenue = qty_forecast · price_planned · (1 - risk_drag)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LATE_DAMPING = 0.10
DISASTER_DAMPING = 0.20


def combine(
    fraud_pm: pd.DataFrame,
    cancel_pm: pd.DataFrame,
    late_pm: pd.DataFrame,
    disaster_pm: pd.DataFrame,
    panel_skeleton: pd.DataFrame,
) -> pd.DataFrame:
    """Join the 4 per-product-month risk components and compute risk_drag.

    `panel_skeleton` must have columns ['product_card_id', 'year_month'] only —
    it defines the universe of rows; the 4 input frames are LEFT-joined.
    Missing values are filled with 0 (no risk known).
    """
    out = panel_skeleton[["product_card_id", "year_month"]].drop_duplicates().copy()

    for df, name in [
        (fraud_pm, "p_fraud"),
        (cancel_pm, "p_cancel"),
        (late_pm, "p_late"),
    ]:
        sub = df.rename(columns={"predicted_rate": name})[
            ["product_card_id", "year_month", name]
        ]
        out = out.merge(sub, on=["product_card_id", "year_month"], how="left")

    out = out.merge(
        disaster_pm[["product_card_id", "year_month", "disaster_index"]],
        on=["product_card_id", "year_month"], how="left",
    )

    # Fill missing with 0 (no risk signal in that cell)
    for c in ("p_fraud", "p_cancel", "p_late", "disaster_index"):
        out[c] = out[c].fillna(0.0).clip(0.0, 1.0)

    out["risk_drag"] = 1 - (
        (1 - out["p_fraud"]) *
        (1 - out["p_cancel"]) *
        (1 - LATE_DAMPING * out["p_late"]) *
        (1 - DISASTER_DAMPING * out["disaster_index"])
    )
    return out
