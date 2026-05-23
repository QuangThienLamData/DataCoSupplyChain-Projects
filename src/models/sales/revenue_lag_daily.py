"""Daily revenue-lag adapter — daily-frequency analogue of `revenue_lag.py`.

Converts a daily `disaster_index` (per product × date) into a daily
`disaster_drag_index` via convolution with a daily impulse-response kernel.

The monthly kernel `(0.05, 0.20, 0.80)` over month offsets 0/1/2 maps to a
daily kernel that preserves the same monthly mass profile while smoothing
within each month:

    daily_weight[d] = monthly_weight[d // 30] / 30,   d in [0, 89]

Total mass per monthly bucket is preserved (0.05 + 0.20 + 0.80 = 1.05),
and the daily kernel still peaks around day 60–89 — matching the M+2
revenue-drag peak observed for Maria/PR.

Scenario kernels (used in the unified pipeline) follow the same rule:

    pessimistic 6-mo: (0.05, 0.15, 0.40, 0.30, 0.20, 0.10) → 180-day daily kernel
    baseline       :  (0.05, 0.20, 0.80)                  →  90-day daily kernel
    optimistic     :  (0.05, 0.10, 0.30)                  →  90-day daily kernel
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30

# Default daily kernel == baseline monthly kernel disaggregated to days.
REVENUE_LAG_WEIGHTS_MONTHLY: tuple[float, ...] = (0.05, 0.20, 0.80)


def monthly_to_daily_kernel(monthly: tuple[float, ...],
                              days_per_month: int = DAYS_PER_MONTH) -> np.ndarray:
    """Spread each monthly bucket uniformly over `days_per_month` days.

    Conserves total mass per bucket. Returns a 1-D numpy array of length
    len(monthly) * days_per_month.
    """
    daily = np.repeat(np.asarray(monthly, dtype=float), days_per_month)
    daily = daily / days_per_month
    return daily


def apply_revenue_lag_daily(
    disaster_df: pd.DataFrame,
    src_col: str = "disaster_index",
    dst_col: str = "disaster_drag_index",
    monthly_weights: tuple[float, ...] = REVENUE_LAG_WEIGHTS_MONTHLY,
    days_per_month: int = DAYS_PER_MONTH,
    date_col: str = "date",
) -> pd.DataFrame:
    """Per (product, date) row in `disaster_df`, derive `dst_col` as a
    truncated daily convolution of `src_col` with the daily kernel
    derived from `monthly_weights`.

    Schema assumption: `disaster_df` has one row per (product_card_id,
    date) and is contiguous per product. Returns a new DataFrame.
    """
    if src_col not in disaster_df.columns:
        raise KeyError(f"{src_col!r} not in disaster_df.columns")

    kernel = monthly_to_daily_kernel(monthly_weights, days_per_month)
    df = (disaster_df
            .sort_values(["product_card_id", date_col])
            .reset_index(drop=True)
            .copy())
    drag = np.zeros(len(df), dtype=float)

    for pid, idx in df.groupby("product_card_id", sort=False).indices.items():
        d = df.loc[idx, src_col].to_numpy(dtype=float)
        # full convolution then truncate so drag[t] uses only past disaster
        full = np.convolve(d, kernel, mode="full")[: len(d)]
        drag[idx] = full

    df[dst_col] = np.clip(drag, 0.0, 1.0)
    return df


if __name__ == "__main__":
    # Demo: single-storm severity pulse at day 60 (90-day window)
    dates = pd.date_range("2017-08-01", periods=180, freq="D")
    sev = np.zeros(180)
    sev[60:75] = 1.0  # 15-day storm severity window
    sev[75:120] = np.linspace(0.8, 0.1, 45)  # decay
    df = pd.DataFrame({
        "product_card_id": [1] * 180,
        "date": dates,
        "disaster_index": sev,
    })
    out = apply_revenue_lag_daily(df)
    peak_day = out["disaster_drag_index"].idxmax()
    print(f"daily severity peak day {sev.argmax()} -> daily drag peak day {peak_day}")
    print(out.iloc[55:125].round(3)[["date", "disaster_index", "disaster_drag_index"]]
          .to_string(index=False))
