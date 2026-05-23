"""Revenue-lag adapter: converts `disaster_index` (storm severity at the
event month, peak at landfall) into `disaster_drag_index` (revenue-impact
shape, peak at M+2).

**Motivation.** A storm has two different timing profiles depending on the
question being asked:

1. *"When is the storm happening?"* — peak at the landfall month, decays
   through immediate aftermath. This is the **severity** profile and is
   what `disaster_index` represents (controlled by `LAG_PROFILE` in
   `src/features/storm_exposure.py`).

2. *"When does revenue actually drop because of this storm?"* — pre-storm
   orders ship during the landfall month, so revenue dips only mildly that
   month; customers can't reorder for weeks because of power / port
   outages, so revenue collapses 1–2 months later. This is the **revenue
   drag** profile and is what M4's sales math actually needs.

Both come from the same underlying storm impulse, just convolved with
different impulse-response kernels. By keeping `disaster_index` for
dashboards and M5 anomaly attribution (where "the storm is happening
now" is the right framing) and deriving a separate `disaster_drag_index`
for M4 (where "what revenue do we lose this month" is the right framing),
we avoid mixing two distinct semantics into one field.

How
---
`disaster_index` already has the severity decay baked in: for a single
storm with peak severity 1.0 at month M, the series is roughly
`[1.00, 0.80, 0.45, 0.25, 0.10]` over `[M, M+1, M+2, M+3, M+4]` (see
`storm_exposure.LAG_PROFILE`).

We convolve that with a small revenue-impact kernel to shift the peak
forward to M+2:

    drag[t] = Σ_k W_k · disaster[t-k]

where `W = [0.05, 0.20, 0.80]` so:
- W_0 = 0.05 means landfall-month severity translates to 5% drag (most
  customer orders already placed/shipped before the storm hit)
- W_1 = 0.20 means severity at M (one month ago) contributes 20% to this
  month's drag (early disruption emerges)
- W_2 = 0.80 means severity at M-1 (two months ago) contributes 80% to
  this month's drag — peak revenue impact lands here

Applied to a [1.00, 0.80, 0.45, 0.25, 0.10] storm series:
- drag[M]   = 0.05 × 1.00                                  = 0.05
- drag[M+1] = 0.05 × 0.80 + 0.20 × 1.00                    = 0.24
- drag[M+2] = 0.05 × 0.45 + 0.20 × 0.80 + 0.80 × 1.00      = 0.98  ← PEAK
- drag[M+3] = 0.05 × 0.25 + 0.20 × 0.45 + 0.80 × 0.80      = 0.74
- drag[M+4] = 0.05 × 0.10 + 0.20 × 0.25 + 0.80 × 0.45      = 0.42

The peak shifts from landfall (M) to M+2, matching observed PR revenue
behaviour around Hurricane Maria.

To recalibrate the weights when new data arrives, see
`src/models/sales/calibrate_damping.py` — fit `implied_drag` against
this convolved series instead of the raw disaster_index for a cleaner
identification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Convolution kernel — drag[t] = Σ_k W_k · disaster[t-k]
REVENUE_LAG_WEIGHTS: tuple[float, ...] = (0.05, 0.20, 0.80)


def apply_revenue_lag_per_product(
    risk_df: pd.DataFrame,
    src_col: str = "disaster_index",
    dst_col: str = "disaster_drag_index",
    weights: tuple[float, ...] = REVENUE_LAG_WEIGHTS,
) -> pd.DataFrame:
    """Per (product, month) row in `risk_df`, derive `dst_col` as a lagged
    convolution of `src_col`. Returns a new DataFrame with the new column
    added (and rows re-sorted by product_card_id, year_month).

    The function is conservative at the leading edge: rows before the
    kernel can be fully applied see only the in-window weights (truncated
    convolution), which gives 0 if no prior storm exists.
    """
    if src_col not in risk_df.columns:
        raise KeyError(f"{src_col!r} not in risk_df.columns")

    w = np.asarray(weights, dtype=float)
    df = risk_df.sort_values(["product_card_id", "year_month"]).reset_index(drop=True).copy()
    drag = np.zeros(len(df), dtype=float)

    for pid, idx in df.groupby("product_card_id", sort=False).indices.items():
        d = df.loc[idx, src_col].to_numpy(dtype=float)
        out = np.zeros_like(d)
        for i in range(len(d)):
            for k, weight in enumerate(w):
                if i - k >= 0:
                    out[i] += weight * d[i - k]
        drag[idx] = out

    df[dst_col] = np.clip(drag, 0.0, 1.0)
    return df


if __name__ == "__main__":
    # Demo on synthetic single-storm series
    storm = pd.Series(
        [0, 0, 1.00, 0.80, 0.45, 0.25, 0.10, 0, 0],
        index=pd.date_range("2017-07-01", periods=9, freq="MS"),
    )
    df = pd.DataFrame({
        "product_card_id": [1] * len(storm),
        "year_month": storm.index,
        "disaster_index": storm.values,
    })
    out = apply_revenue_lag_per_product(df)
    print("disaster_index to disaster_drag_index (peak shifts M to M+2):")
    print(out.round(3).to_string(index=False))
