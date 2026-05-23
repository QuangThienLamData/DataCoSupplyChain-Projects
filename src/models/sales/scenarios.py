"""M4 scenarios: Base, Stress, Tail.

Each scenario specifies how to perturb the inputs *before* the Monte Carlo
integration:

- **Base** — planned_price = trailing baseline; risk_drag as-is.
- **Stress** — list price (no discount) + risk inputs scaled up ×1.5.
- **Tail** — list price + disaster_index forced to its 95th-percentile value
  observed in the historical data (regardless of what the model says for that
  month). Models the "what if a hurricane lands here" question.

Returns one frame per scenario, plus a comparison view.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.sales.forecast import forecast_frame


@dataclass
class Scenario:
    name: str
    price_factor: float
    risk_scale: float            # multiply all order-level risk probs
    disaster_override: float | None  # if set, force disaster_index = this value


def apply_scenario(inputs: pd.DataFrame, sc: Scenario) -> pd.DataFrame:
    df = inputs.copy()
    # Risk scaling — scale order-level probabilities, clip
    for c in ("p_fraud", "p_cancel", "p_late"):
        df[c] = (df[c] * sc.risk_scale).clip(0.0, 1.0)
    if sc.disaster_override is not None:
        df["disaster_index"] = sc.disaster_override
    return df


def run_scenarios(
    base_inputs: pd.DataFrame,
    elasticity_mean: float = -0.687,
    elasticity_se: float = 0.389,
    historical_disaster_p95: float | None = None,
    n_samples: int = 2000,
) -> dict[str, pd.DataFrame]:
    """Run Base / Stress / Tail and return a dict of frames."""
    if historical_disaster_p95 is None:
        historical_disaster_p95 = float(base_inputs["disaster_index"].quantile(0.95))

    scenarios = [
        Scenario("base",   price_factor=1.0,  risk_scale=1.0, disaster_override=None),
        Scenario("stress", price_factor=1.10, risk_scale=1.5, disaster_override=None),
        Scenario("tail",   price_factor=1.10, risk_scale=1.5,
                 disaster_override=historical_disaster_p95),
    ]

    out: dict[str, pd.DataFrame] = {}
    for sc in scenarios:
        perturbed = apply_scenario(base_inputs, sc)
        fc = forecast_frame(
            perturbed,
            planned_price_factor=sc.price_factor,
            elasticity_mean=elasticity_mean,
            elasticity_se=elasticity_se,
            n_samples=n_samples,
        )
        fc["scenario"] = sc.name
        out[sc.name] = fc
    return out


def long_compare(scenarios_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concat scenario frames into a single long-form table for downstream
    plotting / dashboards."""
    return pd.concat(scenarios_dict.values(), ignore_index=True)
