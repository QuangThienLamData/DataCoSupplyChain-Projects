"""TimesFM-2.5 wrapper for the M1 Demand forecaster.

Zero-shot inference per product. We feed each product's monthly quantity series
(history) and ask for `horizon` future months with continuous quantile head.

Degrades gracefully: if `timesfm` or `torch` can't import (or weights can't be
downloaded), the wrapper exposes an `available` flag so callers can skip it.

Output normalization: TimesFM emits 10 quantile channels (mean, P10..P90).
We project to {q10, q50, q90} and clip to non-negative integers, since
demand quantity is by definition >= 0 and integer.
"""
from __future__ import annotations

from . import _preamble  # torch-first import order

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def _try_import():
    try:
        import torch  # noqa: F401
        import timesfm  # noqa: F401
        return True, None
    except Exception as e:  # pragma: no cover - environment-dependent
        return False, e


_AVAILABLE, _IMPORT_ERR = _try_import()


@dataclass
class TimesFMForecaster:
    """Wrapper around `timesfm.TimesFM_2p5_200M_torch`.

    Loads the model once on first call (lazy), then reuses for every product.
    """

    max_context: int = 128         # months of history fed to model (we have ≤33)
    max_horizon: int = 12          # max horizon we'll ever request
    huggingface_repo: str = "google/timesfm-2.5-200m-pytorch"
    name: str = "timesfm"
    _model: Optional[object] = field(default=None, init=False, repr=False)

    # ---- public ----
    @property
    def available(self) -> bool:
        return _AVAILABLE

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not _AVAILABLE:
            raise RuntimeError(f"timesfm/torch unavailable: {_IMPORT_ERR}")
        import torch
        import timesfm

        torch.set_float32_matmul_precision("high")
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.huggingface_repo)
        m.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._model = m
        log.info("TimesFM model loaded and compiled")

    def forecast(self, history: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
        """Single-series interface (matches baselines)."""
        self._ensure_loaded()
        h = np.nan_to_num(np.asarray(history, dtype=float), nan=0.0)
        if len(h) > self.max_context:
            h = h[-self.max_context:]
        # TimesFM needs at least a few values
        if len(h) < 4:
            from .baselines import SeasonalNaiveForecaster
            return SeasonalNaiveForecaster().forecast(h, horizon)

        point_fc, quantile_fc = self._model.forecast(  # type: ignore[union-attr]
            horizon=horizon,
            inputs=[h],
        )
        # quantile_fc shape: (1, horizon, 10) — [mean, P10, P20, ..., P90]
        q = quantile_fc[0]                  # (horizon, 10)
        q10 = q[:, 1]; q50 = q[:, 5]; q90 = q[:, 9]
        # Enforce non-negative and monotonicity
        q50 = np.maximum(q50, 0.0)
        q10 = np.minimum(np.maximum(q10, 0.0), q50)
        q90 = np.maximum(q90, q50)
        return {"q10": q10, "q50": q50, "q90": q90}

    def forecast_batch(
        self,
        histories: list[np.ndarray],
        horizon: int,
    ) -> list[dict[str, np.ndarray]]:
        """Batched inference for multiple series. Cheaper than calling
        forecast() in a loop because TimesFM amortises one model pass."""
        self._ensure_loaded()
        cleaned: list[np.ndarray] = []
        for h in histories:
            arr = np.nan_to_num(np.asarray(h, dtype=float), nan=0.0)
            if len(arr) > self.max_context:
                arr = arr[-self.max_context:]
            cleaned.append(arr)
        point_fc, q_fc = self._model.forecast(horizon=horizon, inputs=cleaned)  # type: ignore[union-attr]
        out: list[dict[str, np.ndarray]] = []
        for i in range(len(cleaned)):
            q = q_fc[i]
            q10 = q[:, 1]; q50 = q[:, 5]; q90 = q[:, 9]
            q50 = np.maximum(q50, 0.0)
            q10 = np.minimum(np.maximum(q10, 0.0), q50)
            q90 = np.maximum(q90, q50)
            out.append({"q10": q10, "q50": q50, "q90": q90})
        return out


def forecast_panel_timesfm(
    panel,
    model: TimesFMForecaster,
    origin,
    horizon: int,
    product_ids: list | None = None,
):
    """Generate TimesFM forecasts for every product. Mirrors
    baselines.forecast_panel_baseline but uses batched inference."""
    import pandas as pd

    clean = panel[panel["data_quality"] == "ok"].copy()
    months = pd.to_datetime(sorted(clean["year_month"].unique()))
    origin = pd.Timestamp(origin)
    if origin not in months:
        raise ValueError(f"origin {origin} not in panel months")
    origin_idx = list(months).index(origin)
    horizon = min(horizon, len(months) - origin_idx - 1)
    if horizon <= 0:
        return pd.DataFrame(columns=[
            "product_card_id", "year_month", "horizon", "q10", "q50", "q90", "model"
        ])
    forecast_months = months[origin_idx + 1: origin_idx + 1 + horizon]
    pids = product_ids if product_ids is not None else sorted(clean["product_card_id"].unique())

    histories = []
    for pid in pids:
        hist = clean[(clean["product_card_id"] == pid) & (clean["year_month"] <= origin)]
        histories.append(hist.sort_values("year_month")["qty"].to_numpy())

    quantiles = model.forecast_batch(histories, horizon)

    frames = []
    for pid, q in zip(pids, quantiles):
        frames.append(pd.DataFrame({
            "product_card_id": pid,
            "year_month": forecast_months,
            "horizon": np.arange(1, horizon + 1),
            "q10": q["q10"], "q50": q["q50"], "q90": q["q90"],
            "model": model.name,
        }))
    return pd.concat(frames, ignore_index=True)
