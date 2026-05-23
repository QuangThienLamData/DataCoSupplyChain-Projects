"""Daily-frequency M1 backtest.

Mirrors `run_backtest.py` but operates on `data/processed/daily_panel.parquet`.
Models compared:
- seasonal_naive (lag-7 weekly)
- SARIMA (weekly seasonality m=7)
- SARIMAX (weekly seasonality m=7, with daily disaster_index exog)

Backtest slices:
- val:  origin 2016-12-31, horizon 181 days (2017-01-01..2017-06-30)
- test: origin 2017-06-30, horizon  92 days (2017-07-01..2017-09-30)

Outputs (forecasts/):
- m1_demand_daily.parquet               long-form quantile forecasts
- m1_demand_daily_metrics.parquet       per-product metrics
- m1_demand_daily_portfolio.parquet     portfolio metrics
"""
from __future__ import annotations

# Force torch-first import order
from . import _preamble  # noqa: F401

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.demand.baselines import (
    SARIMAForecaster, SARIMAXForecaster,
    _empty_quantiles,
)
from src.models.demand.timesfm_model import (
    TimesFMForecaster, forecast_panel_timesfm_daily,
)

# Daily-specific tight order grid: SARIMA(0,1,1)(0,1,0,7) is the "monthly +
# weekly seasonal difference" workhorse; SARIMA(1,1,0)(0,1,1,7) adds a weekly
# MA term. With 700+ daily obs both are well-identified and converge in ~1-2s
# per fit. Eight-order grid drove a multi-hour run; this 3-order grid keeps
# the daily backtest under ~10 min for cohort A (54 products × 2 slices × 2
# SARIMA-family models = 216 fits × ~2s × 3 orders ≈ 22 min worst case).
DAILY_SARIMA_ORDERS = (
    (0, 1, 1, 0, 1, 0),
    (1, 1, 0, 0, 1, 1),
    (0, 1, 1, 0, 0, 1),
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "daily_panel.parquet"
META_PATH = ROOT / "data" / "processed" / "daily_panel_meta.parquet"
FC_DIR = ROOT / "forecasts"

SEASON_M = 7  # weekly seasonality

VAL_ORIGIN = pd.Timestamp("2016-12-31")
VAL_HORIZON = 181
TEST_ORIGIN = pd.Timestamp("2017-06-30")
TEST_HORIZON = 92


# ---------------------------------------------------------------------------
# Daily seasonal-naive — y_hat_t = y_{t-7}
# ---------------------------------------------------------------------------
@dataclass
class SeasonalNaiveDaily:
    season: int = SEASON_M
    name: str = "seasonal_naive"

    def forecast(self, history: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
        h = np.nan_to_num(np.asarray(history, dtype=float), nan=0.0)
        if len(h) < self.season:
            return _empty_quantiles(horizon)
        # Repeat the last seasonal cycle out to horizon
        last_cycle = h[-self.season:]
        reps = int(np.ceil(horizon / self.season))
        mean = np.tile(last_cycle, reps)[:horizon]
        # Bands from in-sample residual std
        resid = h[self.season:] - h[:-self.season]
        sigma = float(np.std(resid)) if len(resid) > 1 else 0.0
        z = 1.2816
        q10 = np.clip(mean - z * sigma, 0, None)
        q90 = np.clip(mean + z * sigma, 0, None)
        return {"q10": q10, "q50": np.clip(mean, 0, None), "q90": q90}


# ---------------------------------------------------------------------------
# Daily backtest harness
# ---------------------------------------------------------------------------
def _wape(actual: np.ndarray, fc: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    fc = np.asarray(fc, dtype=float)
    denom = np.nansum(np.abs(actual))
    return float(np.nansum(np.abs(actual - fc)) / max(denom, 1e-9))


def _smape(actual: np.ndarray, fc: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    f = np.asarray(fc, dtype=float)
    denom = np.abs(a) + np.abs(f)
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2 * np.abs(a[mask] - f[mask]) / denom[mask]))


def _coverage(actual, lo, hi):
    a = np.asarray(actual, dtype=float)
    l = np.asarray(lo, dtype=float)
    h = np.asarray(hi, dtype=float)
    return float(np.mean((a >= l) & (a <= h)))


def _forecast_one(panel: pd.DataFrame, pid: int, forecaster,
                   origin: pd.Timestamp, horizon: int,
                   exog_panel: pd.DataFrame | None,
                   exog_col: str) -> dict:
    hist = panel[(panel["product_card_id"] == pid)
                  & (panel["date"] <= origin)].sort_values("date")
    fc_dates = pd.date_range(origin + pd.Timedelta(days=1),
                                periods=horizon, freq="D")
    y = hist["gross_qty"].fillna(0).to_numpy(dtype=float)
    is_sarimax = forecaster.__class__.__name__ == "SARIMAXForecaster"
    if is_sarimax and exog_panel is not None:
        ex = exog_panel[exog_panel["product_card_id"] == pid].sort_values("date")
        ex_h = ex[ex["date"] <= origin][exog_col].to_numpy(dtype=float)
        ex_f = ex[ex["date"].isin(fc_dates)][exog_col].to_numpy(dtype=float)
        if len(ex_h) != len(y) or len(ex_f) < horizon:
            q = SARIMAForecaster(seasonality=SEASON_M).forecast(y, horizon)
        else:
            q = forecaster.forecast(y, horizon, exog_history=ex_h, exog_future=ex_f[:horizon])
    else:
        q = forecaster.forecast(y, horizon)

    return {
        "product_card_id": pid,
        "fc_dates": fc_dates,
        "q10": q["q10"], "q50": q["q50"], "q90": q["q90"],
    }


def _run_slice(panel: pd.DataFrame, cohort_a: list,
                forecasters: list,
                origin: pd.Timestamp, horizon: int,
                slice_name: str,
                exog_panel: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    log.info("===== slice=%s  origin=%s  horizon=%dd =====",
             slice_name, origin.date(), horizon)
    out_forecasts: list[pd.DataFrame] = []
    out_metrics: list[dict] = []

    actuals_panel = panel[(panel["date"] > origin)
                            & (panel["date"] <= origin + pd.Timedelta(days=horizon))]
    actuals = actuals_panel.set_index(["product_card_id", "date"])["gross_qty"]

    for name, fc_obj in forecasters:
        log.info("model=%s — forecasting %d products", name, len(cohort_a))
        for pid in cohort_a:
            res = _forecast_one(panel, pid, fc_obj, origin, horizon,
                                  exog_panel, exog_col="disaster_index")
            df = pd.DataFrame({
                "product_card_id": pid,
                "date": res["fc_dates"],
                "q10": res["q10"], "q50": res["q50"], "q90": res["q90"],
                "horizon": np.arange(1, len(res["fc_dates"]) + 1),
                "model": name, "slice": slice_name, "cohort": "A_active",
            })
            out_forecasts.append(df)
            # metrics
            actual_series = actuals.reindex(
                pd.MultiIndex.from_product([[pid], res["fc_dates"]],
                                            names=["product_card_id", "date"])
            ).fillna(0).to_numpy(dtype=float)
            out_metrics.append({
                "product_card_id": pid,
                "n_obs": int(horizon),
                "smape": _smape(actual_series, res["q50"]),
                "wape": _wape(actual_series, res["q50"]),
                "coverage_80": _coverage(actual_series, res["q10"], res["q90"]),
                "model": name, "slice": slice_name, "cohort": "A_active",
            })
    return {
        "forecasts": pd.concat(out_forecasts, ignore_index=True),
        "metrics": pd.DataFrame(out_metrics),
    }


def run():
    log.info("loading daily panel + meta...")
    panel = pd.read_parquet(PANEL_PATH)
    meta = pd.read_parquet(META_PATH)
    cohort_a = meta.loc[meta["cohort"] == "A_active", "product_card_id"].tolist()
    log.info("cohort A: %d products", len(cohort_a))

    # Daily disaster exog (history portion of m3_disaster_daily)
    try:
        m3_d = pd.read_parquet(FC_DIR / "m3_disaster_daily.parquet")
        exog_panel = m3_d[m3_d["data_type"] == "actual"][
            ["product_card_id", "date", "disaster_index"]].copy()
        log.info("loaded daily disaster_index exog: %d rows", len(exog_panel))
    except FileNotFoundError:
        exog_panel = None
        log.info("no m3_disaster_daily.parquet; SARIMAX falls back to SARIMA")

    snaive = SeasonalNaiveDaily(season=SEASON_M)
    sarima = SARIMAForecaster(seasonality=SEASON_M,
                                candidate_orders=DAILY_SARIMA_ORDERS)
    sarimax = SARIMAXForecaster(seasonality=SEASON_M,
                                  candidate_orders=DAILY_SARIMA_ORDERS)

    forecasters = [
        ("seasonal_naive", snaive),
        ("sarima",         sarima),
        ("sarimax",        sarimax),
    ]

    # TimesFM runs via batched inference once per slice. Daily history is
    # ~730 obs for val and ~910 for test — well below TimesFM's 2048 context.
    # Horizon ≤ 181 days; max_horizon must cover it.
    tfm = TimesFMForecaster(max_context=1024, max_horizon=max(VAL_HORIZON, TEST_HORIZON))

    all_fc, all_metrics = [], []
    for slice_name, origin, horizon in [
        ("val",  VAL_ORIGIN,  VAL_HORIZON),
        ("test", TEST_ORIGIN, TEST_HORIZON),
    ]:
        bundle = _run_slice(panel, cohort_a, forecasters,
                              origin, horizon, slice_name, exog_panel)
        all_fc.append(bundle["forecasts"])
        all_metrics.append(bundle["metrics"])

        # TimesFM (batched, separate code path)
        log.info("model=timesfm — batched daily forecast %d products, h=%dd",
                 len(cohort_a), horizon)
        tfm_fc = forecast_panel_timesfm_daily(panel, tfm, origin, horizon, cohort_a)
        tfm_fc["slice"] = slice_name
        tfm_fc["cohort"] = "A_active"
        all_fc.append(tfm_fc[["product_card_id", "date", "q10", "q50", "q90",
                                "horizon", "model", "slice", "cohort"]])
        # Per-product metrics
        actuals_idx = (panel[(panel["date"] > origin)
                              & (panel["date"] <= origin + pd.Timedelta(days=horizon))]
                         .set_index(["product_card_id", "date"])["gross_qty"])
        for pid in cohort_a:
            sub = tfm_fc[tfm_fc["product_card_id"] == pid].sort_values("date")
            a_series = actuals_idx.reindex(pd.MultiIndex.from_product(
                [[pid], sub["date"].tolist()],
                names=["product_card_id", "date"]
            )).fillna(0).to_numpy(dtype=float)
            all_metrics.append(pd.DataFrame([{
                "product_card_id": pid,
                "n_obs": int(horizon),
                "smape": _smape(a_series, sub["q50"].to_numpy()),
                "wape":  _wape(a_series, sub["q50"].to_numpy()),
                "coverage_80": _coverage(a_series,
                                          sub["q10"].to_numpy(),
                                          sub["q90"].to_numpy()),
                "model": "timesfm", "slice": slice_name, "cohort": "A_active",
            }]))

    fc_df = pd.concat(all_fc, ignore_index=True)
    mp_df = pd.concat(all_metrics, ignore_index=True)

    # Portfolio metrics
    actuals = panel[["product_card_id", "date", "gross_qty"]]
    bt = fc_df.merge(actuals, on=["product_card_id", "date"], how="left")
    bt["gross_qty"] = bt["gross_qty"].fillna(0)

    portfolio_rows = []
    for (slice_name, model), sub in bt.groupby(["slice", "model"]):
        portfolio_rows.append({
            "n_obs": len(sub),
            "smape_mean": _smape(sub["gross_qty"], sub["q50"]),
            "wape":       _wape(sub["gross_qty"], sub["q50"]),
            "coverage_80": _coverage(sub["gross_qty"], sub["q10"], sub["q90"]),
            "slice": slice_name, "model": model, "cohort": "A_active",
        })
    po_df = pd.DataFrame(portfolio_rows)

    FC_DIR.mkdir(parents=True, exist_ok=True)
    fc_df.to_parquet(FC_DIR / "m1_demand_daily.parquet", index=False)
    mp_df.to_parquet(FC_DIR / "m1_demand_daily_metrics.parquet", index=False)
    po_df.to_parquet(FC_DIR / "m1_demand_daily_portfolio.parquet", index=False)

    log.info("wrote daily backtest parquets")
    print("\n===== PORTFOLIO HEADLINE (cohort A_active) =====")
    print(po_df.sort_values(["slice", "model"]).round(4).to_string(index=False))


if __name__ == "__main__":
    run()
