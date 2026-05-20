"""Layer 2 — Multivariate anomaly via IsolationForest.

For every (product, month) in the clean window, build a feature vector
combining quantity, price, risk, and disaster signals, then fit one
IsolationForest per cohort and score every observation. Rows with a
score in the bottom `contamination` percentile are flagged.

Why per-cohort? Cohort A (active) products have ~33 months of data, so a
single global forest works well. Cohort B (sparse) is fit jointly with
A so its rare records aren't all flagged as outliers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[3]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.parquet"
RISK_DRAG_PATH = ROOT / "forecasts" / "m3_risk_drag.parquet"
ELASTICITY_MONTHLY = ROOT / "forecasts" / "m2_elasticity_monthly.parquet"

# Features fed to the forest — engineered so they're roughly scale-free
FEATURES = [
    "log_qty", "log_p_eff", "discount_rate_avg",
    "p_fraud", "p_cancel", "p_late", "disaster_index",
    "elasticity",
    # Per-product deviations help: how unusual is THIS month relative to
    # the product's typical month?
    "qty_pct_dev", "p_eff_pct_dev",
]

CONTAMINATION = 0.05  # expect ~5% of months to be flagged
RANDOM_STATE = 17


def build_feature_frame(
    panel: pd.DataFrame,
    risk_drag: pd.DataFrame,
    elasticity: pd.DataFrame,
) -> pd.DataFrame:
    clean = panel[panel["data_quality"] == "ok"].copy()

    # Risk components — already at product-month
    clean = clean.merge(
        risk_drag[["product_card_id", "year_month",
                   "p_fraud", "p_cancel", "p_late", "disaster_index"]],
        on=["product_card_id", "year_month"], how="left",
    )
    # Elasticity series
    elast = elasticity[["product_card_id", "year_month", "elasticity"]]
    clean = clean.merge(elast, on=["product_card_id", "year_month"], how="left")

    # Engineered features
    clean["log_qty"] = np.log(clean["qty"].replace(0, np.nan))
    clean["log_p_eff"] = np.log(clean["p_eff"].replace(0, np.nan))
    # Per-product percentage deviation from rolling 12-month median
    for src, tgt in [("qty", "qty_pct_dev"), ("p_eff", "p_eff_pct_dev")]:
        med = (clean.sort_values(["product_card_id", "year_month"])
                    .groupby("product_card_id")[src]
                    .transform(lambda s: s.rolling(12, min_periods=3).median()))
        clean[tgt] = (clean[src] - med) / med.replace(0, np.nan)

    return clean


def fit_score(feat_frame: pd.DataFrame) -> pd.DataFrame:
    """Fit IsolationForest on the feature frame; return per-row scores."""
    X = feat_frame[FEATURES].copy()
    # Impute with column median so rows with NaN aren't dropped wholesale
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    iforest = IsolationForest(
        n_estimators=300,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iforest.fit(X)
    score = -iforest.score_samples(X)   # higher = more anomalous
    pred = iforest.predict(X) == -1

    out = feat_frame[["product_card_id", "year_month"]].copy()
    out["if_score"] = score
    out["if_flag"] = pred
    return out


def detect_iforest(
    panel: pd.DataFrame,
    risk_drag: pd.DataFrame,
    elasticity: pd.DataFrame,
) -> pd.DataFrame:
    feat = build_feature_frame(panel, risk_drag, elasticity)
    scored = fit_score(feat)
    hits = scored[scored["if_flag"]].copy()
    hits["layer"] = "iforest"
    hits = hits.rename(columns={"if_score": "score"})
    return hits[["product_card_id", "year_month", "score", "layer"]]
