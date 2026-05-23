"""Shared LightGBM binary classifier with isotonic calibration.

Used by M3a (fraud), M3b (cancel), M3c (late delivery). All three are
imbalanced binary classification on the same order-level feature set, so
they share the entire pipeline. Per-target overrides go through `params`.

Outputs the standard set of metrics + a calibrated probability column that
downstream code can aggregate to product-month rates.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# Defaults tuned for ~2% positive rate, ~125k train rows
DEFAULT_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
    "metric": "binary_logloss",
}


@dataclass
class TrainedModel:
    booster: lgb.Booster
    calibrator: IsotonicRegression | None
    feature_names: list[str]
    best_iter: int


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, p)) if y_true.sum() > 0 else float("nan"),
        "pr_auc":  float(average_precision_score(y_true, p)) if y_true.sum() > 0 else float("nan"),
        "log_loss": float(log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7))),
        "brier":   float(brier_score_loss(y_true, p)),
        "base_rate": float(y_true.mean()),
        "n":        int(len(y_true)),
    }


def train_classifier(
    data: dict,
    params: dict | None = None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 50,
    calibrate: bool = True,
) -> tuple[TrainedModel, dict]:
    """data: output of `risk.data.split_xy`. Returns (model, metrics_dict)."""
    use_params = {**DEFAULT_PARAMS, **(params or {})}
    feat = data["feature_cols"]

    train_set = lgb.Dataset(data["train"]["X"], label=data["train"]["y"],
                            categorical_feature="auto", free_raw_data=False)
    val_set = lgb.Dataset(data["val"]["X"], label=data["val"]["y"],
                          categorical_feature="auto", reference=train_set,
                          free_raw_data=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        booster = lgb.train(
            use_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[train_set, val_set],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

    # Predict on each split (uncalibrated)
    raw_preds = {
        name: booster.predict(data[name]["X"], num_iteration=booster.best_iteration)
        for name in ("train", "val", "test")
    }

    # Isotonic calibration fit on val (predictions only; orthogonal to GBM training)
    calibrator: IsotonicRegression | None = None
    if calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        calibrator.fit(raw_preds["val"], data["val"]["y"])

    cal_preds = {
        name: (calibrator.predict(raw_preds[name]) if calibrator is not None else raw_preds[name])
        for name in raw_preds
    }

    metrics = {
        "uncalibrated": {name: _metrics(data[name]["y"], raw_preds[name]) for name in raw_preds},
        "calibrated":   {name: _metrics(data[name]["y"], cal_preds[name]) for name in cal_preds},
        "feature_importance": pd.DataFrame({
            "feature": feat,
            "importance_split": booster.feature_importance(importance_type="split"),
            "importance_gain":  booster.feature_importance(importance_type="gain"),
        }).sort_values("importance_gain", ascending=False).reset_index(drop=True),
        "predictions": {name: cal_preds[name] for name in cal_preds},
        "raw_predictions": raw_preds,
    }

    return TrainedModel(booster=booster, calibrator=calibrator,
                        feature_names=feat, best_iter=booster.best_iteration), metrics


def aggregate_to_product_month(
    predictions: np.ndarray,
    order_dates: pd.Series,
    product_ids: pd.Series,
    actuals: np.ndarray | None = None,
) -> pd.DataFrame:
    """Aggregate order-level probabilities to a product-month rate.

    Returns columns:
        product_card_id, year_month, n_orders,
        predicted_rate (mean of calibrated p),
        actual_rate (if actuals given)
    """
    df = pd.DataFrame({
        "product_card_id": product_ids.to_numpy(),
        "year_month": order_dates.dt.to_period("M").dt.to_timestamp().to_numpy(),
        "p": predictions,
    })
    agg = df.groupby(["product_card_id", "year_month"], sort=True).agg(
        n_orders=("p", "size"),
        predicted_rate=("p", "mean"),
    ).reset_index()
    if actuals is not None:
        df["actual"] = actuals
        actual_agg = df.groupby(["product_card_id", "year_month"])["actual"].mean().reset_index()
        actual_agg.columns = ["product_card_id", "year_month", "actual_rate"]
        agg = agg.merge(actual_agg, on=["product_card_id", "year_month"], how="left")
    return agg
