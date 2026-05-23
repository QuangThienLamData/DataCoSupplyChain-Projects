"""M3a — Fraud sub-model.

Target: `is_fraud` ( = order_status == 'SUSPECTED_FRAUD' ).
Base rate ~ 2.2%. We expect payment_type, customer_segment, and order
geography to dominate feature importance.
"""
from __future__ import annotations

from src.models.risk.classifier import train_classifier
from src.models.risk.data import load_for_target


def train():
    data = load_for_target("is_fraud")
    # Class imbalance: scale_pos_weight ≈ neg/pos ≈ 45
    params = {"scale_pos_weight": 45.0, "learning_rate": 0.05}
    return train_classifier(data, params=params), data


if __name__ == "__main__":
    (model, metrics), data = train()
    print("FRAUD — calibrated metrics:")
    for split, m in metrics["calibrated"].items():
        print(f"  {split}: AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"Brier={m['brier']:.4f}  base={m['base_rate']:.4f}")
    print("\nTop features:")
    print(metrics["feature_importance"].head(10).to_string(index=False))
