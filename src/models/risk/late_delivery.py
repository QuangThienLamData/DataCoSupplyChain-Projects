"""M3c — Late-delivery sub-model.

Target: `is_late` ( = late_delivery_risk == 1 ). Base rate ~ 55% — balanced,
so this behaves more like ordinary binary classification than the imbalanced
fraud/cancel models. Useful signals: `shipping_mode`, `days_shipping_scheduled`,
destination geography, season.
"""
from __future__ import annotations

from src.models.risk.classifier import train_classifier
from src.models.risk.data import load_for_target


def train():
    data = load_for_target("is_late")
    # Balanced target → no scale_pos_weight needed
    params = {"learning_rate": 0.05, "num_leaves": 63}
    return train_classifier(data, params=params), data


if __name__ == "__main__":
    (model, metrics), data = train()
    print("LATE-DELIVERY — calibrated metrics:")
    for split, m in metrics["calibrated"].items():
        print(f"  {split}: AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"Brier={m['brier']:.4f}  base={m['base_rate']:.4f}")
    print("\nTop features:")
    print(metrics["feature_importance"].head(10).to_string(index=False))
