"""Run the full M2 elasticity pipeline and write artifacts.

Outputs
-------
- forecasts/m2_elasticity_summary.parquet
    one row per product, with:
        elasticity (per-product OLS), se, ci_lo, ci_hi,
        elasticity_blup (mixed-effects shrunk),
        elasticity_final (imputed where non-identifiable),
        identifiable flag, n_obs, price_cv, r2
- forecasts/m2_elasticity_pool.parquet
    one row, the pooled-FE headline estimate (β_pool, SE, CI, R²).
- forecasts/m2_elasticity_monthly.parquet
    long-form ε_{p, month} time series for downstream M4 consumption.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.models.elasticity.estimators import (
    fit_mixed_effects,
    fit_per_product,
    fit_pooled_fe,
    impute_from_category,
)
from src.models.elasticity.features import build_elasticity_frame, cohort_products
from src.models.elasticity.rolling import build_monthly_elasticity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
OUT_SUMMARY = ROOT / "forecasts" / "m2_elasticity_summary.parquet"
OUT_POOL = ROOT / "forecasts" / "m2_elasticity_pool.parquet"
OUT_MONTHLY = ROOT / "forecasts" / "m2_elasticity_monthly.parquet"


def run() -> None:
    log.info("building elasticity feature frame")
    frame = build_elasticity_frame()
    A = cohort_products("A_active")
    fA = frame[frame["product_card_id"].isin(A)]
    log.info("cohort A frame: %s rows, %d products", fA.shape, fA["product_card_id"].nunique())

    # 1. Pooled FE (the production headline)
    log.info("fitting pooled OLS with product fixed effects")
    pool = fit_pooled_fe(fA)
    pool_df = pd.DataFrame([pool])
    log.info("  pooled elasticity = %.4f (SE %.4f, R² %.3f)",
             pool["elasticity_own"], pool["elasticity_own_se"], pool["r2"])

    # 2. Per-product OLS
    log.info("fitting per-product OLS")
    per_product = fit_per_product(fA)
    log.info("  identifiable: %d/%d  median β=%.3f  mean β=%.3f",
             per_product["identifiable"].sum(),
             len(per_product),
             per_product.loc[per_product["identifiable"], "elasticity"].median(),
             per_product.loc[per_product["identifiable"], "elasticity"].mean())

    # 3. Mixed effects (identifiable products only, for stability)
    ident_ids = per_product.loc[per_product["identifiable"], "product_card_id"].tolist()
    log.info("fitting mixed-effects model on %d identifiable products", len(ident_ids))
    fA_i = fA[fA["product_card_id"].isin(ident_ids)]
    mix = fit_mixed_effects(fA_i)
    log.info("  pop_β=%.4f  σ²_β=%.4f  converged=%s",
             mix["population_elasticity"], mix["sigma_beta2"], mix["converged"])

    # Merge BLUPs back into per_product
    per_product = per_product.merge(
        mix["per_product"], on="product_card_id", how="left",
    )

    # 4. Impute non-identifiable from category mean / pooled estimate
    summary = impute_from_category(per_product, fA, fallback=pool["elasticity_own"])
    log.info("  imputed count: %d", int(summary["imputed"].sum()))

    # 5. Monthly time-varying series
    log.info("computing rolling-window monthly elasticity (12-mo window)")
    monthly = build_monthly_elasticity(frame, A, per_product)
    log.info("  monthly rows: %d  unique months: %d",
             len(monthly), monthly["year_month"].nunique())

    # Save
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUT_SUMMARY, index=False)
    pool_df.to_parquet(OUT_POOL, index=False)
    monthly.to_parquet(OUT_MONTHLY, index=False)
    log.info("wrote: %s, %s, %s", OUT_SUMMARY.name, OUT_POOL.name, OUT_MONTHLY.name)

    # Headline print
    print("\n===== M2 ELASTICITY HEADLINE =====")
    print(f"Pooled FE elasticity: {pool['elasticity_own']:+.3f}  "
          f"95% CI [{pool['elasticity_own_ci_lo']:+.3f}, {pool['elasticity_own_ci_hi']:+.3f}]  "
          f"R² {pool['r2']:.3f}")
    print(f"Mixed-effects pop:    {mix['population_elasticity']:+.3f}  "
          f"(SE {mix['population_se']:.3f}, identifiable only)")
    print(f"Per-product median:   {per_product.loc[per_product['identifiable'],'elasticity'].median():+.3f}")
    print(f"Non-identifiable products imputed from category mean: {int(summary['imputed'].sum())}")
    print("\nPer-product elasticity distribution (identifiable only):")
    print(summary.loc[~summary["imputed"], "elasticity_final"].describe().round(3))


if __name__ == "__main__":
    run()
