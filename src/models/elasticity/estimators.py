"""Frequentist elasticity estimators for the M2 model.

Three views, increasing complexity / shrinkage:

1. **Pooled OLS w/ product fixed effects** — single global β shared across
   products. Most robust statistically; loses heterogeneity. Output: 1 number
   with standard error.

2. **Per-product OLS** — separate regression per product → product-specific
   β_p. High variance for products with little price variation. Output: one
   row per product.

3. **Mixed-effects (random slopes)** — statsmodels `MixedLM` with random
   intercept *and* random slope on `log_p_eff` per product. Empirical-Bayes
   shrinkage toward the population mean stabilises sparse products. Output:
   one row per product (BLUP estimate).

All three use the same RHS:
    log_qty ~ log_p_eff + log_p_eff_cat + log_qty_lag1 + C(month)

with month dummies absorbing seasonality and `log_qty_lag1` controlling for
demand persistence.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

FORMULA_FULL = (
    "log_qty ~ log_p_eff + log_p_eff_cat + log_qty_lag1 + C(month)"
)
FORMULA_NO_LAG = "log_qty ~ log_p_eff + log_p_eff_cat + C(month)"


def _safe_fit(df: pd.DataFrame, formula: str = FORMULA_FULL, robust: bool = True):
    """OLS fit with HC3 (robust) covariance; silences statsmodels chatter."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=df).fit(
            cov_type="HC3" if robust else "nonrobust"
        )
    return model


# ---------------------------------------------------------------------------
# 1. Pooled with product FE
# ---------------------------------------------------------------------------
def fit_pooled_fe(df: pd.DataFrame) -> dict:
    """Single elasticity across all products, with product fixed effects."""
    sub = df.dropna(subset=["log_qty_lag1"]).copy()
    sub["product_id_str"] = sub["product_card_id"].astype(str)
    model = _safe_fit(sub, FORMULA_FULL + " + C(product_id_str)")
    beta = float(model.params["log_p_eff"])
    se = float(model.bse["log_p_eff"])
    gamma = float(model.params.get("log_p_eff_cat", np.nan))
    return {
        "n_obs": int(model.nobs),
        "elasticity_own": beta,
        "elasticity_own_se": se,
        "elasticity_own_ci_lo": beta - 1.96 * se,
        "elasticity_own_ci_hi": beta + 1.96 * se,
        "elasticity_cross": gamma,
        "r2": float(model.rsquared),
        "r2_adj": float(model.rsquared_adj),
    }


# ---------------------------------------------------------------------------
# 2. Per-product OLS
# ---------------------------------------------------------------------------
@dataclass
class PerProductResult:
    product_card_id: float
    n_obs: int
    elasticity: float
    se: float
    ci_lo: float
    ci_hi: float
    elasticity_cross: float
    r2: float
    price_cv: float           # coefficient of variation of p_eff
    identifiable: bool        # True if price_cv > threshold AND n_obs adequate


def fit_per_product(
    df: pd.DataFrame,
    products: list | None = None,
    min_obs: int = 18,
    min_price_cv: float = 0.005,
) -> pd.DataFrame:
    """Per-product OLS. Returns one row per product. Products with too little
    price variation (cv < threshold) or too few observations are flagged
    `identifiable=False` and given NaN coefficients — callers should impute
    from the category mean or the pooled estimate.
    """
    if products is None:
        products = sorted(df["product_card_id"].unique())

    rows: list[PerProductResult] = []
    for pid in products:
        g = df[df["product_card_id"] == pid].dropna(subset=["log_qty_lag1"])
        cv = float(g["p_eff"].std() / g["p_eff"].mean()) if len(g) else float("nan")
        if len(g) < min_obs or not np.isfinite(cv) or cv < min_price_cv:
            rows.append(PerProductResult(
                product_card_id=pid, n_obs=len(g),
                elasticity=float("nan"), se=float("nan"),
                ci_lo=float("nan"), ci_hi=float("nan"),
                elasticity_cross=float("nan"), r2=float("nan"),
                price_cv=cv, identifiable=False,
            ))
            continue

        try:
            model = _safe_fit(g, FORMULA_FULL)
            beta = float(model.params["log_p_eff"])
            se = float(model.bse["log_p_eff"])
            rows.append(PerProductResult(
                product_card_id=pid, n_obs=int(model.nobs),
                elasticity=beta, se=se,
                ci_lo=beta - 1.96 * se, ci_hi=beta + 1.96 * se,
                elasticity_cross=float(model.params.get("log_p_eff_cat", np.nan)),
                r2=float(model.rsquared),
                price_cv=cv, identifiable=True,
            ))
        except Exception:
            rows.append(PerProductResult(
                product_card_id=pid, n_obs=len(g),
                elasticity=float("nan"), se=float("nan"),
                ci_lo=float("nan"), ci_hi=float("nan"),
                elasticity_cross=float("nan"), r2=float("nan"),
                price_cv=cv, identifiable=False,
            ))
    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# 3. Mixed-effects with random slopes (empirical Bayes shrinkage)
# ---------------------------------------------------------------------------
def fit_mixed_effects(df: pd.DataFrame) -> dict:
    """`log_qty ~ log_p_eff + ... + (1 + log_p_eff | product_card_id)`.

    Returns:
        {
          'population_elasticity': mean β,
          'population_se': std err,
          'per_product': DataFrame with BLUP β_p per product,
          'sigma_alpha2': variance of random intercepts,
          'sigma_beta2': variance of random slopes,
          'converged': bool,
        }
    """
    sub = df.dropna(subset=["log_qty_lag1"]).copy()
    sub["product_id_str"] = sub["product_card_id"].astype(str)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Note: statsmodels MixedLM only fits one random effect group; we
        # specify the random part via `re_formula='~log_p_eff'` for random slope.
        md = smf.mixedlm(
            FORMULA_FULL,
            data=sub,
            groups=sub["product_id_str"],
            re_formula="~log_p_eff",
        )
        try:
            res = md.fit(method="lbfgs", maxiter=200)
            converged = bool(res.converged)
        except Exception:
            res = md.fit(method="powell", maxiter=200)
            converged = False

    pop_beta = float(res.fe_params["log_p_eff"])
    pop_se = float(res.bse_fe["log_p_eff"])

    # Random effects: per-group {Intercept, log_p_eff} deviations from pop mean
    re = res.random_effects  # dict: group_id_str -> Series
    pp_rows = []
    for gid, eff in re.items():
        slope_dev = float(eff.get("log_p_eff", 0.0))
        pp_rows.append({
            "product_card_id": float(gid),
            "elasticity_blup": pop_beta + slope_dev,
        })
    pp_df = pd.DataFrame(pp_rows)

    # Var components
    cov_re = res.cov_re  # 2x2 random-effects cov
    try:
        sigma_alpha2 = float(cov_re.iloc[0, 0])
        sigma_beta2 = float(cov_re.iloc[1, 1])
    except Exception:
        sigma_alpha2 = float("nan"); sigma_beta2 = float("nan")

    return {
        "population_elasticity": pop_beta,
        "population_se": pop_se,
        "per_product": pp_df,
        "sigma_alpha2": sigma_alpha2,
        "sigma_beta2": sigma_beta2,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Helper: impute non-identifiable products from category mean
# ---------------------------------------------------------------------------
def impute_from_category(per_product: pd.DataFrame, frame: pd.DataFrame,
                          fallback: float) -> pd.DataFrame:
    """Fill NaN elasticities (non-identifiable products) with the mean of
    identifiable products in the same category; if category empty, use
    `fallback` (e.g. pooled or population estimate)."""
    pp = per_product.copy()
    cats = (frame[["product_card_id", "category_id"]]
            .drop_duplicates("product_card_id"))
    pp = pp.merge(cats, on="product_card_id", how="left")

    cat_mean = (pp[pp["identifiable"]]
                .groupby("category_id")["elasticity"].mean()
                .rename("elasticity_cat_mean"))
    pp = pp.merge(cat_mean, on="category_id", how="left")
    pp["elasticity_final"] = pp["elasticity"].fillna(pp["elasticity_cat_mean"])
    pp["elasticity_final"] = pp["elasticity_final"].fillna(fallback)
    pp["imputed"] = ~pp["identifiable"]
    return pp
