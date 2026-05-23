# DataCo Forecasting Stack — Model Summary

End-to-end forecasting + risk pipeline on the DataCo Supply Chain monthly product panel (2015-01 → 2018-01, 37 months, 118 products).

**Split**: train ≤ 2016-12 · val 2017-01 → 06 · test 2017-07 → 2018-01. The 7-month test window deliberately spans the full Atlantic 2017 hurricane season (Harvey + Irma + Maria) and the post-Maria PR economic collapse.

```
                  ┌────────────────────────────────────────────────────────┐
M3 Risk           │   fraud │ cancel │ late │ disaster (T1+T2+T3 + lag)   │
                  └────────────────────────────────────────────────────────┘
                                          │
M1 Demand         TimesFM (production) + SARIMA / ETS / seasonal-naive baselines
                                          │
M2 Elasticity     pooled-FE β ≈ −0.805  → planned-price multiplier
                                          │
M4 Sales          Monte Carlo: qty × price × (1 − risk_drag) → q10 / q50 / q90
                                          │
M5 Anomaly        STL · IsolationForest · forecast-deviation → severity alerts
```

---

## M1 — Demand forecast

| | |
|---|---|
| Target | `gross_qty` (all orders, pre-fraud/cancel) |
| Production | **TimesFM 2.5** (Google foundation model, 200M params) |
| Baselines | seasonal-naive · ETS (Holt-Winters) · **SARIMA** (Box-Jenkins) |
| Cohorts | A_active (n=54, ≥18 train months) · B_sparse (category fallback) |
| Output | q10 / q50 / q90 per (product, month) |

**Backtest (cohort A):**

| Slice | seasonal-naive | ETS | SARIMA | **TimesFM** |
|---|---|---|---|---|
| val (calm) — WAPE | 0.130 | 0.130 | 0.141 | **0.117** |
| val cov₈₀ | 0.87 | 0.87 | 0.50 | 0.59 |
| test (hurricane) — WAPE | 1.703 | 1.703 | **1.469** | 1.496 |
| test cov₈₀ | 0.67 | 0.67 | 0.75 | **0.87** |

TimesFM wins the calm val by 17%. SARIMA edges TimesFM on test (the hurricane-driven qty crash is closer to SARIMA's homoskedastic assumption with a step). Both struggle on test in absolute terms — M1 is intentionally univariate, the disaster damping in M3+M4 is what recovers operational accuracy.

**Classical assumption tests** (`forecasts/m1_ts_assumptions.parquet`, cohort A):

| Test | % pass | H0 |
|---|---|---|
| ADF (stationarity) | 43% | unit root |
| KPSS (stationarity) | 93% | stationary |
| Hyndman-Wang seasonality > 0.6 | 69% | — |
| Ljung-Box (white residuals) | 69% | white noise |
| Jarque-Bera (normal residuals) | 83% | normal |

Per-product recommendation: 36 SARIMA · 16 ETS · 2 TimesFM. We still ship TimesFM in production because it wins calm-period WAPE, has cleaner CI coverage, and is robust to structural breaks (no per-product assumption testing needed). See `notebooks/m1_assumptions.ipynb`.

**Metric meanings.** WAPE = Σ|forecast − actual| / Σ|actual| — interpret as "off by X% of total volume". sMAPE bounded 0–2, symmetric. Coverage₈₀ = fraction of actuals inside (q10, q90); a well-calibrated model lands near 0.80.

---

## M2 — Price elasticity

| | |
|---|---|
| Model | log(qty) = α + β · log(p_eff) + product/month FE (pooled fixed-effects) |
| Headline | **β = −0.805 (SE 0.412)** — a 10% price cut lifts demand ~8.1% |
| Diagnostic | per-product OLS + mixed-effects random slopes |

**Why pooled, not per-product.** Only price variation in this dataset is discount level; products don't have natural price experiments. Per-product β is not statistically identifiable. Mixed-effects confirms random-slope variance is small. M4 uses the single pooled β.

---

## M3 — Risk (4 components feed into a single `risk_drag`)

$$ \text{risk\_drag} = 1 − (1 − p_\text{fraud})(1 − p_\text{cancel})(1 − \text{LATE} \cdot p_\text{late})(1 − \text{DISASTER} \cdot \text{disaster\_drag\_index}) $$

### M3a-c — Fraud / Cancellation / Late delivery

| | Model | ROC-AUC | PR-AUC | Brier | Base rate |
|---|---|---|---|---|---|
| Fraud | LightGBM + isotonic | **0.881** | 0.093 | 0.022 | 2.4% |
| Cancellation | LightGBM + isotonic | **0.871** | 0.080 | 0.020 | 2.2% |
| Late delivery | LightGBM + isotonic | 0.729 | **0.775** | 0.197 | 55.2% |

Fraud and cancel are largely driven by `payment_type == 'TRANSFER'` (dominant feature). Late is noisier (logistics-driven, not order-property).

### M3d — Disaster (3-tier storm-prediction stack)

| Tier | Lead time | Source | Role |
|---|---|---|---|
| **T1 historical** | post-event | NOAA HURDAT2 best-track | Calibrates DAMPING; provides per-region severity |
| **T2 operational** | **5–7 days** | NHC active-storms cone + cone-uncertainty | Per-region per-day forward severity during hurricane season |
| **T3 seasonal** | **1–6 months** | NOAA CPC seasonal outlook | Climatology × multiplier (0.7 / 1.0 / 1.3 / 1.5) |

**Tier 2 backtest on Irma (Sep 6–7, 2017 PR landfall):**

| As-of date | PR forward severity | Lead time |
|---|---|---|
| 2017-09-01 | 0.20 | 5–7 days, low confidence |
| **2017-09-05** | **0.78** | 24–48h, high confidence |
| 2017-09-06 | **0.87** | ~24h peak warning |

### Two-field severity vs revenue-lag architecture

| Field | Peak (Maria 2017) | Used by |
|---|---|---|
| `disaster_index` | **Sep (landfall)** | Reporting / dashboards / M5 attribution |
| `disaster_drag_index` | **Nov (M+2)** | M4 sales drag math |

Both derive from the same storm impulse via different impulse-response kernels:
- Severity kernel `LAG_PROFILE = [1.00, 0.80, 0.45, 0.25, 0.10]` (peak at landfall, decay)
- Revenue-lag kernel `REVENUE_LAG_WEIGHTS = [0.05, 0.20, 0.80]` (peak at M+2 — pre-storm orders ship in Sep; reorders blocked for weeks; revenue collapses Nov–Dec)

**Calibration:** `DISASTER_DAMPING = 0.873, R² = 0.568` (n=18 known-event rows, fit on `disaster_drag_index`). `LATE_DAMPING = 0.10` (industry prior; no in-data signal).

---

## M4 — Sales forecast (revenue $)

| | |
|---|---|
| Construction | Monte Carlo (2000 draws): qty (Lognormal from M1 quantiles) × elasticity-adjusted price × (1 − risk_drag) |
| Output | q10 / q50 / q90 / mean / std per (product, month) in **3 views** |

**Three views:**
1. **pre-risk** — backtest target `gross_revenue`; M1 + M2 only.
2. **historical-risk** — pre-risk × (1 − fraud)(1 − cancel); backtest target `revenue_realized`.
3. **forward-risk-adj** — pre-risk × full risk_drag (with Tier-2+3 forward disaster + revenue lag); operational view.

**Backtest WAPE (test = 7-month hurricane window):**

| View | val (calm) | test (hurricane) | Notes |
|---|---|---|---|
| pre-risk vs `gross_revenue` | **0.120** | 1.314 | M1 + M2 ceiling |
| historical-risk vs `revenue_realized` | **0.120** | 1.352 | + observed fraud/cancel |
| **forward-risk-adj** vs `revenue_realized` | **0.165** | **1.181** | Tier-2+3 forward + revenue lag |
| forward-risk-adj (legacy T1 historical) | 0.568 | 1.273 | reference only |

The Tier-2+3 forward + revenue-lag substitution improves the forward view by 7–71% WAPE vs the legacy Tier-1 historical disaster. The remaining test WAPE 1.18 reflects M1's univariate inability to predict the post-Maria qty crash — even the best disaster model can't fully recover when M1's pre-risk forecast is itself ~13× too high.

---

## M5 — Anomaly detection

| Layer | Method | Hits |
|---|---|---|
| L1 STL residual | per-product STL on qty / p_eff / revenue / elasticity, flag \|z\| > 3 | 391 |
| L2 IsolationForest | multivariate over 9 features, contamination 5% | 194 |
| L3 Forecast-deviation | actual outside M1's P5–P95, disaster-widened, ≥2 consecutive | 318 |

**Severity rule.** Critical = 3 layers OR (2 layers ∧ z > 5) · Warn = 2 layers OR z > 4.5 · Info = 1 layer.

**Validation (hurricane re-detection):** all 4 critical alerts land on Sep 2017 with `known_disaster=True` and `suspected_driver='disaster-known'`. Three independent layers agreed on the hurricane month — the system correctly *expects* it (so it can be filtered out for true-unexpected investigation).

---

## Forward forecast pipeline (Feb–Jul 2018, 3 scenarios)

**Flow:** M3 → M1 → M2 → M4. Each model produces history + 3 forecast scenarios with q10/q50/q90 bands. All outputs written to parquet files with a `data_type` column distinguishing actuals from forecasts.

**Scenario configuration** (`src/models/forecast_pipeline.py`):

| Scenario | M3 revenue-lag kernel | M2 elasticity β | Reading |
|---|---|---|---|
| pessimistic | `[0.05, 0.15, 0.40, 0.30, 0.20, 0.10]` 6-month tail | −0.39 (β + SE) | Slow recovery, less price-responsive |
| baseline | `[0.05, 0.20, 0.80]` calibrated 3-month | −0.805 (pooled-FE) | Model-consistent default |
| optimistic | `[0.05, 0.10, 0.30]` shorter tail | −1.22 (β − SE) | Fast recovery, more price-responsive |

**M4 sales — 6-month totals (Feb–Jul 2018):**

| Scenario | q10 | q50 | q90 |
|---|---|---|---|
| Pessimistic | $2.66M | **$4.99M** | $9.41M |
| Baseline ⭐ | $2.88M | **$5.39M** | $10.14M |
| Optimistic | $2.88M | **$5.39M** | $10.14M |

Baseline ≈ optimistic because the M3 drag model says Maria's residual decays to ~0 by April. Pessimistic carries a 6-month kernel and stays meaningfully lower through Feb-Mar before converging.

**Output parquet files** (`forecasts/`):

| File | Rows | Schema |
|---|---|---|
| `m3_pipeline.parquet` | 5,680 | product, year_month, data_type, scenario, disaster_index, disaster_drag_index, actual_disaster_index |
| `m1_pipeline.parquet` | 2,970 | product, year_month, data_type, scenario, q10, q50, q90, actual_gross_qty |
| `m2_pipeline.parquet` | 55 | year_month, data_type, scenario, elasticity_q10/q50/q90 |
| `m4_pipeline.parquet` | 2,970 | product, year_month, data_type, scenario, q10, q50, q90, actual_revenue_realized |

History rows: `data_type='actual'`, `scenario=None`, actual value filled. Forecast rows: `data_type='forecast'`, scenario set, q10/q50/q90 filled. See `notebooks/forecast_pipeline.ipynb` for full visualization.

---

## How to run

```bash
# Build the monthly panel from raw data
python -m src.features.build_panel

# Train models (run in order or independently after panel is built)
python -m src.models.demand.run_backtest      # M1 — TimesFM + baselines
python -m src.models.elasticity.run            # M2 — pooled-FE β
python -m src.models.risk.run                  # M3 — fraud/cancel/late + disaster
python -m src.models.sales.calibrate_damping   # DISASTER_DAMPING calibration
python -m src.models.sales.run                 # M4 — sales backtest
python -m src.models.anomaly.run               # M5 — anomaly alerts
python -m src.models.demand.assumptions        # TS assumption diagnostics

# Forward forecast pipeline (3 scenarios, Feb-Jul 2018)
python -m src.models.forecast_pipeline
```

**Storm-prediction stack** (used during production hurricane season):

```bash
# Refresh HURDAT2 historical (annual)
python -m src.data.hurdat2_ingest

# Live NHC active-storms (daily during hurricane season)
python -m src.models.sales.forward_forecast       # use_live=True for production

# Tier 3 seasonal outlook — edit SCENARIO_CONFIG in seasonal_outlook.py
# when NOAA CPC publishes the May / August update
```

## Repository layout

```
src/
├── data/        hurdat2_ingest · nhc_active_storms
├── features/    build_panel · region_centroids · storm_exposure · forward_exposure
└── models/
    ├── demand/         M1 — TimesFM + baselines + TS assumption tests
    ├── elasticity/     M2 — pooled-FE β
    ├── risk/           M3 — fraud / cancel / late + disaster (HURDAT2 v2 + Tier 2 + Tier 3)
    ├── sales/          M4 — sales MC + revenue-lag + damping calibration + forecast_pipeline
    └── anomaly/        M5 — STL + IsolationForest + forecast-deviation

notebooks/    one validation notebook per model (executed end-to-end, with plots)
forecasts/    all model outputs (parquet)
data/         processed panel + HURDAT2 raw text
```

## Known limitations

1. **Test WAPE is high (1.18 for M4 forward-risk-adj)** because M1 is univariate and can't predict the post-Maria qty crash; M3's customer-mix-weighted damping captures part of the impact but not all.
2. **DISASTER_DAMPING calibrated on n=18** known-event rows (R² 0.57). Point estimate is sound but SE is non-trivial.
3. **Tier 3 climatology** is built from the same 2014-2017 window we test on. Production deployment should use 10-15 years of HURDAT2.
4. **Revenue-lag kernel** has 3 lags → model says Maria drag is ~0 by April 2018. For PR specifically, real economic recovery took longer; the pessimistic scenario hedges this.
5. **2018 hurricane season** Tier-3 outlook is a "Near Normal" stub (NOAA publishes May); refresh once available.
