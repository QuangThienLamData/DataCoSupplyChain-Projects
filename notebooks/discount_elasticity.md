# Discount → AOV / Quantity Elasticity + What-If

See `notebooks/discount_elasticity.ipynb` for the full code.

---

## 1. Non-linearity proof

Within-product FE + HC3 robust SE:

| Spec | β on `disc` | β on `disc²` | R² |
|---|---|---|---|
| **Linear** | −0.08 (p = 0.16) | — | 0.0000 |
| **Quadratic** | +12.75 (p < 1e−200) | **−58.32 (p < 1e−200)** | 0.094 |

The squared term's p-value (~0) decisively rejects the linear spec.

## 2. Non-linear basket formulas

```
log( AOV(d) / AOV(0) )       =  +12.75 · d  −  58.32 · d²
log( qty/order(d) / qty(0) ) =  +13.36 · d  −  60.79 · d²
```

Peak at **d\* ≈ 11%**.

## 3. Number-of-orders elasticity

From the monthly panel, within-product FE: `log(num_orders) ~ log(price)`

```
β_orders = −0.469   (p = 0.22)
```

So `num_orders(d) = num_orders(0) × (1 − d)^(−0.47)`.

A 10% discount → +5.1% more orders.

---

## 4. Hybrid what-if formula

Combining the two — orders elasticity drives **how many** customers, quadratic drives **how big** each basket:

```
Sales(d) / Sales(0)  =  (1 − d)^(−0.47)  ×  exp( 12.75·d − 58.32·d² )
                       └──── orders ────┘  └────── AOV per order ──────┘
```

Profit (with COGS unchanged, baseline gross margin m₀ = 12%):

```
margin(d)  =  (m₀ − d) / (1 − d)
Profit(d) / Profit(0)  =  Sales(d)/Sales(0)  ×  margin(d) / m₀
```

## 5. What-if table

| Discount | %Δorders | %ΔAOV | **%ΔSales** | New margin | **%ΔProfit** |
|---|---|---|---|---|---|
| 0% | 0% | 0% | 0% | +12.0% | 0% |
| 2% | +0.95% | +26% | +27% | +10.2% | **+8%** |
| **2.86%** ← profit max | **+1.4%** | **+37%** | **+39%** | **+9.4%** | **+9%** |
| 5% | +2.4% | +63% | +67% | +7.4% | +3% |
| **5.4%** ← profit breakeven | +2.6% | +69% | +73% | +7.0% | 0% |
| 8% | +4.0% | +91% | +99% | +4.3% | −28% |
| 10% | +5.1% | +100% | +110% | +2.2% | −61% |
| **11.4%** ← sales max | **+5.9%** | **+101%** | **+113%** | **+0.7%** | **−87%** |
| 12% | +6.2% | +100% | +112% | 0.0% | −100% (margin gone) |
| 15% | +7.9% | +82% | +97% | −3.5% | −158% |
| 20% | +11.0% | +24% | +38% | −10.0% | −215% |

## 6. Conclusion

| Goal | Optimal discount | Result |
|---|---|---|
| **Maximise sales** | **~11.4%** | Sales 2.12× baseline (profit −87%) |
| **Maximise profit** | **~2.9%** | Sales +39%, profit **+9%** |
| **Profit-breakeven vs baseline** | **~5.4%** | Beyond this, profit drops below baseline |
| **Margin breakeven** | **12.0%** | Beyond this, every order loses money |

**Two key thresholds:**

- **At ~3% discount**: profit is maximised — small AOV/orders lift, margin still healthy.
- **At ~5% discount**: profit equals baseline — volume lift exactly offsets margin loss.
- **Beyond 5%**: every extra discount point burns profit, even though sales keep growing until ~11%.

If the goal is **growth at all costs** (e.g., market-share play), 10–12% maximises top-line.
If the goal is **earnings**, the sweet spot is **2–3%**.

## Caveats

- Orders elasticity p = 0.22 — point estimate is +5% lift at 10% disc, but the data could support anything from 0 to +10%.
- Quadratic R² ≈ 9% — significant but most variation is product mix / occasion, not discount.
- COGS assumed fixed. If suppliers offer volume rebates at higher discounts (lower COGS), the profit curves shift up.
