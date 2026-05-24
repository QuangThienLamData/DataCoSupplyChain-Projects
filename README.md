# DataCo Supply Chain — Dashboard Insights

## Adhoc Analysis

End-to-end EDA, root-cause analysis, and 6-month forward forecast on the DataCo Global supply-chain dataset (180,519 orders, Jan 2015 – Jan 2018).

---

### 1. EDA — Four business problems surfaced (`eda.ipynb`)

| # | Problem | Headline |
|---|---|---|
| 1 | **Late deliveries** | **54.8%** of orders arrive late — the dominant operational issue |
| 2 | **Suspected fraud** | **4,062 orders (2.3%)** flagged, **100% via TRANSFER** payment, concentrated in LATAM & Europe |
| 3 | **Profitability** | **18.7%** of orders are loss-making; ~10% discounting is applied uniformly across categories with no targeting |
| 4 | **Stuck / cancelled orders** | **33.3%** of orders stuck in `PENDING`/`PENDING_PAYMENT`, 2% cancelled — payment-gateway or fulfillment failures |

---

### 2. Root-cause analysis — Late deliveries (`analysis.ipynb`)

Seven hypotheses tested against an issue tree (location, product, customer segment, fraud, hour, distance, schedule). **Six rejected, one accepted.**

#### Rejected — none of these discriminate

| Branch | Late-rate spread (First / Second Class) | Verdict |
|---|---|---|
| Geography (country / market / region) | 83 – 88% | flat |
| Product / department | 83 – 87% | flat |
| Customer segment | 84 – 85% | flat |
| Fraud / cancel | non-fraud 87–89%; fraud is 100% cancelled, never late | not a cause |
| Order hour of day | 84 – 85% | no effect |
| Cross-border vs domestic | 100% cross-border; Pearson r ≈ 0.006 with distance | constant, not a variable |

An OLS regression on every operational dimension (HC3 robust SE, all assumptions checked) confirms the issue-tree result in one shot: **only `is_second_class` is significant** (coef ≈ +1.0 day, p < 0.001). R² ≈ 0.18 — the rest is unstructured noise.

#### Accepted — the shipping-promise model itself is broken (Over-optimistic estimation)

- **First Class** — every single order is scheduled at **1 day** but actually arrives in **2 days**. Deterministic +1-day miss, zero variance. Schedule is structurally wrong.
- **Second Class** — promised at **2 days** (the best case), but actual transit time is spread across 2–6 days. Only ~20% hit the promise.
- **Standard Class** — promised at 4 days, actually beats its schedule, because the quote already reflects realistic transit time.

<img width="768" height="219" alt="image" src="https://github.com/user-attachments/assets/4b73dbeb-e863-4b11-9140-65b7f89db6b4" />

<img width="523" height="317" alt="image" src="https://github.com/user-attachments/assets/b619548d-e760-4d42-acc0-8d8c40a2245c" />

Recalculate the R2/MSE of the prediction value. I get R2 = 26% ==> Poor estiamtion values for shipping date.

<img width="285" height="106" alt="image" src="https://github.com/user-attachments/assets/cbb15f31-1465-4302-9a77-d1b3441e25fa" />


**Root cause:** the promise is calibrated to best-case, not P80/P90 of observed transit time. **Fix:** re-baseline First Class to 2 days; quote Second Class on a realistic percentile or as an interval.

## Insights from Dashboard

### 1. Revenue Collapse in Late 2017: The Hurricane Effect

Total revenue experienced a sharp decline beginning in October 2017. This drop was not driven by internal business factors — it was the direct consequence of **Hurricanes Irma and Maria**, two of the most destructive Atlantic hurricanes on record, striking in September 2017.

![Revenue trend chart](https://github.com/user-attachments/assets/85921fa2-d07e-480c-bab1-b20ac300c56b)

---

### 2. The Revenue–Order Paradox

The aftermath of the hurricanes revealed a counterintuitive pattern: **revenue fell sharply while order volume surged**. On the surface, this appears contradictory — more orders should mean more revenue. The explanation lies in a dramatic shift in what customers were buying.

![Order vs Revenue chart](https://github.com/user-attachments/assets/8c823f17-f2e0-4ee1-90bf-2a748c80a103)

Average quantity per order dropped by **70%** and average order value dropped by **35%** during this period. Prior to the hurricanes, the highest-revenue categories were **outdoor and sporting goods** — high-ticket items customers purchase in bulk. Once the storms hit, demand for these categories collapsed to near zero, as outdoor activities became impossible. 

The LATAM market -- the market which exposed to two hurricane, experienced corrupted demand.

The only category sustaining meaningful demand was **Electronic Devices**. While electronics carry a high average order value, customers typically purchase just one unit at a time to meet an immediate need — providing no volume uplift to offset the loss elsewhere.

![Category breakdown](https://github.com/user-attachments/assets/43a0cfb2-967d-420b-b507-4bd1505f6fce)
<img width="1692" height="314" alt="image" src="https://github.com/user-attachments/assets/87cbad1f-d807-46f7-81e8-89479d0c5f35" />

> **Business implication:** Revenue concentration in weather-sensitive categories (outdoor/sporting goods) creates structural vulnerability to climate events. Diversifying the high-revenue category mix would reduce this exposure.

---

### 3. Optimal Discount Range: 1–10%

Analysis of discount depth against profit margin and order value reveals a clear sweet spot.

<img width="753" height="232" alt="image" src="https://github.com/user-attachments/assets/176fd76b-fad5-4b5c-b5d5-29f6e164961e" />

Discounts in the **1–10% range** consistently achieve the best trade-off: they are sufficient to encourage customers to increase their order value, while preserving an acceptable profit margin. Discounts beyond this threshold provide diminishing returns — order values do not increase proportionally, and margin erosion accelerates.

> **Recommendation:** Standardise promotional discount rates within the 1–10% band. Discounts above 10% should require approval and be reserved for strategic clearance scenarios only.

---

### 4. Fraud Concentration: Pet Supplies at Pacific Asia — Corporate Segment

Aggregated fraud metrics can mask highly localised risk patterns. Drilling into the **Corporate customer segment by category and market** surfaces a critical anomaly.

![Fraud heatmap — Corporate segment](https://github.com/user-attachments/assets/805c6aac-6c3f-4e7c-8b09-ddfcd93160c4)

The **Pet Shop category in the Pacific Asia market** carries a fraud rate of **6%** — approximately **three times the platform-wide average of 2.3%**. Two factors make this especially significant:

- This category is **exclusively present in the Pacific Asia market**, meaning its fraud risk is entirely concentrated in one geography with no offsetting volume elsewhere.
- The anomaly only surfaces when filtering to the Corporate segment, suggesting the fraud pattern is **segment-specific** rather than a broad market issue.

> **Recommendation:** Flag all Corporate-segment Pet Shop transactions in Pacific Asia for enhanced review. Consider temporarily requiring manual approval for this category-market-segment combination while an investigation is conducted.

---

### 5. Health & Beauty at Pacific Asia: A Compounding Risk Profile

The Pacific Asia market presents a second, overlapping risk in the **Health & Beauty category** that compounds the concerns raised above.

![Health & Beauty fraud analysis](https://github.com/user-attachments/assets/5719c822-d596-4999-9edf-e294f04b70bc)

Within this category at Pacific Asia, two metrics stand out:

- **20% of on-time delivered orders are flagged as suspected fraud** — meaning timely delivery is providing no signal of legitimacy and may even be used to avoid scrutiny.
- **33% of orders in this category result in a financial loss**, suggesting that fraudulent activity is being combined with deep discounting or return abuse to extract value from the business.

The combination of high fraud incidence and high loss-making rate in the same category and market points to a **structured exploitation pattern** rather than isolated incidents.

> **Recommendation:** Suspend or tighten controls on Health & Beauty transactions in Pacific Asia immediately. Delivery status alone should not be used as a fraud proxy in this market. A joint review with the fraud and finance teams is warranted to assess total financial exposure and determine whether chargebacks or recoveries are possible.

---

### Summary of Key Actions

| # | Finding | Priority | Recommended Action |
|---|---|---|---|
| 1 | Revenue collapse driven by hurricane-related demand shift | Medium | Diversify revenue mix away from weather-sensitive categories |
| 2 | 70% drop in avg qty/order post-Oct 2017 | Medium | Build demand-resilience into category planning |
| 3 | 1–10% discount is the optimal range | High | Cap standard discounts at 10%; require approval above threshold |
| 4 | Pet Shop fraud rate 3× average in Pacific Asia (Corporate) | Critical | Immediate review of Corporate Pet Shop orders in Pacific Asia |
| 5 | Health & Beauty: 20% fraud + 33% loss rate in Pacific Asia | Critical | Tighten controls; suspend high-risk transaction patterns |

## Dashboard Building
### Overall
<img width="1168" height="670" alt="image" src="https://github.com/user-attachments/assets/7de5eb0c-e501-4551-b524-c07c64f37c04" />

### Sale
<img width="1171" height="665" alt="image" src="https://github.com/user-attachments/assets/b8225fca-489a-48d4-8b3c-327159b753a8" />

### Demand
<img width="1169" height="664" alt="image" src="https://github.com/user-attachments/assets/08b94e25-3727-488d-ae86-bb14ec88d736" />

### Delivery
<img width="1171" height="662" alt="image" src="https://github.com/user-attachments/assets/02d5dc09-81b0-4c63-9b69-09cc2ae2b711" />

### Risk
<img width="1165" height="658" alt="image" src="https://github.com/user-attachments/assets/2373ffcd-8ea0-400c-a34c-240361aad005" />

### Forecast
<img width="1163" height="666" alt="image" src="https://github.com/user-attachments/assets/c1522b79-629e-4ecd-88ff-009f76280f5a" />



