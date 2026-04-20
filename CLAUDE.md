# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

**Python**: 3.14, virtual environment at `.scmvenv/`

**IMPORTANT — broken pip launcher**: `.scmvenv/Scripts/pip.exe` has a hardcoded path to a non-existent `.venv-1` and will fail. Always invoke pip as:
```bash
.scmvenv/Scripts/python.exe -m pip install <package>
```

Run scripts with:
```bash
.scmvenv/Scripts/python.exe script.py
```

## Dataset

**File**: `data/DataCoSupplyChainDataset.csv`
- Encoding: `latin1`
- 180,519 rows × 53 columns
- Field descriptions: `data/DescriptionDataCoSupplyChain.csv`

**Key column groups**:
- **Order**: `Order Id`, `order date (DateOrders)`, `Order Status`, `Order Region`, `Order Country`, `Order City`, `Market`
- **Shipping**: `shipping date (DateOrders)`, `Shipping Mode`, `Days for shipping (real)`, `Days for shipment (scheduled)`, `Delivery Status`, `Late_delivery_risk`
- **Financials**: `Sales`, `Order Item Total`, `Benefit per order`, `Order Profit Per Order`, `Order Item Profit Ratio`, `Order Item Discount`, `Order Item Discount Rate`
- **Customer**: `Customer Id`, `Customer Segment` (Consumer/Corporate/Home Office), `Customer City/Country/State`
- **Product**: `Product Name`, `Product Price`, `Category Name`, `Department Name`

**Delivery Status values**: `Advance shipping`, `Late delivery`, `Shipping canceled`, `Shipping on time`

**Order Status values**: `COMPLETE`, `PENDING`, `CLOSED`, `PENDING_PAYMENT`, `CANCELED`, `PROCESSING`, `SUSPECTED_FRAUD`, `ON_HOLD`, `PAYMENT_REVIEW`

**Markets**: Africa, Europe, LATAM, Pacific Asia, USCA

## Project Purpose

DataCo supply chain analytics project. The goal is to build Apache Superset dashboards that:
1. Describe the data (overview of sales, orders, customers, products by region/time/segment)
2. Surface operational problems (late deliveries, cancellations, fraud, low-margin orders, discount abuse)

## Superset Dashboard Setup

Three-step workflow (run once):

```bash
# 1. Load CSV into SQLite (produces data/dataco.db)
.scmvenv/Scripts/python.exe load_to_sqlite.py

# 2. Start Superset (requires Docker Desktop)
docker compose up -d
# Wait ~30s for initialization

# 3. Create the dashboard via Superset API
.scmvenv/Scripts/python.exe -m pip install requests
.scmvenv/Scripts/python.exe create_dashboard.py
```

Open `http://localhost:8088` (admin / admin) to view the result.

**Dashboard sections (15 charts total):**
- KPIs: Total Orders, Revenue, Profit, Late Deliveries, Suspected Fraud, Negative-Profit Orders
- Delivery Operations: status breakdown, status by shipping mode, order status dist.
- Revenue Analysis: monthly trend, top-10 categories, revenue by market
- Operational Problems: shipping delay dist., delivery by segment, problem orders by market

## data_download.py

Downloads the dataset from Kaggle using `kagglehub`:
```bash
.scmvenv/Scripts/python.exe data_download.py
```
Requires Kaggle credentials configured (`~/.kaggle/kaggle.json`).
