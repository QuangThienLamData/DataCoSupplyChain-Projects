"""
Loads DataCoSupplyChainDataset.csv into data/dataco.db (SQLite).
Run once before starting Superset.

Usage:
    .scmvenv/Scripts/python.exe load_to_sqlite.py
"""
import csv
import sqlite3
from datetime import datetime

CSV_PATH  = "data/DataCoSupplyChainDataset.csv"
DB_PATH   = "data/dataco.db"
TABLE     = "supply_chain"

# Mapping from raw CSV header -> clean SQL column name
COLUMN_MAP = {
    "Type":                          "payment_type",
    "Days for shipping (real)":      "days_shipping_real",
    "Days for shipment (scheduled)": "days_shipping_scheduled",
    "Benefit per order":             "benefit_per_order",
    "Sales per customer":            "sales_per_customer",
    "Delivery Status":               "delivery_status",
    "Late_delivery_risk":            "late_delivery_risk",
    "Category Id":                   "category_id",
    "Category Name":                 "category_name",
    "Customer City":                 "customer_city",
    "Customer Country":              "customer_country",
    "Customer Email":                "customer_email",
    "Customer Fname":                "customer_fname",
    "Customer Id":                   "customer_id",
    "Customer Lname":                "customer_lname",
    "Customer Password":             "customer_password",
    "Customer Segment":              "customer_segment",
    "Customer State":                "customer_state",
    "Customer Street":               "customer_street",
    "Customer Zipcode":              "customer_zipcode",
    "Department Id":                 "department_id",
    "Department Name":               "department_name",
    "Latitude":                      "latitude",
    "Longitude":                     "longitude",
    "Market":                        "market",
    "Order City":                    "order_city",
    "Order Country":                 "order_country",
    "Order Customer Id":             "order_customer_id",
    "order date (DateOrders)":       "order_date",
    "Order Id":                      "order_id",
    "Order Item Cardprod Id":        "order_item_cardprod_id",
    "Order Item Discount":           "order_item_discount",
    "Order Item Discount Rate":      "order_item_discount_rate",
    "Order Item Id":                 "order_item_id",
    "Order Item Product Price":      "order_item_product_price",
    "Order Item Profit Ratio":       "order_item_profit_ratio",
    "Order Item Quantity":           "order_item_quantity",
    "Sales":                         "sales",
    "Order Item Total":              "order_item_total",
    "Order Profit Per Order":        "order_profit_per_order",
    "Order Region":                  "order_region",
    "Order State":                   "order_state",
    "Order Status":                  "order_status",
    "Order Zipcode":                 "order_zipcode",
    "Product Card Id":               "product_card_id",
    "Product Category Id":           "product_category_id",
    "Product Description":           "product_description",
    "Product Image":                 "product_image",
    "Product Name":                  "product_name",
    "Product Price":                 "product_price",
    "Product Status":                "product_status",
    "shipping date (DateOrders)":    "shipping_date",
    "Shipping Mode":                 "shipping_mode",
}

NUMERIC_COLS = {
    "days_shipping_real", "days_shipping_scheduled", "benefit_per_order",
    "sales_per_customer", "late_delivery_risk", "category_id", "customer_id",
    "department_id", "latitude", "longitude", "order_customer_id", "order_id",
    "order_item_cardprod_id", "order_item_discount", "order_item_discount_rate",
    "order_item_id", "order_item_product_price", "order_item_profit_ratio",
    "order_item_quantity", "sales", "order_item_total", "order_profit_per_order",
    "product_card_id", "product_category_id", "product_price", "product_status",
}

DATE_COLS = {"order_date", "shipping_date"}


def parse_date(val: str):
    if not val:
        return None
    try:
        return datetime.strptime(val.strip(), "%m/%d/%Y %H:%M").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return val


def col_type(name: str) -> str:
    if name in NUMERIC_COLS:
        return "REAL"
    if name in DATE_COLS:
        return "DATETIME"
    return "TEXT"


def main():
    conn = sqlite3.connect(DB_PATH)

    clean_cols = list(COLUMN_MAP.values())
    # Computed columns appended at the end
    extra_cols = ["shipping_delay_days INTEGER"]

    col_defs = [f'"{c}" {col_type(c)}' for c in clean_cols] + extra_cols
    conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    conn.execute(f"CREATE TABLE {TABLE} ({', '.join(col_defs)})")

    orig_headers = list(COLUMN_MAP.keys())
    placeholders = ", ".join("?" * (len(clean_cols) + 1))  # +1 for shipping_delay_days

    rows_inserted = 0
    batch = []

    with open(CSV_PATH, encoding="latin1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = []
            for orig in orig_headers:
                clean = COLUMN_MAP[orig]
                val = row.get(orig, "").strip()
                if clean in NUMERIC_COLS:
                    try:
                        values.append(float(val) if val else None)
                    except ValueError:
                        values.append(None)
                elif clean in DATE_COLS:
                    values.append(parse_date(val))
                else:
                    values.append(val or None)

            # Computed: shipping delay (positive = delivered late)
            try:
                delay = int(row.get("Days for shipping (real)", 0) or 0) - \
                        int(row.get("Days for shipment (scheduled)", 0) or 0)
            except (ValueError, TypeError):
                delay = None
            values.append(delay)

            batch.append(values)
            rows_inserted += 1

            if len(batch) >= 5000:
                conn.executemany(f"INSERT INTO {TABLE} VALUES ({placeholders})", batch)
                batch.clear()

    if batch:
        conn.executemany(f"INSERT INTO {TABLE} VALUES ({placeholders})", batch)

    conn.commit()
    conn.close()
    print(f"Loaded {rows_inserted:,} rows into {DB_PATH}")


if __name__ == "__main__":
    main()
