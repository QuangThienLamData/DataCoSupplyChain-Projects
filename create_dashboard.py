"""
Creates a DataCo Supply Chain dashboard in a running Superset instance.

Prerequisites:
  1. Run load_to_sqlite.py  →  data/dataco.db created
  2. Start Superset (Docker OR local venv), then run this script.

Usage:
    .scmvenv/Scripts/python.exe -m pip install requests
    .scmvenv/Scripts/python.exe create_dashboard.py

SQLITE_URI below must match how Superset can reach dataco.db:
  - Docker:  "sqlite:////app/data/dataco.db"   (mounted at /app/data)
  - Local:   "sqlite:////absolute/path/to/data/dataco.db"  (host path)
"""
import json
import os
import sys
import time
import uuid

import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL   = "http://localhost:8088"
USERNAME   = "admin"
PASSWORD   = "admin"

# Auto-detect: use Docker path if running in container, otherwise absolute host path
_here = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
_local_db = os.path.join(_here, "data", "dataco.db").replace("\\", "/")
SQLITE_URI = os.environ.get(
    "DATACO_SQLITE_URI",
    f"sqlite:///{_local_db}"   # local default; override with env var for Docker
)
DB_NAME    = "DataCo Supply Chain"
DATASET    = "supply_chain"
DASH_TITLE = "DataCo Supply Chain — Overview & Operations"


# ── Metric helpers ────────────────────────────────────────────────────────────
def count_metric():
    return {
        "expressionType": "SIMPLE",
        "column": None,
        "aggregate": "COUNT",
        "label": "COUNT(*)",
        "optionName": "metric_count",
    }


def sum_metric(col, label=None):
    return {
        "expressionType": "SIMPLE",
        "column": {"column_name": col, "type": "DOUBLE PRECISION"},
        "aggregate": "SUM",
        "label": label or f"SUM({col})",
        "optionName": f"metric_SUM__{col}",
    }


def sql_metric(expression, label):
    return {
        "expressionType": "SQL",
        "sqlExpression": expression,
        "label": label,
        "optionName": f"metric_sql_{label.replace(' ', '_')}",
    }


# ── Chart builders ────────────────────────────────────────────────────────────
def big_number(ds_id, name, metric, subheader="", y_fmt="SMART_NUMBER"):
    return {
        "slice_name": name,
        "viz_type": "big_number_total",
        "params": json.dumps({
            "viz_type": "big_number_total",
            "datasource": f"{ds_id}__table",
            "metric": metric,
            "subheader": subheader,
            "y_axis_format": y_fmt,
            "time_range": "No filter",
        }),
    }


def pie_chart(ds_id, name, groupby, metric=None):
    return {
        "slice_name": name,
        "viz_type": "pie",
        "params": json.dumps({
            "viz_type": "pie",
            "datasource": f"{ds_id}__table",
            "groupby": groupby,
            "metric": metric or count_metric(),
            "row_limit": 100,
            "sort_by_metric": True,
            "donut": False,
            "show_legend": True,
            "time_range": "No filter",
        }),
    }


def bar_chart(ds_id, name, groupby, metrics=None, columns=None,
              stacked=False, row_limit=50, y_fmt=None):
    params = {
        "viz_type": "dist_bar",
        "datasource": f"{ds_id}__table",
        "groupby": groupby,
        "columns": columns or [],
        "metrics": metrics or [count_metric()],
        "row_limit": row_limit,
        "bar_stacked": stacked,
        "show_legend": True,
        "time_range": "No filter",
    }
    if y_fmt:
        params["y_axis_format"] = y_fmt
    return {
        "slice_name": name,
        "viz_type": "dist_bar",
        "params": json.dumps(params),
    }


def line_chart(ds_id, name, time_col, metrics=None, groupby=None, grain="P1M"):
    return {
        "slice_name": name,
        "viz_type": "line",
        "params": json.dumps({
            "viz_type": "line",
            "datasource": f"{ds_id}__table",
            "granularity_sqla": time_col,
            "time_grain_sqla": grain,
            "time_range": "No filter",
            "metrics": metrics or [count_metric()],
            "groupby": groupby or [],
            "contribution": False,
            "show_legend": True,
            "x_axis_showminmax": False,
        }),
    }


# ── Dashboard layout builder ──────────────────────────────────────────────────
def build_position_json(rows_of_chart_ids, chart_meta):
    """
    rows_of_chart_ids: list of rows, each row is a list of chart IDs (ints)
    chart_meta: dict of chart_id -> slice_name
    Returns the position_json string Superset expects.
    """
    row_ids = []
    components = {}

    for row_idx, chart_ids in enumerate(rows_of_chart_ids):
        row_id = f"ROW-{row_idx}"
        row_ids.append(row_id)
        width = 12 // len(chart_ids)
        child_ids = []

        for chart_id in chart_ids:
            comp_id = f"CHART-{chart_id}"
            child_ids.append(comp_id)
            is_kpi = len(chart_ids) >= 4
            components[comp_id] = {
                "children": [],
                "id": comp_id,
                "meta": {
                    "chartId": chart_id,
                    "height": 11 if is_kpi else 50,
                    "sliceName": chart_meta.get(chart_id, ""),
                    "uuid": str(uuid.uuid4()),
                    "width": width,
                },
                "parents": ["ROOT_ID", "GRID_ID", row_id],
                "type": "CHART",
            }

        components[row_id] = {
            "children": child_ids,
            "id": row_id,
            "meta": {"background": "BACKGROUND_TRANSPARENT"},
            "parents": ["ROOT_ID", "GRID_ID"],
            "type": "ROW",
        }

    layout = {
        "DASHBOARD_VERSION_KEY": "v2",
        "GRID_ID": {
            "children": row_ids,
            "id": "GRID_ID",
            "type": "GRID",
        },
        "HEADER_ID": {
            "id": "HEADER_ID",
            "meta": {"text": DASH_TITLE},
            "type": "HEADER",
        },
        "ROOT_ID": {
            "children": ["GRID_ID"],
            "id": "ROOT_ID",
            "type": "ROOT",
        },
    }
    layout.update(components)
    return json.dumps(layout)


# ── Superset API client ───────────────────────────────────────────────────────
class SupersetClient:
    def __init__(self):
        self.session = requests.Session()
        self.token = None

    def wait_for_ready(self, timeout=120):
        print("Waiting for Superset to be ready...", end="", flush=True)
        for _ in range(timeout):
            try:
                r = self.session.get(f"{BASE_URL}/health", timeout=3)
                if r.status_code == 200:
                    print(" ready.")
                    return
            except requests.exceptions.ConnectionError:
                pass
            print(".", end="", flush=True)
            time.sleep(1)
        print()
        sys.exit("Superset did not become ready in time. Is Docker running?")

    def login(self):
        r = self.session.post(f"{BASE_URL}/api/v1/security/login", json={
            "username": USERNAME,
            "password": PASSWORD,
            "provider": "db",
            "refresh": True,
        })
        r.raise_for_status()
        self.token = r.json()["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        print(f"Logged in as {USERNAME}")

    def _headers(self):
        """Headers for mutating requests (POST/PUT/DELETE)."""
        return {"Content-Type": "application/json"}

    def get(self, path, **kwargs):
        return self.session.get(f"{BASE_URL}{path}", **kwargs)

    def post(self, path, **kwargs):
        return self.session.post(f"{BASE_URL}{path}", **kwargs)

    def put(self, path, **kwargs):
        return self.session.put(f"{BASE_URL}{path}", **kwargs)

    # ── Resource creation ─────────────────────────────────────────────────────
    def find_or_create_database(self):
        # Check if already exists
        r = self.get("/api/v1/database/", params={"q": json.dumps({"filters": [
            {"col": "database_name", "opr": "eq", "val": DB_NAME}
        ]})})
        existing = r.json().get("result", [])
        if existing:
            db_id = existing[0]["id"]
            print(f"  Database already exists (id={db_id})")
            return db_id

        r = self.post("/api/v1/database/", headers=self._headers(), json={
            "database_name": DB_NAME,
            "sqlalchemy_uri": SQLITE_URI,
            "expose_in_sqllab": True,
        })
        r.raise_for_status()
        db_id = r.json()["id"]
        print(f"  Created database (id={db_id})")
        return db_id

    def find_or_create_dataset(self, db_id):
        r = self.get("/api/v1/dataset/", params={"q": json.dumps({"filters": [
            {"col": "table_name", "opr": "eq", "val": DATASET}
        ]})})
        existing = r.json().get("result", [])
        if existing:
            ds_id = existing[0]["id"]
            print(f"  Dataset already exists (id={ds_id})")
            return ds_id

        r = self.post("/api/v1/dataset/", headers=self._headers(), json={
            "database": db_id,
            "schema": "main",
            "table_name": DATASET,
        })
        r.raise_for_status()
        ds_id = r.json()["id"]
        print(f"  Created dataset (id={ds_id})")

        # Refresh so Superset detects column types from SQLite schema
        self.post(f"/api/v1/dataset/{ds_id}/refresh", headers=self._headers())
        return ds_id

    def create_chart(self, spec, ds_id):
        payload = {
            "slice_name": spec["slice_name"],
            "viz_type": spec["viz_type"],
            "datasource_id": ds_id,
            "datasource_type": "table",
            "params": spec["params"],
        }
        r = self.post("/api/v1/chart/", headers=self._headers(), json=payload)
        r.raise_for_status()
        chart_id = r.json()["id"]
        print(f"    [{chart_id}] {spec['slice_name']}")
        return chart_id

    def create_dashboard(self, title, chart_ids, position_json):
        r = self.post("/api/v1/dashboard/", headers=self._headers(), json={
            "dashboard_title": title,
            "published": True,
            "position_json": position_json,
            "slices": chart_ids,
        })
        r.raise_for_status()
        result = r.json()
        dash_id = result.get("id") or result.get("result", {}).get("id")
        return dash_id


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    client = SupersetClient()
    client.wait_for_ready()
    client.login()

    print("\n[1/4] Setting up database and dataset...")
    db_id = client.find_or_create_database()
    ds_id = client.find_or_create_dataset(db_id)

    print("\n[2/4] Creating charts...")

    # ── Section 1: KPI scorecards ─────────────────────────────────────────────
    kpi_specs = [
        big_number(ds_id, "Total Orders",
                   count_metric(), "orders"),
        big_number(ds_id, "Total Revenue",
                   sum_metric("sales", "Total Revenue"), "USD", "$,.0f"),
        big_number(ds_id, "Total Profit",
                   sum_metric("order_profit_per_order", "Total Profit"), "USD", "$,.0f"),
        big_number(ds_id, "Late Deliveries",
                   sql_metric("SUM(CASE WHEN delivery_status = 'Late delivery' THEN 1 ELSE 0 END)",
                              "Late Deliveries"),
                   "orders (54.8% of total)"),
        big_number(ds_id, "Suspected Fraud",
                   sql_metric("SUM(CASE WHEN order_status = 'SUSPECTED_FRAUD' THEN 1 ELSE 0 END)",
                              "Suspected Fraud"),
                   "flagged orders"),
        big_number(ds_id, "Negative-Profit Orders",
                   sql_metric("SUM(CASE WHEN order_profit_per_order < 0 THEN 1 ELSE 0 END)",
                              "Negative Profit Orders"),
                   "orders (18.7% of total)"),
    ]

    # ── Section 2: Delivery operations ───────────────────────────────────────
    delivery_specs = [
        pie_chart(ds_id, "Delivery Status Breakdown",
                  ["delivery_status"]),
        bar_chart(ds_id, "Delivery Status by Shipping Mode",
                  groupby=["shipping_mode"],
                  columns=["delivery_status"],
                  stacked=True),
        bar_chart(ds_id, "Order Status Distribution",
                  groupby=["order_status"]),
    ]

    # ── Section 3: Revenue analysis ───────────────────────────────────────────
    revenue_specs = [
        line_chart(ds_id, "Monthly Revenue Trend",
                   time_col="order_date",
                   metrics=[sum_metric("sales", "Revenue")],
                   grain="P1M"),
        bar_chart(ds_id, "Revenue by Category — Top 10",
                  groupby=["category_name"],
                  metrics=[sum_metric("sales", "Revenue")],
                  row_limit=10,
                  y_fmt="$,.0f"),
        pie_chart(ds_id, "Revenue by Market",
                  ["market"],
                  metric=sum_metric("sales", "Revenue")),
    ]

    # ── Section 4: Operational problems ──────────────────────────────────────
    problems_specs = [
        bar_chart(ds_id, "Shipping Delay Distribution (actual - scheduled days)",
                  groupby=["shipping_delay_days"],
                  row_limit=20),
        bar_chart(ds_id, "Delivery Status by Customer Segment",
                  groupby=["customer_segment"],
                  columns=["delivery_status"],
                  stacked=True),
        bar_chart(ds_id, "Problem Order Status by Market",
                  groupby=["market"],
                  columns=["order_status"],
                  stacked=True),
    ]

    all_sections = [kpi_specs, delivery_specs, revenue_specs, problems_specs]
    section_names = ["KPI scorecards", "Delivery operations",
                     "Revenue analysis", "Operational problems"]

    rows_of_ids = []
    all_ids = []
    chart_meta = {}

    for specs, name in zip(all_sections, section_names):
        print(f"  {name}:")
        row_ids = []
        for spec in specs:
            cid = client.create_chart(spec, ds_id)
            row_ids.append(cid)
            all_ids.append(cid)
            chart_meta[cid] = spec["slice_name"]
        rows_of_ids.append(row_ids)

    print("\n[3/4] Building dashboard layout...")
    position_json = build_position_json(rows_of_ids, chart_meta)

    print("\n[4/4] Creating dashboard...")
    dash_id = client.create_dashboard(DASH_TITLE, all_ids, position_json)
    if dash_id:
        print(f"\nDashboard created (id={dash_id})")
        print(f"Open: {BASE_URL}/superset/dashboard/{dash_id}/")
    else:
        print("\nDashboard created. Visit http://localhost:8088/dashboard/list/")


if __name__ == "__main__":
    main()
