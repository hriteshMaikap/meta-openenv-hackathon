"""
server/fault_injector.py — Generates broken datasets for RL training.

Design principle: every fault has a DETERMINISTIC ground truth fix.
The grader can verify the fix without an LLM judge.

Fault taxonomy (from ELT-Bench analysis of real pipeline failures):
  LEVEL 1 (Easy)  — single-table, independent column faults
  LEVEL 2 (Medium)— cross-table referential and business rule faults
  LEVEL 3 (Hard)  — temporal schema drift mid-episode

All faults are tagged with their check name so the grader knows
exactly what to score. The agent does NOT see the fault tags —
it must discover them by profiling and validating.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────
# Fault descriptor
# ─────────────────────────────────────────────────────────────

@dataclass
class Fault:
    name: str               # e.g. "null_fk"
    check_name: str         # grader check it violates
    description: str        # human-readable description
    affected_columns: List[str] = field(default_factory=list)
    severity: str = "medium"  # "low" | "medium" | "high"


# ─────────────────────────────────────────────────────────────
# Easy dataset (single table: orders)
# ─────────────────────────────────────────────────────────────

EASY_FAULTS = [
    Fault("date_format_mix",    "type_check",       "ISO + natural language dates mixed", ["order_date"]),
    Fault("duplicate_pk",       "uniqueness_check",  "Duplicate order_id rows", ["order_id"]),
    Fault("null_fk",            "null_check",        "NULL customer_id (FK violation)", ["customer_id"]),
    Fault("negative_amount",    "range_check",       "amount < 0", ["amount"]),
    Fault("outlier_amount",     "range_check",       "amount > 10000 (outlier)", ["amount"]),
    Fault("status_case",        "type_check",        "COMPLETED vs completed case mismatch", ["status"]),
]

# Schema contract the agent must satisfy
EASY_TARGET_SCHEMA = {
    "order_id":    {"dtype": "int64",   "nullable": False, "unique": True},
    "customer_id": {"dtype": "object",  "nullable": False, "pattern": r"C_\d+"},
    "amount":      {"dtype": "float64", "nullable": False, "min": 0.0, "max": 10_000.0},
    "order_date":  {"dtype": "datetime64[ns]", "nullable": False},
    "status":      {"dtype": "object",  "nullable": False, "values": ["completed", "pending", "cancelled"]},
}


def _make_easy_base(n: int, rng: random.Random) -> pd.DataFrame:
    """Generate a clean base orders table."""
    statuses = ["completed", "pending", "cancelled"]
    return pd.DataFrame({
        "order_id":    list(range(1000, 1000 + n)),
        "customer_id": [f"C_{rng.randint(10, 999):03d}" for _ in range(n)],
        "amount":      [round(rng.uniform(5.0, 500.0), 2) for _ in range(n)],
        "order_date":  pd.date_range("2024-01-01", periods=n, freq="h"),
        "status":      [rng.choice(statuses) for _ in range(n)],
    })


def _inject_easy_faults(
    df: pd.DataFrame,
    faults: List[Fault],
    rng: random.Random,
) -> Tuple[pd.DataFrame, List[str]]:
    """Inject selected faults into the dataframe. Returns (df, fault_names)."""
    df = df.copy()
    injected = []

    for fault in faults:
        n = len(df)
        affected = max(1, int(n * rng.uniform(0.03, 0.12)))  # 3–12% of rows
        idx = rng.sample(range(n), k=min(affected, n))

        if fault.name == "date_format_mix":
            natural_formats = [
                lambda d: d.strftime("%b %d %Y"),
                lambda d: d.strftime("%B %d, %Y"),
                lambda d: d.strftime("%d-%m-%Y"),
            ]
            fmt = rng.choice(natural_formats)
            df.loc[idx, "order_date"] = df.loc[idx, "order_date"].apply(
                lambda d: fmt(d) if isinstance(d, pd.Timestamp) else fmt(pd.Timestamp(d))
            )

        elif fault.name == "duplicate_pk":
            # Duplicate a block of rows at the end
            dupes = df.iloc[idx[:max(1, affected // 2)]].copy()
            df = pd.concat([df, dupes], ignore_index=True)

        elif fault.name == "null_fk":
            df.loc[idx, "customer_id"] = None

        elif fault.name == "negative_amount":
            df.loc[idx, "amount"] = [-rng.uniform(1.0, 200.0) for _ in idx]

        elif fault.name == "outlier_amount":
            df.loc[idx, "amount"] = [rng.uniform(15_000, 500_000) for _ in idx]

        elif fault.name == "status_case":
            mixed = ["COMPLETED", "Completed", "PENDING", "Cancelled"]
            df.loc[idx, "status"] = [rng.choice(mixed) for _ in idx]

        injected.append(fault.name)

    return df, injected


# ─────────────────────────────────────────────────────────────
# Medium dataset (3 tables: orders + customers + products)
# ─────────────────────────────────────────────────────────────

MEDIUM_TARGET_SCHEMA = {
    "order_id":    {"dtype": "int64",  "nullable": False, "unique": True},
    "customer_name": {"dtype": "object", "nullable": False},
    "region":      {"dtype": "object",  "nullable": False, "values": [
                        "North East", "North West", "South East",
                        "South West", "Midwest", "West Coast"]},
    "tier":        {"dtype": "object",  "nullable": False, "values": ["gold", "silver", "bronze"]},
    "category":    {"dtype": "object",  "nullable": False},
    "qty":         {"dtype": "int64",   "nullable": False, "min": 1},
    "revenue":     {"dtype": "float64", "nullable": False, "min": 0.0},
    "margin_pct":  {"dtype": "float64", "nullable": False, "min": 0.0, "max": 1.0},
}

VALID_REGIONS = ["North East", "North West", "South East", "South West", "Midwest", "West Coast"]
REGION_TYPOS = {
    "North East": ["Nort East", "NorthEast", "north east"],
    "South West": ["SouthWest", "Soth West", "south west"],
    "Midwest":    ["Mid-West", "Mid West", "midwest"],
    "West Coast": ["WestCoast", "West coast", "west coast"],
}


def _make_medium_tables(n_orders: int, rng: random.Random):
    """Generate three clean tables."""
    n_customers = max(20, n_orders // 10)
    n_products = max(10, n_orders // 20)

    customers = pd.DataFrame({
        "customer_id": [f"C_{i:04d}" for i in range(n_customers)],
        "name":        [f"Customer_{i}" for i in range(n_customers)],
        "region":      [rng.choice(VALID_REGIONS) for _ in range(n_customers)],
        "tier":        [rng.choice(["gold", "silver", "bronze"]) for _ in range(n_customers)],
    })

    products = pd.DataFrame({
        "product_id":  [f"P_{i:04d}" for i in range(n_products)],
        "category":    [rng.choice(["Electronics", "Clothing", "Food", "Books"]) for _ in range(n_products)],
        "cost_price":  [round(rng.uniform(5.0, 200.0), 2) for _ in range(n_products)],
        "sale_price":  None,  # computed below
    })
    products["sale_price"] = products["cost_price"].apply(
        lambda c: round(c * rng.uniform(1.2, 3.0), 2)
    )

    valid_cids = customers["customer_id"].tolist()
    valid_pids = products["product_id"].tolist()
    orders = pd.DataFrame({
        "order_id":    list(range(5000, 5000 + n_orders)),
        "customer_id": [rng.choice(valid_cids) for _ in range(n_orders)],
        "product_id":  [rng.choice(valid_pids) for _ in range(n_orders)],
        "qty":         [rng.randint(1, 20) for _ in range(n_orders)],
        "price":       None,  # set below
    })
    # Use products.sale_price for clean price
    price_map = dict(zip(products["product_id"], products["sale_price"]))
    orders["price"] = orders["product_id"].map(price_map)

    return orders, customers, products


def _inject_medium_faults(orders, customers, products, rng):
    """Inject cross-table faults. Returns (orders, customers, products, fault_names)."""
    orders, customers, products = orders.copy(), customers.copy(), products.copy()
    injected = []

    # 1. Orphaned customer FKs
    n_orphan = max(2, int(len(orders) * 0.05))
    idx = rng.sample(range(len(orders)), k=n_orphan)
    orders.loc[idx, "customer_id"] = [f"C_GHOST_{i}" for i in range(n_orphan)]
    injected.append("orphaned_customer_fk")

    # 2. Orphaned product FKs
    n_orphan_p = max(2, int(len(orders) * 0.04))
    idx_p = rng.sample([i for i in range(len(orders)) if i not in idx], k=n_orphan_p)
    orders.loc[idx_p, "product_id"] = [f"P_GHOST_{i}" for i in range(n_orphan_p)]
    injected.append("orphaned_product_fk")

    # 3. Impossible margin (price < cost_price)
    n_margin = max(2, int(len(orders) * 0.06))
    idx_m = rng.sample(range(len(orders)), k=n_margin)
    cost_map = dict(zip(products["product_id"], products["cost_price"]))
    for i in idx_m:
        pid = orders.loc[i, "product_id"]
        cost = cost_map.get(pid, 10.0)
        orders.loc[i, "price"] = round(cost * rng.uniform(0.3, 0.9), 2)
    injected.append("negative_margin")

    # 4. Region typos in customers
    typo_candidates = [r for r in REGION_TYPOS]
    n_typos = max(3, int(len(customers) * 0.15))
    idx_r = rng.sample(range(len(customers)), k=min(n_typos, len(customers)))
    for i in idx_r:
        region = customers.loc[i, "region"]
        typos = REGION_TYPOS.get(region, [region.lower()])
        customers.loc[i, "region"] = rng.choice(typos)
    injected.append("region_typos")

    # 5. Category case inconsistency
    n_case = max(3, int(len(products) * 0.20))
    idx_c = rng.sample(range(len(products)), k=min(n_case, len(products)))
    products.loc[idx_c, "category"] = products.loc[idx_c, "category"].str.upper()
    injected.append("category_case")

    return orders, customers, products, injected


# ─────────────────────────────────────────────────────────────
# Hard task — adds schema drift on top of medium faults
# ─────────────────────────────────────────────────────────────

SCHEMA_DRIFT_SCENARIOS = [
    {
        "name": "rename_and_add",
        "changes": [
            {"type": "rename", "old_col": "order_date", "new_col": "created_at"},
            {"type": "add",    "new_col": "currency_code", "dtype": "str", "default": "USD"},
        ],
        "description": "'order_date' renamed to 'created_at'; new column 'currency_code' (NOT NULL, default 'USD') added",
    },
    {
        "name": "type_change_and_add",
        "changes": [
            {"type": "type_change", "col": "amount", "new_dtype": "Decimal(10,2)"},
            {"type": "add",         "new_col": "region_code", "dtype": "str", "default": "US"},
        ],
        "description": "'amount' type changed to DECIMAL(10,2); new 'region_code' column added",
    },
    {
        "name": "rename_two",
        "changes": [
            {"type": "rename", "old_col": "customer_id", "new_col": "client_id"},
            {"type": "rename", "old_col": "status",      "new_col": "order_status"},
        ],
        "description": "'customer_id' renamed to 'client_id'; 'status' renamed to 'order_status'",
    },
]


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

class FaultInjector:
    """
    Generates broken datasets for ETL agent training.
    Reproducible given the same seed.
    """

    def make_easy_episode(self, seed: int, n_rows: int = 700):
        rng = random.Random(seed)
        np.random.seed(seed)

        base = _make_easy_base(n_rows, rng)

        # Sample 3–5 faults per episode (not always the same set)
        n_faults = rng.randint(3, len(EASY_FAULTS))
        selected = rng.sample(EASY_FAULTS, k=n_faults)

        broken, injected = _inject_easy_faults(base, selected, rng)

        # Gold is the clean base
        gold = _make_easy_base(n_rows, rng)  # same seed, same data, no faults

        return {
            "df_broken": broken,
            "df_gold":   base,              # clean original
            "faults_planted": injected,
            "target_schema": EASY_TARGET_SCHEMA,
            "gold_row_count": n_rows,       # approximate (before dedup)
        }

    def make_medium_episode(self, seed: int, n_orders: int = 500):
        rng = random.Random(seed)
        np.random.seed(seed)

        orders, customers, products = _make_medium_tables(n_orders, rng)
        orders_b, customers_b, products_b, injected = _inject_medium_faults(
            orders, customers, products, rng
        )

        return {
            "orders": orders_b,
            "customers": customers_b,
            "products": products_b,
            "orders_clean": orders,
            "customers_clean": customers,
            "products_clean": products,
            "faults_planted": injected,
            "target_schema": MEDIUM_TARGET_SCHEMA,
        }

    def make_hard_episode(self, seed: int, n_orders: int = 400, drift_at_step: int = 8):
        rng = random.Random(seed)
        base_episode = self.make_medium_episode(seed, n_orders)

        # Pick schema drift scenario
        drift = rng.choice(SCHEMA_DRIFT_SCENARIOS)

        return {
            **base_episode,
            "schema_drift_step": drift_at_step,
            "schema_drift": drift,
            "faults_planted": base_episode["faults_planted"] + ["schema_drift"],
        }