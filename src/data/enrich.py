"""Attach synthetic supplier, lead time, unit cost, MOQ, holding cost."""
"""Attach synthetic supply-side fields to each SKU.

Demand is real. Supplier region, lead time, unit cost, order cost and
holding cost are MODELLED using the rules in config/suppliers.yaml.

Run:
    python src/data/enrich.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"


def load_yaml(name: str) -> dict:
    with open(ROOT / "config" / name) as f:
        return yaml.safe_load(f)


def enrich(master: pd.DataFrame, params: dict, suppliers: dict) -> pd.DataFrame:
    rng = np.random.default_rng(params["random_seed"])
    m = master.copy()

    # 1. Price tier (terciles): low / medium / high
    m["price_tier"] = pd.qcut(
        m["unit_price"], 3, labels=["low", "medium", "high"]
    ).astype(str)

    # 2. Assign a supplier region based on price tier
    regions = suppliers["regions"]
    assign = suppliers["assignment_by_price_tier"]
    region_col = []
    for tier in m["price_tier"]:
        names = list(assign[tier].keys())
        probs = np.array(list(assign[tier].values()), dtype=float)
        probs = probs / probs.sum()
        region_col.append(rng.choice(names, p=probs))
    m["supplier_region"] = region_col

    # 3. Unit cost = price * U(low, high) * (1 + region premium)
    low, high = params["unit_cost_ratio_range"]
    ratio = rng.uniform(low, high, size=len(m))
    premium = m["supplier_region"].map(
        {r: v["unit_cost_premium"] for r, v in regions.items()}
    )
    m["unit_cost"] = (m["unit_price"] * ratio * (1 + premium)).round(4)

    # 4. Lead time (days -> weeks) and order cost, from the region table
    def col(field):
        return m["supplier_region"].map({r: v[field] for r, v in regions.items()})

    m["lead_time_mean_days"] = col("lead_time_mean_days")
    m["lead_time_std_days"] = col("lead_time_std_days")
    m["lead_time_mean_wk"] = (m["lead_time_mean_days"] / 7).round(4)
    m["lead_time_std_wk"] = (m["lead_time_std_days"] / 7).round(4)
    m["on_time_pct"] = col("on_time_pct")
    m["moq_multiplier"] = col("moq_multiplier")
    m["order_cost"] = col("order_cost")
    m["delay_prob"] = col("delay_prob")
    m["delay_extra_days"] = col("delay_extra_days")

    # 5. Holding cost per unit per year
    m["holding_cost_per_unit_yr"] = (
        m["unit_cost"] * params["holding_cost_pct"]
    ).round(4)

    return m


def main() -> None:
    params = load_yaml("params.yaml")
    suppliers = load_yaml("suppliers.yaml")

    master = pd.read_csv(PROCESSED / "sku_master_base.csv", index_col=0)
    master.index.name = "StockCode"

    out = enrich(master, params, suppliers)
    out.to_csv(PROCESSED / "sku_master.csv")

    print(f"Enriched SKUs: {len(out):,}")
    print("\nSKUs per supplier region:")
    print(out["supplier_region"].value_counts().to_string())
    print("\nAverage unit cost by region:")
    print(out.groupby("supplier_region")["unit_cost"].mean().round(2).to_string())
    print("\nSaved to data/processed/sku_master.csv")


if __name__ == "__main__":
    main()