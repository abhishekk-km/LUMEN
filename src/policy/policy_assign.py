"""Map ABC-XYZ class to policy parameters."""
"""Apply EOQ, safety stock, ROP and MOQ to every SKU, differentiated by ABC-XYZ.

Run:
    python src/policy/policy_assign.py

Outputs:
    data/processed/sku_policy.csv
    outputs/policy_table.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.policy.eoq import eoq
from src.policy.safety_stock import safety_stock
from src.policy.reorder_point import reorder_point

PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


def assign_policies(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    d = df.copy()
    ppy = params["periods_per_year"]
    matrix = params["policy_matrix"]

    # 1. Target service level = base by ABC class + adjustment by ABC-XYZ cell
    base = d["abc"].map(params["service_level"])
    adj = d["abc_xyz"].map(lambda c: matrix[c]["service_adj"])
    d["service_level"] = (base + adj).clip(0.50, 0.999)
    d["policy"] = d["abc_xyz"].map(lambda c: matrix[c]["policy"])

    # 2. EOQ, then respect the supplier MOQ
    d["annual_demand"] = d["mean_wk"] * ppy
    d["eoq"] = eoq(d["annual_demand"], d["order_cost"], d["holding_cost_per_unit_yr"])
    # MOQ = moq_multiplier x 4 weeks of average demand (documented assumption)
    d["moq"] = np.ceil(d["moq_multiplier"] * 4 * d["mean_wk"])
    d["order_qty"] = np.ceil(np.maximum(d["eoq"], d["moq"])).clip(lower=1)

    # 3. Safety stock over the lead time, then reorder point
    std = d["std_wk"].fillna(0)
    d["safety_stock"] = np.ceil(safety_stock(
        d["service_level"], d["mean_wk"], std,
        d["lead_time_mean_wk"], d["lead_time_std_wk"]))
    d["rop"] = np.ceil(reorder_point(
        d["mean_wk"], d["lead_time_mean_wk"], d["safety_stock"]))

    # 4. Max level (continuous and min-max policies): order Q when position <= ROP
    d["max_level"] = d["rop"] + d["order_qty"]

    # 5. Periodic review (R,S): review every R weeks, order up to S
    #    R is roughly one EOQ cycle, kept between 1 and 8 weeks.
    d["review_wk"] = np.where(
        d["policy"].str.startswith("periodic"),
        np.clip(np.round(d["order_qty"] / d["mean_wk"]), 1, 8), np.nan)
    horizon = d["review_wk"] + d["lead_time_mean_wk"]
    z = norm.ppf(d["service_level"])
    ss_periodic = z * np.sqrt(
        horizon * std ** 2 + d["mean_wk"] ** 2 * d["lead_time_std_wk"] ** 2)
    order_up_to = np.ceil(d["mean_wk"] * horizon + ss_periodic)
    is_periodic = d["policy"].str.startswith("periodic")
    d.loc[is_periodic, "max_level"] = order_up_to[is_periodic]
    d.loc[is_periodic, "safety_stock"] = np.ceil(ss_periodic[is_periodic])

    # 6. Expected annual cost under the policy
    d["avg_inventory"] = d["order_qty"] / 2 + d["safety_stock"]
    d["annual_holding_cost"] = d["avg_inventory"] * d["holding_cost_per_unit_yr"]
    d["annual_order_cost"] = (d["annual_demand"] / d["order_qty"]) * d["order_cost"]
    d["annual_policy_cost"] = d["annual_holding_cost"] + d["annual_order_cost"]
    d["avg_inventory_value"] = d["avg_inventory"] * d["unit_cost"]
    return d


def main() -> None:
    with open(ROOT / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)

    df = pd.read_csv(PROCESSED / "sku_classified.csv", index_col=0, dtype={"StockCode": str})
    df.index = df.index.astype(str)
    df.index.name = "StockCode"

    out = assign_policies(df, params)
    out.to_csv(PROCESSED / "sku_policy.csv")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cols = ["description", "abc_xyz", "pattern", "supplier_region", "policy",
            "service_level", "mean_wk", "std_wk", "lead_time_mean_wk",
            "eoq", "moq", "order_qty", "safety_stock", "rop", "max_level",
            "review_wk", "avg_inventory_value", "annual_policy_cost"]
    xl = out[cols].round(2).sort_values("annual_policy_cost", ascending=False)
    xl.to_excel(OUTPUTS / "policy_table.xlsx")

    print(f"Policies assigned: {len(out):,}")
    print("\nPolicy mix:")
    print(out["policy"].value_counts().to_string())
    print("\nBy ABC class:")
    summary = out.groupby("abc").agg(
        skus=("abc", "size"),
        avg_service=("service_level", "mean"),
        inventory_value=("avg_inventory_value", "sum"),
        annual_cost=("annual_policy_cost", "sum"))
    print(summary.round(2).to_string())
    print(f"\nTotal avg inventory value: {out['avg_inventory_value'].sum():,.0f}")
    print(f"Total annual policy cost:  {out['annual_policy_cost'].sum():,.0f}")
    print("\nSaved: data/processed/sku_policy.csv and outputs/policy_table.xlsx")


if __name__ == "__main__":
    main()