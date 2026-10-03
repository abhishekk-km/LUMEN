"""Profile demand, classify ABC-XYZ, save data/processed/sku_classified.csv.

Run:
    python src/analysis/run_analysis.py
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.demand_profile import profile_demand
from src.analysis.classification import abc_classify, xyz_classify

PROCESSED = ROOT / "data" / "processed"
MIN_OBS_WEEKS = 13      # need at least a quarter of history to estimate mean/std


def main() -> None:
    with open(ROOT / "config" / "params.yaml") as f:
        params = yaml.safe_load(f)

    weekly = pd.read_csv(PROCESSED / "sku_weekly_demand.csv",
                         index_col=0, dtype={"StockCode": str})
    weekly.index = weekly.index.astype(str)
    weekly.columns = pd.to_datetime(weekly.columns)

    master = pd.read_csv(PROCESSED / "sku_master.csv",
                         index_col=0, dtype={"StockCode": str})
    master.index = master.index.astype(str)

    prof = profile_demand(weekly, params["history_weeks"])

    # Keep SKUs that have enough history in the design window
    keep = (prof["n_obs_weeks"] >= MIN_OBS_WEEKS) & (prof["mean_wk"] > 0)
    print(f"SKUs total: {len(prof):,} | usable for policy design: {int(keep.sum()):,}")
    print(f"Excluded (launched late or too little history): {int((~keep).sum()):,}")
    prof = prof[keep]

    df = prof.join(master, how="inner")

    # ABC on history revenue = history units x median price
    df["hist_revenue"] = df["hist_units"] * df["unit_price"]
    df["abc"] = abc_classify(df["hist_revenue"],
                             params["abc"]["a_cutoff"], params["abc"]["b_cutoff"])
    df["xyz"] = xyz_classify(df["cv"], params["xyz"]["x_max_cv"],
                             params["xyz"]["y_max_cv"])
    df["abc_xyz"] = df["abc"] + df["xyz"]

    df.to_csv(PROCESSED / "sku_classified.csv")

    print("\nABC counts:")
    print(df["abc"].value_counts().sort_index().to_string())
    print("\nABC x XYZ (number of SKUs):")
    print(pd.crosstab(df["abc"], df["xyz"]).to_string())
    print("\nDemand pattern:")
    print(df["pattern"].value_counts().to_string())
    print("\nSaved to data/processed/sku_classified.csv")


if __name__ == "__main__":
    main()