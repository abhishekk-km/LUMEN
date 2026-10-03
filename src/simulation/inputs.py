"""Shared loader for simulation scripts."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


def load_yaml(name):
    with open(ROOT / "config" / name) as f:
        return yaml.safe_load(f)


def load_inputs():
    """Return params, SKU policy table, test-window demand (T x n_sku array)."""
    params = load_yaml("params.yaml")

    pol = pd.read_csv(PROCESSED / "sku_policy.csv", index_col=0, dtype={"StockCode": str})
    pol.index = pol.index.astype(str)

    weekly = pd.read_csv(PROCESSED / "sku_weekly_demand.csv",
                         index_col=0, dtype={"StockCode": str})
    weekly.index = weekly.index.astype(str)

    test = weekly.iloc[:, params["history_weeks"]:]          # out-of-sample window
    test = test.reindex(pol.index).fillna(0).clip(lower=0)
    demand = test.to_numpy(dtype=float).T                     # shape (T, n_sku)
    return params, pol, demand