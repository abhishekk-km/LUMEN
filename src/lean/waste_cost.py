"""Cost of excess stock, obsolescence, expediting."""
"""Put a number on the inventory-related Muda for one simulated policy.

Thresholds (change them here, state them in the README):
  excess stock   = average on-hand above 8 weeks of demand cover
  slow stock     = average on-hand above 26 weeks of demand cover
  write-down     = 10% per year on the slow-stock value
"""
import numpy as np
import pandas as pd

from src.lean.waste_mapping import MUDA

EXCESS_COVER_WK = 8
SLOW_COVER_WK = 26
WRITEDOWN_RATE = 0.10


def waste_costs(res, df, holding_pct):
    ann = 52.0 / res["T"]
    uc = df["unit_cost"].to_numpy(float)
    price = df["unit_price"].to_numpy(float)
    oc = df["order_cost"].to_numpy(float)
    mean_wk = df["mean_wk"].to_numpy(float)

    oh = res["avg_onhand"].mean(axis=0)                       # avg over runs
    excess = np.maximum(oh - mean_wk * EXCESS_COVER_WK, 0) * uc
    slow = np.maximum(oh - mean_wk * SLOW_COVER_WK, 0) * uc

    cost = {
        "Inventory": excess.sum() * holding_pct,
        "Overproduction": slow.sum() * WRITEDOWN_RATE,
        "Waiting": (res["lost"].mean(axis=0) * (price - uc)).sum() * ann,
        "Transportation": (res["n_orders"].mean(axis=0) * oc).sum() * ann,
    }
    out = MUDA.copy()
    out["annual_cost"] = out["muda"].map(cost)
    out["excess_stock_value"] = np.where(out["muda"] == "Inventory", excess.sum(), np.nan)
    return out