"""Economic Order Quantity."""
"""Economic Order Quantity.

EOQ = sqrt(2 * D * S / H)
  D = annual demand (units/year)
  S = cost per order
  H = holding cost per unit per year
Assumes constant demand and no quantity discounts. Real life is messier,
so we also round up to the supplier MOQ later in policy_assign.py.
"""
import numpy as np


def eoq(annual_demand, order_cost, holding_cost_per_unit_yr):
    d = np.asarray(annual_demand, dtype=float)
    s = np.asarray(order_cost, dtype=float)
    h = np.asarray(holding_cost_per_unit_yr, dtype=float)
    h = np.where(h <= 0, np.nan, h)
    return np.sqrt(2 * d * s / h)