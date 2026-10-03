"""Reorder point."""
"""Reorder point = demand during lead time + safety stock.

ROP = mu_d * L + SS
When inventory position falls to ROP, place an order.
"""
import numpy as np


def reorder_point(mean_demand_wk, lead_time_wk, safety_stock_units):
    return np.asarray(mean_demand_wk) * np.asarray(lead_time_wk) + safety_stock_units