"""Kanban / min-max, consumption-triggered ordering."""
"""PULL (lean): consumption-triggered replenishment, Kanban / min-max style.

When inventory position (on hand + on order) falls to the reorder point,
that is the 'Kanban card'. Order enough to return to the max level.
Each SKU uses its OWN ROP and max level, sized by its class, variability
and lead time. Nothing is ordered unless stock was actually consumed.
"""
import numpy as np


class PullPolicy:
    name = "pull"

    def __init__(self, df):
        self.rop = df["rop"].to_numpy(float)
        self.max_level = df["max_level"].to_numpy(float)
        self.moq = df["moq"].to_numpy(float)

    def order_qty(self, t, on_hand, on_order):
        ip = on_hand + on_order
        qty = np.ceil(np.maximum(self.max_level - ip, self.moq))
        return np.where(ip <= self.rop, qty, 0.0)