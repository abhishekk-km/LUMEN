"""Fixed-cycle, forecast-driven ordering."""
"""PUSH (non-lean): forecast-driven, fixed-cycle batch ordering.

Every `cycle_wk` weeks the planner orders up to:
    mean_demand * (cycle + lead_time + buffer_wk)
The buffer is a flat number of weeks for EVERY SKU. It ignores demand
variability, lead-time variability and ABC class. Orders follow the
calendar, not actual consumption.

buffer_wk is the knob that decides how much safety the planner holds.
"""
import numpy as np


class PushPolicy:
    name = "push"

    def __init__(self, df, cycle_wk=4, buffer_wk=3):
        self.cycle = cycle_wk
        self.moq = df["moq"].to_numpy(float)
        mean = df["mean_wk"].to_numpy(float)
        lt = df["lead_time_mean_wk"].to_numpy(float)
        self.target = mean * (cycle_wk + lt + buffer_wk)

    def order_qty(self, t, on_hand, on_order):
        if t % self.cycle != 0:
            return np.zeros_like(on_hand)
        ip = on_hand + on_order
        need = np.maximum(self.target - ip, 0)
        qty = np.where(need > 0, np.maximum(need, self.moq), 0)
        return np.ceil(qty)