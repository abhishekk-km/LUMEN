"""Demand spike, supplier delay, lead-time variability injectors."""
"""Scenario definitions: demand spike, supplier delay, lead-time variability."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Scenario:
    name: str
    demand_mult: np.ndarray                 # length T, multiplies actual demand
    extra_days_by_week: np.ndarray          # length T, days added to lead time of
    affected_region: Optional[str] = None   # orders PLACED that week, in this region
    std_mult: float = 1.0                   # multiplies lead-time std for all SKUs


def baseline(T):
    return Scenario("baseline", np.ones(T), np.zeros(T))


def demand_spike(T, start, dur, mult):
    m = np.ones(T)
    m[start:start + dur] = mult
    return Scenario(f"spike_x{mult}", m, np.zeros(T))


def supplier_delay(T, region, extra_days, start, dur):
    e = np.zeros(T)
    e[start:start + dur] = extra_days
    return Scenario(f"delay_{region}_+{extra_days}d", np.ones(T), e, region)


def lt_variability(T, std_mult):
    return Scenario(f"lt_std_x{std_mult}", np.ones(T), np.zeros(T), None, std_mult)


def combined(T, spike_mult, spike_start, spike_dur, region, extra_days,
             delay_start, delay_dur):
    m = np.ones(T)
    m[spike_start:spike_start + spike_dur] = spike_mult
    e = np.zeros(T)
    e[delay_start:delay_start + delay_dur] = extra_days
    return Scenario(f"spike_x{spike_mult}+delay_{extra_days}d", m, e, region)


def build_all(T, cfg):
    s = [baseline(T)]
    sp = cfg["demand_spike"]
    for mult in sp["multiplier"]:
        s.append(demand_spike(T, sp["start_week"], sp["duration_weeks"], mult))
    sd = cfg["supplier_delay"]
    for d in sd["extra_days"]:
        s.append(supplier_delay(T, sd["region"], d, sd["start_week"], sd["duration_weeks"]))
    for x in cfg["lead_time_variability"]["std_multiplier"]:
        s.append(lt_variability(T, x))
    c = cfg["combined"]
    s.append(combined(T, c["spike_multiplier"], sp["start_week"], sp["duration_weeks"],
                      sd["region"], c["delay_extra_days"], sd["start_week"],
                      sd["duration_weeks"]))
    return s