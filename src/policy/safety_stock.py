"""Safety stock with demand and lead-time variability."""
"""Safety stock with BOTH demand and lead-time variability.

SS = z * sqrt( L * sigma_d^2 + mu_d^2 * sigma_L^2 )
  z       = normal quantile of the target service level (cycle service level)
  mu_d    = mean demand per week
  sigma_d = std of weekly demand
  L       = mean lead time in weeks
  sigma_L = std of lead time in weeks
Assumes demand is normal and independent week to week. For lumpy and
intermittent items this is only an approximation (state this in the README).
"""
import numpy as np
from scipy.stats import norm


def safety_stock(service_level, mean_demand_wk, std_demand_wk,
                 lead_time_wk, lead_time_std_wk):
    z = norm.ppf(np.asarray(service_level, dtype=float))
    var = (lead_time_wk * np.square(std_demand_wk)
           + np.square(mean_demand_wk) * np.square(lead_time_std_wk))
    return z * np.sqrt(var)