"""Mean, std, CV^2, ADI (intermittency)."""
"""Demand profile per SKU, computed on the HISTORY window only.

Metrics: mean, std, CV, ADI (intermittency), and the Syntetos-Boylan
demand pattern (smooth / intermittent / erratic / lumpy).
"""
import numpy as np
import pandas as pd


def profile_demand(weekly: pd.DataFrame, history_weeks: int) -> pd.DataFrame:
    # NaN = SKU not yet stocked, so it is ignored. Zeros = real zero demand.
    hist = weekly.iloc[:, :history_weeks]

    n_obs = hist.notna().sum(axis=1)
    n_demand = (hist > 0).sum(axis=1)
    mean = hist.mean(axis=1)
    std = hist.std(axis=1)

    nz = hist.where(hist > 0)                  # only weeks with demand
    nz_mean = nz.mean(axis=1)
    nz_std = nz.std(axis=1).fillna(0)

    out = pd.DataFrame({
        "n_obs_weeks": n_obs,
        "n_demand_weeks": n_demand,
        "mean_wk": mean,
        "std_wk": std,
        "cv": std / mean.replace(0, np.nan),
        "adi": n_obs / n_demand.replace(0, np.nan),
        "cv2_nonzero": (nz_std / nz_mean) ** 2,
        "hist_units": hist.sum(axis=1),
    })

    # Syntetos-Boylan cut-offs: ADI 1.32, CV^2 0.49
    smooth_adi = out["adi"] < 1.32
    smooth_cv = out["cv2_nonzero"] < 0.49
    out["pattern"] = np.select(
        [smooth_adi & smooth_cv, ~smooth_adi & smooth_cv, smooth_adi & ~smooth_cv],
        ["smooth", "intermittent", "erratic"],
        default="lumpy",
    )
    return out