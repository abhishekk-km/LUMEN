"""ABC (value) and XYZ (variability) classification."""
"""ABC (value) and XYZ (variability) classification."""
import numpy as np
import pandas as pd


def abc_classify(value: pd.Series, a_cutoff: float, b_cutoff: float) -> pd.Series:
    """Pareto: A = top a_cutoff of value, B = up to b_cutoff, C = rest."""
    v = value.clip(lower=0).sort_values(ascending=False)
    share = v / v.sum()
    cum_before = share.cumsum() - share          # share consumed BEFORE this SKU
    cls = np.select(
        [cum_before < a_cutoff, cum_before < b_cutoff], ["A", "B"], default="C"
    )
    return pd.Series(cls, index=v.index).reindex(value.index)


def xyz_classify(cv: pd.Series, x_max: float, y_max: float) -> pd.Series:
    """X = stable, Y = moderate, Z = erratic (by coefficient of variation)."""
    return pd.Series(
        np.select([cv <= x_max, cv <= y_max], ["X", "Y"], default="Z"),
        index=cv.index,
    )