"""Fill rate, turnover, DOI, stockout rate, carrying cost."""
"""KPI layer. All money values are annualised so results are comparable.

fill rate     = value of demand served from stock / value of demand (price weighted)
stockout rate = SKU-weeks with a stockout / SKU-weeks with demand
turnover      = annual COGS / average inventory value
DOI           = 365 / turnover
carrying cost = average inventory value x holding cost %
"""
import numpy as np
import pandas as pd


def kpis(res, df, holding_pct, mask=None):
    m = np.ones(len(df), bool) if mask is None else np.asarray(mask)
    T = res["T"]
    ann = 52.0 / T

    uc = df["unit_cost"].to_numpy(float)[m]
    price = df["unit_price"].to_numpy(float)[m]
    oc = df["order_cost"].to_numpy(float)[m]

    dem = res["demand"][m]
    sales = res["sales"][:, m]
    lost = res["lost"][:, m]
    oh = res["avg_onhand"][:, m]

    fill = (sales * price).sum(1) / (dem * price).sum()
    stockout = res["stockout_weeks"][:, m].sum(1) / res["demand_weeks"][m].sum()
    inv_value = (oh * uc).sum(1)
    cogs = (sales * uc).sum(1) * ann
    turnover = cogs / inv_value
    doi = 365.0 / turnover
    carrying = inv_value * holding_pct
    ordering = (res["n_orders"][:, m] * oc).sum(1) * ann
    lost_margin = (lost * (price - uc)).sum(1) * ann

    out = {
        "fill_rate": fill,
        "stockout_rate": stockout,
        "avg_inventory_value": inv_value,
        "turnover": turnover,
        "days_of_inventory": doi,
        "carrying_cost": carrying,
        "ordering_cost": ordering,
        "lost_margin": lost_margin,
        "total_cost": carrying + ordering + lost_margin,
        "orders_per_year": res["n_orders"][:, m].sum(1) * ann,
    }
    return {k: float(np.mean(v)) for k, v in out.items()}


def kpis_by_class(res, df, holding_pct):
    rows = {"ALL": kpis(res, df, holding_pct)}
    for c in ["A", "B", "C"]:
        rows[c] = kpis(res, df, holding_pct, (df["abc"] == c).to_numpy())
    return pd.DataFrame(rows).T