"""Core daily/weekly inventory simulation loop."""
"""Core weekly inventory simulator (vectorised over runs and SKUs).

Weekly order of events:
 1. Receive orders arriving this week
 2. Demand occurs; unmet demand is LOST (lost-sales model)
 3. Policy looks at inventory position and decides what to order
 4. Order gets a random lead time (lognormal + delay events + scenario effects)

Demand is the REAL test-window demand. Randomness comes from lead times.
Lead times are rounded up to whole weeks (minimum 1) because the model is weekly.
"""
import numpy as np

MAX_LT = 20   # cap on lead time, weeks


def simulate(df, demand, policy, scenario, n_runs=30, seed=42):
    rng = np.random.default_rng(seed)
    T, n = demand.shape
    R = n_runs

    uc = df["unit_cost"].to_numpy(float)
    price = df["unit_price"].to_numpy(float)
    mean_lt = np.maximum(df["lead_time_mean_wk"].to_numpy(float), 1e-3)
    std_lt = df["lead_time_std_wk"].to_numpy(float) * scenario.std_mult
    delay_p = df["delay_prob"].to_numpy(float)
    delay_wk = df["delay_extra_days"].to_numpy(float) / 7.0

    if scenario.affected_region:
        aff = df["supplier_region"].to_numpy() == scenario.affected_region
    else:
        aff = np.zeros(n, dtype=bool)

    # Lognormal parameters that match the target mean and std
    sig2 = np.log1p((std_lt / mean_lt) ** 2)
    mu = np.log(mean_lt) - sig2 / 2
    sig = np.sqrt(sig2)

    # Start each SKU mid-cycle: ROP plus half an order
    init = (df["rop"] + df["order_qty"] / 2).to_numpy(float)
    on_hand = np.tile(init, (R, 1))
    on_order = np.zeros((R, n))
    arrivals = np.zeros((T + MAX_LT + 2, R, n))

    sales_tot = np.zeros((R, n))
    lost_tot = np.zeros((R, n))
    inv_sum = np.zeros((R, n))
    so_weeks = np.zeros((R, n))
    n_orders = np.zeros((R, n))
    ts = {k: np.zeros((T, R)) for k in
          ("inv_value", "sales_val", "demand_val", "lost_units")}

    for t in range(T):
        arr = arrivals[t]
        on_hand = on_hand + arr
        on_order = on_order - arr

        dem = demand[t] * scenario.demand_mult[t]
        sales = np.minimum(on_hand, dem)
        lost = dem - sales
        on_hand = on_hand - sales

        sales_tot += sales
        lost_tot += lost
        inv_sum += on_hand
        so_weeks += (lost > 1e-9)
        ts["inv_value"][t] = (on_hand * uc).sum(axis=1)
        ts["sales_val"][t] = (sales * price).sum(axis=1)
        ts["demand_val"][t] = (dem * price).sum()
        ts["lost_units"][t] = lost.sum(axis=1)

        # Ordering decision
        qty = policy.order_qty(t, on_hand, on_order)
        placed = qty > 0

        # Always draw (keeps random numbers aligned across scenarios)
        lt = rng.lognormal(mu, sig, size=(R, n))
        lt = lt + (rng.random((R, n)) < delay_p) * delay_wk
        lt = lt + aff * (scenario.extra_days_by_week[t] / 7.0)
        L = np.clip(np.ceil(lt), 1, MAX_LT).astype(int)

        if placed.any():
            r_idx, s_idx = np.nonzero(placed)
            np.add.at(arrivals, (t + L[r_idx, s_idx], r_idx, s_idx), qty[r_idx, s_idx])
            on_order = on_order + qty
            n_orders += placed

    total_dem = (demand * scenario.demand_mult[:, None])
    return {
        "T": T,
        "demand": total_dem.sum(axis=0),
        "demand_weeks": (total_dem > 0).sum(axis=0),
        "sales": sales_tot,
        "lost": lost_tot,
        "avg_onhand": inv_sum / T,
        "stockout_weeks": so_weeks,
        "n_orders": n_orders,
        "ts": ts,
    }