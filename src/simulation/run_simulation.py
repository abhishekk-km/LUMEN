"""Baseline push vs pull, service-vs-cost curve, and lean waste table.

Run:
    python src/simulation/run_simulation.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.simulation.inputs import load_inputs, OUTPUTS
from src.simulation.engine import simulate
from src.simulation.pull_policy import PullPolicy
from src.simulation.push_policy import PushPolicy
from src.simulation.scenarios import baseline
from src.kpi.metrics import kpis, kpis_by_class
from src.lean.waste_cost import waste_costs
from src.viz.plots import plot_weekly_inventory, plot_service_cost_curve

N_RUNS = 30


def main():
    params, df, demand = load_inputs()
    seed = params["random_seed"]
    h = params["holding_cost_pct"]
    T = demand.shape[0]
    base = baseline(T)
    (OUTPUTS / "figures").mkdir(parents=True, exist_ok=True)
    print(f"SKUs: {len(df):,} | test weeks: {T} | runs: {N_RUNS}")

    res_push = simulate(df, demand, PushPolicy(df, 4, 3), base, N_RUNS, seed)
    res_pull = simulate(df, demand, PullPolicy(df), base, N_RUNS, seed)

    k_push = kpis_by_class(res_push, df, h)
    k_pull = kpis_by_class(res_pull, df, h)
    k_push.insert(0, "policy", "push")
    k_pull.insert(0, "policy", "pull")
    summary = pd.concat([k_push, k_pull])
    summary.index.name = "class"
    summary.round(3).to_csv(OUTPUTS / "kpi_summary.csv")

    show = ["fill_rate", "stockout_rate", "avg_inventory_value", "turnover",
            "days_of_inventory", "total_cost"]
    print("\n=== BASELINE: PUSH vs PULL (all SKUs) ===")
    print(summary.loc["ALL", ["policy"] + show].round(3).to_string())
    print("\n=== BY ABC CLASS ===")
    print(summary.reset_index().query("`class` != 'ALL'")
          [["class", "policy"] + show].round(3).to_string(index=False))

    # Fair comparison: sweep push buffer to see how much stock push needs
    print("\n=== PUSH BUFFER SWEEP (to match pull's service) ===")
    points = []
    for b in [0, 1, 2, 3, 4, 6, 8, 12]:
        r = simulate(df, demand, PushPolicy(df, 4, b), base, N_RUNS, seed)
        k = kpis(r, df, h)
        points.append((b, k["avg_inventory_value"], k["fill_rate"]))
        print(f"buffer {b:>2}w | fill {k['fill_rate']:.3f} | inventory {k['avg_inventory_value']:,.0f}")
    kp = kpis(res_pull, df, h)
    print(f"pull       | fill {kp['fill_rate']:.3f} | inventory {kp['avg_inventory_value']:,.0f}")

    plot_service_cost_curve(points, (kp["avg_inventory_value"], kp["fill_rate"]),
                            OUTPUTS / "figures" / "service_vs_inventory.png")
    plot_weekly_inventory(res_push["ts"], res_pull["ts"],
                          OUTPUTS / "figures" / "weekly_inventory.png")

    # Lean waste (Muda) table
    w_push = waste_costs(res_push, df, h)
    w_pull = waste_costs(res_pull, df, h)
    w = w_push[["muda", "measured_by", "annual_cost"]].rename(columns={"annual_cost": "push_cost"})
    w["pull_cost"] = w_pull["annual_cost"]
    w.round(0).to_csv(OUTPUTS / "waste_table.csv", index=False)
    print("\n=== LEAN WASTE (annual cost) ===")
    print(w.round(0).to_string(index=False))
    print(f"\nTotal quantified waste  push: {w['push_cost'].sum():,.0f} | pull: {w['pull_cost'].sum():,.0f}")
    print("\nSaved: outputs/kpi_summary.csv, waste_table.csv, figures/")


if __name__ == "__main__":
    main()