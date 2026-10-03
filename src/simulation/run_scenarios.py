"""Stress-test push and pull under demand spike, supplier delay, LT variability.

Run:
    python src/simulation/run_scenarios.py
"""
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.simulation.inputs import load_inputs, load_yaml, OUTPUTS
from src.simulation.engine import simulate
from src.simulation.pull_policy import PullPolicy
from src.simulation.push_policy import PushPolicy
from src.simulation.scenarios import build_all
from src.kpi.metrics import kpis
from src.viz.plots import plot_scenarios

N_RUNS = 30


def main():
    params, df, demand = load_inputs()
    cfg = load_yaml("scenarios.yaml")
    seed = params["random_seed"]
    h = params["holding_cost_pct"]
    T = demand.shape[0]

    policies = [PushPolicy(df, 4, 3), PullPolicy(df)]
    rows = []
    for sc in build_all(T, cfg):
        for pol in policies:
            res = simulate(df, demand, pol, sc, N_RUNS, seed)
            k: dict[str, Any] = dict(kpis(res, df, h))
            k["scenario"] = sc.name
            k["policy"] = pol.name
            rows.append(k)
            print(f"done {sc.name:<28} {pol.name}")

    out = pd.DataFrame(rows)
    base = out[out["scenario"] == "baseline"].set_index("policy").to_dict("index")
    out["fill_drop_pts"] = [
        (base[r.policy]["fill_rate"] - r.fill_rate) * 100 for r in out.itertuples()]
    out["inventory_change_pct"] = [
        (r.avg_inventory_value / base[r.policy]["avg_inventory_value"] - 1) * 100
        for r in out.itertuples()]

    cols = ["scenario", "policy", "fill_rate", "fill_drop_pts", "stockout_rate",
            "avg_inventory_value", "inventory_change_pct", "days_of_inventory",
            "lost_margin", "total_cost"]
    out = out[cols]
    out.round(3).to_csv(OUTPUTS / "scenario_results.csv", index=False)
    plot_scenarios(out, OUTPUTS / "figures" / "scenario_fill_rate.png")

    print("\n=== SCENARIO RESULTS ===")
    print(out[["scenario", "policy", "fill_rate", "fill_drop_pts", "avg_inventory_value",
               "lost_margin"]].round(3).to_string(index=False))
    print("\nSaved: outputs/scenario_results.csv and figures/scenario_fill_rate.png")


if __name__ == "__main__":
    main()