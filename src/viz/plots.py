"""Plotting helpers."""
"""Plot helpers (saved to files, no pop-up windows)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_weekly_inventory(ts_push, ts_pull, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for ts, label in [(ts_push, "Push"), (ts_pull, "Pull")]:
        ax.plot(ts["inv_value"].mean(axis=1), label=label)
    ax.set_title("Inventory value over the test window")
    ax.set_xlabel("Week of test window")
    ax.set_ylabel("Inventory value (at cost)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_service_cost_curve(push_points, pull_point, path):
    """push_points: list of (buffer_wk, inventory_value, fill_rate)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [p[1] for p in push_points]
    y = [p[2] * 100 for p in push_points]
    ax.plot(x, y, "o-", label="Push (flat buffer sweep)")
    for b, xv, yv in push_points:
        ax.annotate(f"{b}w", (xv, yv * 100), textcoords="offset points", xytext=(4, 4))
    ax.plot([pull_point[0]], [pull_point[1] * 100], "r*", markersize=16, label="Pull (class-based)")
    ax.set_xlabel("Average inventory value (at cost)")
    ax.set_ylabel("Fill rate (%)")
    ax.set_title("Service vs inventory: same fill rate, how much stock?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenarios(df, path):
    """df columns: scenario, policy, fill_rate."""
    piv = df.pivot(index="scenario", columns="policy", values="fill_rate") * 100
    ax = piv.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Fill rate (%)")
    ax.set_title("Fill rate under stress scenarios")
    ax.set_ylim(max(0, np.nanmin(piv.values) - 5), 100)
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()