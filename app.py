"""LUMEN dashboard: Lean Unified Materials & Execution Network.
Run from the project folder:  streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PROC, OUT = ROOT / "data" / "processed", ROOT / "outputs"

st.set_page_config(page_title="LUMEN", page_icon="💡", layout="wide")


def money(v):
    return f"${v/1e6:.2f}M" if abs(v) >= 1e6 else f"${v/1e3:,.0f}K"


@st.cache_data
def load():
    pol = pd.read_csv(PROC / "sku_policy.csv", index_col=0, dtype={"StockCode": str})
    pol.index = pol.index.astype(str)
    return (pol, pd.read_csv(OUT / "kpi_summary.csv"),
            pd.read_csv(OUT / "scenario_results.csv"),
            pd.read_csv(OUT / "waste_table.csv"))


@st.cache_resource
def sim_inputs():
    from src.simulation.inputs import load_inputs
    return load_inputs()


pol, kpi, sc, waste = load()

# ---------------- Sidebar filters ----------------
st.sidebar.title("💡 LUMEN")
st.sidebar.caption("Lean Unified Materials & Execution Network")
abc_sel = st.sidebar.multiselect("ABC class", ["A", "B", "C"], ["A", "B", "C"])
reg_sel = st.sidebar.multiselect("Supplier region", sorted(pol["supplier_region"].unique()),
                                 sorted(pol["supplier_region"].unique()))
light_only = st.sidebar.checkbox("Lighting sub-catalogue only")
df = pol[pol["abc"].isin(abc_sel) & pol["supplier_region"].isin(reg_sel)]
if light_only:
    df = df[df["is_lighting"].astype(bool)]
st.sidebar.metric("SKUs in view", f"{len(df):,}")
st.sidebar.caption("Demand: real UCI Online Retail II (B2B wholesale). "
                   "Supplier lead times, costs and MOQs are modelled (config/suppliers.yaml).")

st.title("LUMEN: Inventory Optimization & Lean Supply Control Tower")
t1, t2, t3, t4, t5, t6 = st.tabs(["Executive summary", "Portfolio (ABC-XYZ)", "Policy explorer",
                                  "Push vs Pull (Lean)", "Scenario stress tests", "Simulation lab"])

# ---------------- 1. Executive summary ----------------
with t1:
    a = kpi[kpi["class"] == "ALL"].set_index("policy")
    push, pull = a.loc["push"], a.loc["pull"]
    c = st.columns(6)
    c[0].metric("Fill rate (pull)", f"{pull.fill_rate*100:.1f}%", f"{(pull.fill_rate-push.fill_rate)*100:+.1f} pts vs push")
    c[1].metric("Stockout rate", f"{pull.stockout_rate*100:.1f}%", f"{(pull.stockout_rate-push.stockout_rate)*100:+.1f} pts", delta_color="inverse")
    c[2].metric("Avg inventory", money(pull.avg_inventory_value), f"{(pull.avg_inventory_value/push.avg_inventory_value-1)*100:+.1f}%", delta_color="inverse")
    c[3].metric("Days of inventory", f"{pull.days_of_inventory:.0f}", f"{pull.days_of_inventory-push.days_of_inventory:+.0f} d", delta_color="inverse")
    c[4].metric("Turnover", f"{pull.turnover:.2f}x", f"{pull.turnover-push.turnover:+.2f}")
    c[5].metric("Total cost-to-serve", money(pull.total_cost), f"{(pull.total_cost/push.total_cost-1)*100:+.1f}%", delta_color="inverse")
    c = st.columns(4)
    c[0].metric("Carrying cost", money(pull.carrying_cost), f"{(pull.carrying_cost/push.carrying_cost-1)*100:+.1f}%", delta_color="inverse")
    c[1].metric("Ordering cost", money(pull.ordering_cost), f"{(pull.ordering_cost/push.ordering_cost-1)*100:+.1f}%", delta_color="inverse")
    c[2].metric("Lost margin (stockouts)", money(pull.lost_margin), f"{(pull.lost_margin/push.lost_margin-1)*100:+.1f}%", delta_color="inverse")
    c[3].metric("Orders / year", f"{pull.orders_per_year:,.0f}", f"{(pull.orders_per_year/push.orders_per_year-1)*100:+.1f}%", delta_color="inverse")

    ap = kpi[(kpi["class"] == "A") & (kpi["policy"] == "pull")].iloc[0]
    (st.success if ap.fill_rate >= 0.95 else st.warning)(
        f"A-class fill rate under pull: {ap.fill_rate*100:.1f}% (target ≥ 95%)")

    comp = pd.DataFrame({"Push": [push.carrying_cost, push.ordering_cost, push.lost_margin],
                         "Pull": [pull.carrying_cost, pull.ordering_cost, pull.lost_margin]},
                        index=["Carrying", "Ordering", "Lost margin"]).T.reset_index()
    fig = px.bar(comp.melt("index"), x="index", y="value", color="variable", barmode="stack",
                 labels={"index": "", "value": "Annual cost ($)", "variable": "Component"},
                 title="Cost-to-serve breakdown")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Read this honestly: pull holds slightly more stock than the default push at baseline. "
            "The saving comes from fewer orders and fewer stockouts. The fair test is equal service "
            "(see Simulation lab, equal-service sweep).")

# ---------------- 2. Portfolio ----------------
with t2:
    c = st.columns(4)
    c[0].metric("SKUs", f"{len(df):,}")
    c[1].metric("Inventory value (policy)", money(df.avg_inventory_value.sum()))
    c[2].metric("Annual policy cost", money(df.annual_policy_cost.sum()))
    c[3].metric("Avg target service", f"{df.service_level.mean()*100:.1f}%")

    l, r = st.columns(2)
    p = df.sort_values("hist_revenue", ascending=False)
    p["cum_rev"] = p["hist_revenue"].cumsum() / p["hist_revenue"].sum() * 100
    p["cum_sku"] = np.arange(1, len(p) + 1) / len(p) * 100
    f = px.line(p, x="cum_sku", y="cum_rev", title="Pareto curve (revenue vs SKU share)",
                labels={"cum_sku": "% of SKUs", "cum_rev": "% of revenue"})
    f.add_hline(y=80, line_dash="dash")
    l.plotly_chart(f, use_container_width=True)
    ct = pd.crosstab(df["abc"], df["xyz"])
    r.plotly_chart(px.imshow(ct, text_auto=True, color_continuous_scale="Blues",
                             title="ABC x XYZ (SKU count)"), use_container_width=True)
    l, r = st.columns(2)
    l.plotly_chart(px.treemap(df.reset_index(), path=["abc", "xyz"], values="avg_inventory_value",
                              title="Where the inventory money sits"), use_container_width=True)
    r.plotly_chart(px.sunburst(df.reset_index(), path=["supplier_region", "abc"], values="avg_inventory_value",
                               title="Inventory value by supplier region and class"), use_container_width=True)
    l, r = st.columns(2)
    l.plotly_chart(px.bar(df["pattern"].value_counts().reset_index(), x="pattern", y="count",
                          title="Demand pattern mix (smooth / intermittent / erratic / lumpy)"),
                   use_container_width=True)
    r.plotly_chart(px.box(df, x="supplier_region", y="lead_time_mean_wk", color="abc",
                          title="Lead time (weeks) by supplier region"), use_container_width=True)

# ---------------- 3. Policy explorer ----------------
with t3:
    cols = ["description", "abc_xyz", "pattern", "supplier_region", "policy", "service_level",
            "mean_wk", "std_wk", "cv", "lead_time_mean_wk", "eoq", "moq", "order_qty",
            "safety_stock", "rop", "max_level", "avg_inventory_value", "annual_policy_cost"]
    st.dataframe(df[cols].round(2), use_container_width=True, height=300)
    st.download_button("Download policy table (CSV)", df[cols].round(2).to_csv().encode(), "policy_table.csv")

    st.subheader("SKU deep-dive: service level vs cost trade-off")
    if len(df):
        top = df.sort_values("annual_policy_cost", ascending=False).head(500)
        sku = st.selectbox("SKU", top.index, format_func=lambda s: f"{s} | {top.loc[s,'description'][:40]}")
        s = df.loc[sku]
        c = st.columns(6)
        for col, (k, v) in zip(c, [("Class", s.abc_xyz), ("Policy", s.policy), ("EOQ", f"{s.eoq:.0f}"),
                                   ("Safety stock", f"{s.safety_stock:.0f}"), ("ROP", f"{s.rop:.0f}"),
                                   ("Max level", f"{s.max_level:.0f}")]):
            col.metric(k, v)
        sl = np.linspace(0.80, 0.999, 40)
        ss = norm.ppf(sl) * np.sqrt(s.lead_time_mean_wk * (0 if np.isnan(s.std_wk) else s.std_wk) ** 2
                                    + s.mean_wk ** 2 * s.lead_time_std_wk ** 2)
        cost = ss * s.holding_cost_per_unit_yr
        fig = go.Figure(go.Scatter(x=sl * 100, y=cost, mode="lines", name="Annual cost of safety stock"))
        fig.add_vline(x=s.service_level * 100, line_dash="dash", annotation_text="current target")
        fig.update_layout(xaxis_title="Cycle service level (%)", yaxis_title="Safety stock holding cost ($/yr)",
                          title="Each extra point of service costs more than the last")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- 4. Push vs Pull ----------------
with t4:
    metric = st.selectbox("Metric", ["fill_rate", "stockout_rate", "avg_inventory_value", "turnover",
                                     "days_of_inventory", "carrying_cost", "ordering_cost",
                                     "lost_margin", "total_cost", "orders_per_year"])
    st.plotly_chart(px.bar(kpi, x="class", y=metric, color="policy", barmode="group",
                           title=f"{metric} by ABC class: push vs pull"), use_container_width=True)
    st.subheader("Muda (7 wastes) read through inventory")
    w = waste.copy()
    st.dataframe(w, use_container_width=True)
    wm = w.dropna(subset=["push_cost"]).melt(["muda"], ["push_cost", "pull_cost"])
    st.plotly_chart(px.bar(wm, x="muda", y="value", color="variable", barmode="group",
                           title="Quantified waste ($/yr)"), use_container_width=True)
    st.caption("Motion and Over-processing are not simulated. Waste totals compare policies at different "
               "service levels, so read them next to the equal-service sweep.")

# ---------------- 5. Scenarios ----------------
with t5:
    m = st.selectbox("Metric ", ["fill_rate", "fill_drop_pts", "stockout_rate", "lost_margin",
                                 "avg_inventory_value", "total_cost"], key="scm")
    st.plotly_chart(px.bar(sc, x="scenario", y=m, color="policy", barmode="group",
                           title=f"{m} across scenarios"), use_container_width=True)
    pv = sc.pivot(index="scenario", columns="policy", values="fill_drop_pts")
    pv["pull_advantage_pts"] = pv["push"] - pv["pull"]
    st.subheader("Fill-rate drop vs baseline (points)")
    st.dataframe(pv.round(2), use_container_width=True)
    worst = sc[sc.policy == "pull"].sort_values("fill_rate").iloc[0]
    st.warning(f"Worst pull scenario: {worst.scenario} at {worst.fill_rate*100:.1f}% fill. "
               "Note: pull degrades faster than push under lead-time-variability and supplier-delay "
               "scenarios, because its safety stock is tuned tightly. Re-estimate it when supplier "
               "performance shifts.")

# ---------------- 6. Simulation lab ----------------
with t6:
    from src.kpi.metrics import kpis
    from src.simulation.engine import simulate
    from src.simulation.pull_policy import PullPolicy
    from src.simulation.push_policy import PushPolicy
    from src.simulation.scenarios import Scenario

    params, sdf, demand = sim_inputs()
    T = demand.shape[0]
    h = params["holding_cost_pct"]
    st.caption("Live Monte Carlo on the real test-window demand. Randomness comes from lead times.")
    c1, c2, c3 = st.columns(3)
    cycle = c1.slider("Push review cycle (weeks)", 1, 8, 4)
    buf = c1.slider("Push buffer (weeks)", 0, 12, 3)
    runs = c1.slider("Monte Carlo runs", 5, 30, 10)
    sp_m = c2.slider("Demand spike multiplier", 1.0, 4.0, 1.0, 0.5)
    sp_s = c2.slider("Spike start week", 0, T - 1, 10)
    sp_d = c2.slider("Spike duration (weeks)", 1, 26, 8)
    region = c3.selectbox("Delayed region", ["none"] + sorted(sdf["supplier_region"].unique()))
    dl = c3.slider("Extra delay (days)", 0, 84, 0, 7)
    dl_s = c3.slider("Delay start week", 0, T - 1, 8)
    dl_d = c3.slider("Delay duration (weeks)", 1, 40, 20)
    stdm = c3.slider("Lead-time std multiplier", 1.0, 3.0, 1.0, 0.5)
    sweep = st.checkbox("Also run equal-service sweep (slower)")

    if st.button("Run simulation", type="primary"):
        mult = np.ones(T); mult[sp_s:sp_s + sp_d] = sp_m
        extra = np.zeros(T); extra[dl_s:dl_s + dl_d] = dl
        scn = Scenario("custom", mult, extra, None if region == "none" else region, stdm)
        seed = params["random_seed"]
        with st.spinner("Simulating..."):
            res = {p.name: simulate(sdf, demand, p, scn, runs, seed)
                   for p in (PushPolicy(sdf, cycle, buf), PullPolicy(sdf))}
        a_mask = (sdf["abc"] == "A").to_numpy()
        rows = []
        for name, r_ in res.items():
            for lab, mk in (("ALL", None), ("A-class", a_mask)):
                rows.append({"policy": name, "scope": lab, **kpis(r_, sdf, h, mk)})
        tab = pd.DataFrame(rows)
        st.dataframe(tab.round(3), use_container_width=True)
        pa = tab[(tab.policy == "pull") & (tab.scope == "A-class")].fill_rate.iloc[0]
        (st.success if pa >= 0.95 else st.error)(f"Pull A-class fill rate: {pa*100:.1f}% (target ≥ 95%)")

        l, r = st.columns(2)
        inv = pd.DataFrame({n: x["ts"]["inv_value"].mean(axis=1) for n, x in res.items()})
        l.plotly_chart(px.line(inv, title="Inventory value by week", labels={"index": "Test week", "value": "$ at cost"}),
                       use_container_width=True)
        fr = pd.DataFrame({n: x["ts"]["sales_val"].mean(axis=1) / x["ts"]["demand_val"].mean(axis=1)
                           for n, x in res.items()})
        r.plotly_chart(px.line(fr, title="Weekly fill rate (watch the dip and recovery)",
                               labels={"index": "Test week", "value": "Fill rate"}), use_container_width=True)

        if sweep:
            pts = []
            for b in [0, 2, 4, 6, 8, 12]:
                k = kpis(simulate(sdf, demand, PushPolicy(sdf, cycle, b), scn, runs, seed), sdf, h)
                pts.append({"buffer": b, "inventory": k["avg_inventory_value"], "fill": k["fill_rate"] * 100})
            pts = pd.DataFrame(pts)
            kp = kpis(res["pull"], sdf, h)
            fig = px.line(pts, x="inventory", y="fill", markers=True, text="buffer",
                          title="Equal-service test: inventory needed per fill rate")
            fig.add_trace(go.Scatter(x=[kp["avg_inventory_value"]], y=[kp["fill_rate"] * 100], mode="markers",
                                     marker=dict(size=16, color="red", symbol="star"), name="Pull"))
            fig.update_layout(xaxis_title="Avg inventory value ($)", yaxis_title="Fill rate (%)")
            st.plotly_chart(fig, use_container_width=True)