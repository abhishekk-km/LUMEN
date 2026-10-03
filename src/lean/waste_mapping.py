"""Map inventory outcomes to the 7 Muda."""
"""The 7 Muda (wastes) read through an inventory lens, with the KPI that measures each."""
import pandas as pd

MUDA = pd.DataFrame([
    ("Inventory", "Stock above what demand needs ties up cash and space",
     "Days of inventory, carrying cost"),
    ("Overproduction", "Ordering ahead of real consumption (push batches, big MOQs)",
     "Stock beyond 26 weeks of cover, write-down risk"),
    ("Waiting", "Customers and downstream lines wait for stock-outs",
     "Stockout rate, lost margin"),
    ("Transportation", "Frequent small POs and expediting add handling and freight",
     "Orders per year, ordering cost"),
    ("Motion", "Planner time chasing shortages and re-planning",
     "Not simulated"),
    ("Over-processing", "Manual checks and re-forecasting where a simple Kanban rule works",
     "Not simulated"),
    ("Defects", "Obsolete, damaged or aged stock written off",
     "Covered under Overproduction proxy"),
], columns=["muda", "inventory_meaning", "measured_by"])