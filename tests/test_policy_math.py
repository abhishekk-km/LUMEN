"""Sanity checks on EOQ / safety stock / ROP."""
"""Sanity checks on the policy formulas.

Run:
    python tests/test_policy_math.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.policy.eoq import eoq
from src.policy.safety_stock import safety_stock
from src.policy.reorder_point import reorder_point


def test_eoq_known_value():
    # sqrt(2 * 1000 * 50 / 5) = 141.42
    assert abs(float(eoq(1000, 50, 5)) - 141.4214) < 1e-3


def test_eoq_grows_with_order_cost():
    assert float(eoq(1000, 100, 5)) > float(eoq(1000, 50, 5))


def test_safety_stock_demand_only():
    # z(0.95)=1.6449, L=4, sigma_d=2 -> 1.6449 * sqrt(4*4) = 6.58
    ss = float(safety_stock(0.95, 10, 2, 4, 0))
    assert abs(ss - 6.5794) < 1e-2


def test_safety_stock_rises_with_service_level():
    assert float(safety_stock(0.99, 10, 2, 4, 1)) > float(safety_stock(0.90, 10, 2, 4, 1))


def test_safety_stock_rises_with_lead_time_variability():
    assert float(safety_stock(0.95, 10, 2, 4, 2)) > float(safety_stock(0.95, 10, 2, 4, 0))


def test_rop():
    # 10/week * 4 weeks + 7 safety stock = 47
    assert float(reorder_point(10, 4, 7)) == 47.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed.")