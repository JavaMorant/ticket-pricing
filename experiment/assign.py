#!/usr/bin/env python3
"""Deterministic arm assignment — autumn 2026 randomised tier-pricing experiment.

FROZEN 2026-08-10 with experiment/DESIGN.md. Do not modify after freeze.

Assignment: permuted blocks of size 2 in announcement order within each design
block (B1..B4). The first event of pair p in block b draws its arm from the
parity of the first byte of SHA-256("{SEED}:{b}:{p}"); its pair-mate gets the
opposite arm. Every assignment is recomputable from this file alone.

Rounding: prices shift by ±15% then round to the nearest £0.50 in exact integer
arithmetic (units of £1/2000, so both the shifted value and the 50p grid are
exact); ties round AWAY from the baseline rung so a tie amplifies the dose.

Usage:
    python3 assign.py --block B1 --index 1 --baseline 6,8,10,11
"""

from __future__ import annotations

import argparse
import hashlib

SEED = "ticket-pricing-autumn-2026-v1"
BLOCKS = ("B1", "B2", "B3", "B4")
HIGH_PCT = 115
LOW_PCT = 85

# £1 = 2000 units, so a 15% shift of any whole-pence price and the 50p grid
# (1000 units) are both exactly representable.
UNITS_PER_POUND = 2000
GRID = UNITS_PER_POUND // 2  # £0.50


def pair_first_arm(block: str, pair: int) -> str:
    digest = hashlib.sha256(f"{SEED}:{block}:{pair}".encode()).digest()
    return "HIGH" if digest[0] & 1 else "LOW"


def arm_for(block: str, index: int) -> str:
    if block not in BLOCKS:
        raise ValueError(f"unknown block {block!r}; expected one of {BLOCKS}")
    if index < 1:
        raise ValueError("index is 1-based")
    pair = (index + 1) // 2
    first = pair_first_arm(block, pair)
    if index % 2 == 1:
        return first
    return "LOW" if first == "HIGH" else "HIGH"


def _to_units(pounds: str) -> int:
    """Parse '6' or '9.50' into exact price units, rejecting sub-penny input."""
    value = round(float(pounds) * 100)
    if abs(float(pounds) * 100 - value) > 1e-6:
        raise ValueError(f"price {pounds!r} is not a whole number of pence")
    return value * (UNITS_PER_POUND // 100)


def shifted_rung(baseline_units: int, arm: str) -> int:
    pct = HIGH_PCT if arm == "HIGH" else LOW_PCT
    value = baseline_units * pct // 100  # exact: units divisible by 20
    lo = (value // GRID) * GRID
    hi = lo + GRID
    d_lo, d_hi = value - lo, hi - value
    if d_lo == d_hi:  # exact tie: round away from the baseline rung
        return hi if abs(hi - baseline_units) > abs(lo - baseline_units) else lo
    return lo if d_lo < d_hi else hi


def realised_ladder(baseline: list[int], arm: str) -> list[int]:
    return [shifted_rung(rung, arm) for rung in baseline]


def fmt(units: int) -> str:
    return f"£{units / UNITS_PER_POUND:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", required=True, choices=BLOCKS)
    ap.add_argument("--index", required=True, type=int, help="1-based announcement order within the block")
    ap.add_argument("--baseline", required=True, help="comma-separated ladder incl. door, e.g. 6,8,10,11")
    args = ap.parse_args()

    baseline = [_to_units(p) for p in args.baseline.split(",")]
    arm = arm_for(args.block, args.index)
    ladder = realised_ladder(baseline, arm)

    print(f"block={args.block} index={args.index} pair={(args.index + 1) // 2} → arm={arm}")
    print("baseline:", " / ".join(fmt(p) for p in baseline))
    print("realised:", " / ".join(fmt(p) for p in ladder))


if __name__ == "__main__":
    main()
