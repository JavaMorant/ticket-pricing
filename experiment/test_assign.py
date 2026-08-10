"""Regression tests for the frozen assignment mechanism. Run: python3 -m pytest experiment/"""

from assign import UNITS_PER_POUND, arm_for, realised_ladder, shifted_rung


def units(pounds: float) -> int:
    return round(pounds * UNITS_PER_POUND)


def test_pairs_get_opposite_arms():
    for block in ("B1", "B2", "B3", "B4"):
        for pair_start in (1, 3, 5, 7):
            a, b = arm_for(block, pair_start), arm_for(block, pair_start + 1)
            assert {a, b} == {"HIGH", "LOW"}


def test_assignment_is_deterministic():
    assert all(arm_for("B1", i) == arm_for("B1", i) for i in range(1, 9))


def test_rounding_non_tie_cases():
    # B1 HIGH: 6→6.90→7.00, 8→9.20→9.00, 10→11.50 exact, 11→12.65→12.50
    assert realised_ladder([units(6), units(8), units(10), units(11)], "HIGH") == [
        units(7), units(9), units(11.5), units(12.5)
    ]
    # B1 LOW: 6→5.10→5.00, 8→6.80→7.00, 10→8.50 exact, 11→9.35→9.50
    assert realised_ladder([units(6), units(8), units(10), units(11)], "LOW") == [
        units(5), units(7), units(8.5), units(9.5)
    ]


def test_rounding_ties_round_away_from_baseline():
    # £5 × 1.15 = £5.75, tie between 5.50/6.00 → away from £5 → £6.00
    assert shifted_rung(units(5), "HIGH") == units(6)
    # £5 × 0.85 = £4.25, tie between 4.00/4.50 → away from £5 → £4.00
    assert shifted_rung(units(5), "LOW") == units(4)


def test_b2_worked_ladders_match_design_doc():
    base = [units(5), units(7), units(9), units(10)]
    assert realised_ladder(base, "HIGH") == [units(6), units(8), units(10.5), units(11.5)]
    assert realised_ladder(base, "LOW") == [units(4), units(6), units(7.5), units(8.5)]


def test_unit_1_golden_assignment():
    """Pin unit #1 (B1, index 1) so the frozen assignment can never silently drift."""
    assert arm_for("B1", 1) == "LOW"
    assert arm_for("B1", 2) == "HIGH"
