"""Table builders on a tiny synthetic two-event fixture."""

from __future__ import annotations

import pandas as pd
import pytest

from pricing.normalize import DOOR_ORDINAL
from pricing.tables import build_event_tier, build_events, build_transactions, write_table
from tests.synthetic import fake_costs, fake_transactions


@pytest.fixture
def built():
    events = build_events(fake_costs())
    transactions = build_transactions(fake_transactions(), events)
    event_tier = build_event_tier(transactions)
    return events, transactions, event_tier


def test_events_one_row_per_event_with_cost_stack(built):
    events, _, _ = built
    assert len(events) == 2
    assert list(events["event_id"]) == ["E1", "E2"]
    # 1500 guarantee + 800 venue + 300 security + 200 production + 150 staff + 400 marketing
    assert events.loc[events["event_id"] == "E1", "fixed_cost_stack"].iloc[0] == 3350.0
    assert events.loc[events["event_id"] == "E2", "fixed_cost_stack"].iloc[0] == 4950.0
    assert events["event_date"].dtype.kind == "M"


def test_events_drops_pii(built):
    events, _, _ = built
    assert "finance_contact_email" not in events.columns


def test_transactions_row_count_and_no_pii(built):
    _, transactions, _ = built
    assert len(transactions) == 17  # 10 for E1, 7 for E2
    for col in ["buyer_name", "buyer_email", "phone", "address", "postcode"]:
        assert col not in transactions.columns


def test_transactions_normalised_and_joined(built):
    _, transactions, _ = built
    e1_eb = transactions[
        (transactions["event_id"] == "E1") & (transactions["tier_name"] == "Early Bird")
    ]
    assert len(e1_eb) == 4
    assert set(e1_eb["tier_ordinal"]) == {0}
    # Fee treatment is UNKNOWN, so buyer_price is provisionally face + fee.
    assert set(e1_eb["buyer_price"]) == {9.0}
    assert set(e1_eb["brand"]) == {"Brand A"}
    assert (e1_eb["lead_time_days"] > 0).all()

    door = transactions[transactions["tier_name"] == "Door"]
    assert set(door["tier_ordinal"]) == {DOOR_ORDINAL}


def test_comps_flagged_and_excluded_from_sales(built):
    _, transactions, event_tier = built
    comps = transactions[transactions["is_comp"]]
    assert len(comps) == 2
    assert set(comps["tier_name"]) == {"Guestlist", "Comp"}

    # Comp rows carry no units_sold, only units_comp.
    comp_rows = event_tier[event_tier["tier_ordinal"].isna()]
    assert len(comp_rows) == 2
    assert comp_rows["units_sold"].sum() == 0
    assert comp_rows["units_comp"].sum() == 2


def test_event_tier_panel_shape_and_windows(built):
    _, _, event_tier = built
    # E1: ordinals 0, 1, Door + one unmapped comp row. E2: 0, 1, Door + one comp row.
    assert len(event_tier) == 8

    e1_eb = event_tier[(event_tier["event_id"] == "E1") & (event_tier["tier_ordinal"] == 0)].iloc[0]
    assert e1_eb["units_sold"] == 4
    assert e1_eb["n_orders"] == 4
    assert e1_eb["price"] == 9.0
    assert e1_eb["face_price"] == 8.0
    assert e1_eb["window_open"] == pd.Timestamp("2026-01-10")
    assert e1_eb["window_close"] == pd.Timestamp("2026-01-10 18:00")
    assert e1_eb["lead_time_open_days"] == 50

    # "Release 2" on E2 normalises to the same ordinal as "Tier 1" on E1.
    e2_t1 = event_tier[(event_tier["event_id"] == "E2") & (event_tier["tier_ordinal"] == 1)].iloc[0]
    assert e2_t1["tier_name"] == "Release 2"
    assert e2_t1["units_sold"] == 2


def test_uk_dates_are_read_day_first_and_iso_dates_are_not():
    """SPEC data is UK dd/mm/yyyy. "03/04/2024" is 3 April; "2024-03-04" is 4 March."""
    costs = fake_costs()
    costs.loc[0, "event_date"] = "03/04/2024"  # 3 April, UK style
    costs.loc[1, "event_date"] = "2024-03-04"  # 4 March, ISO — must not be flipped
    events = build_events(costs)
    dates = dict(zip(events["event_id"], events["event_date"]))
    assert dates["E1"] == pd.Timestamp("2024-04-03")
    assert dates["E2"] == pd.Timestamp("2024-03-04")


def test_mixed_gmt_and_bst_timestamps_do_not_kill_the_run():
    """A column mixing +00:00 and +01:00 used to raise mid-ingest; lead time needs naive
    London wall-clock on both sides of the subtraction."""
    events = build_events(fake_costs())
    tx = fake_transactions()
    tx["purchased_at"] = ["2026-01-05T20:00:00+00:00"] * 8 + ["2026-06-05T20:00:00+01:00"] * 9
    transactions = build_transactions(tx, events)
    assert transactions["purchased_at"].dt.tz is None
    assert set(transactions["purchased_at"].dt.hour) == {20}  # wall-clock preserved
    assert transactions["lead_time_days"].notna().all()


def test_cancelled_column_of_y_and_n_is_read_correctly():
    """`.astype(bool)` on "N" is True, which would have marked every event cancelled."""
    costs = fake_costs()
    costs["cancelled"] = ["N", "Y"]
    events = build_events(costs)
    assert list(events.sort_values("event_id")["cancelled"]) == [False, True]


def test_prices_with_currency_symbols_are_not_free_tickets():
    """A "£10.00" price column made every ticket a comp and units_sold zero."""
    events = build_events(fake_costs())
    tx = fake_transactions()
    tx["price_paid"] = [f"£{p:,.2f}" for p in [10.0] * 8 + [1250.0] * 9]
    transactions = build_transactions(tx, events)
    assert set(transactions["price_paid"]) == {10.0, 1250.0}
    # Only the two comp-by-NAME rows (Guestlist, Comp) survive as comps.
    assert transactions["is_comp"].sum() == 2


def test_duplicate_event_rows_are_refused_before_they_fan_out_the_join():
    """A totals row or a per-year sheet repeating an event doubles units_sold silently."""
    costs = pd.concat([fake_costs(), fake_costs().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="event_id must be unique"):
        build_events(costs)


def test_event_tier_keeps_one_row_per_tier_not_per_ordinal():
    """Five tiers at five prices in one event must stay five rows. Grouping on the
    ordinal collapsed them (SPEC 4.2 identifies beta off exactly this variation)."""
    events = build_events(fake_costs().iloc[[0]])
    tiers = ["Tier 1", "Release 2", "Standard", "General Admission", "VIP"]
    prices = [12.0, 14.0, 10.0, 16.0, 20.0]
    tx = pd.DataFrame(
        {
            "order_id": [f"FAKE-ORD-{i:03d}" for i in range(5)],
            "event_key": "E1",
            "tier_name": tiers,
            "price_paid": prices,
            "booking_fee": 0.0,
            "purchased_at": pd.Timestamp("2026-01-10"),
            "quantity": 1,
            "platform": "fixr",
        }
    )
    panel = build_event_tier(build_transactions(tx, events))
    assert len(panel) == 5
    assert sorted(panel["tier_name"]) == sorted(tiers)
    assert sorted(panel["price"]) == sorted(prices)
    assert panel["units_sold"].sum() == 5


def test_a_door_sale_on_the_night_is_zero_days_out_not_minus_one():
    """event_date is midnight, so subtracting a 23:30 timestamp gave -1: bought half an
    hour before midnight, "one day after" an event that was still going on. Lead time is
    whole calendar days between the two DATES, and it is never negative."""
    events = build_events(fake_costs())
    tx = fake_transactions()
    tx.loc[tx["tier_name"] == "Door", "purchased_at"] = pd.Timestamp("2026-03-01 23:30")
    tx.loc[tx["tier_name"] == "Door", "event_key"] = "E1"  # event_date 2026-03-01
    transactions = build_transactions(tx, events)

    door = transactions[transactions["tier_name"] == "Door"]
    assert list(door["lead_time_days"]) == [0, 0, 0]
    assert (transactions["lead_time_days"] >= 0).all()


def test_event_tier_window_lead_time_uses_the_same_date_floor():
    events = build_events(fake_costs())
    tx = fake_transactions()
    tx.loc[tx["tier_name"] == "Door", "purchased_at"] = pd.Timestamp("2026-02-28 23:30")
    tx.loc[tx["tier_name"] == "Door", "event_key"] = "E1"
    panel = build_event_tier(build_transactions(tx, events))
    door = panel[(panel["event_id"] == "E1") & (panel["tier_name"] == "Door")].iloc[0]
    assert door["lead_time_open_days"] == 1  # 28 Feb -> 1 Mar, not 0 for a 23:30 sale


def test_fee_treatment_travels_with_the_price(built):
    """SPEC 8.6. buyer_price is written as price_paid + booking_fee while the treatment is
    UNKNOWN, so the guess must be carried on the row — not only mentioned in a report that
    nothing downstream reads."""
    _, transactions, event_tier = built
    assert set(transactions["fee_treatment"]) == {"UNKNOWN"}
    assert set(event_tier["fee_treatment"]) == {"UNKNOWN"}
    # And the provisional number it marks: face 8.00 + fee 1.00.
    e1_eb = transactions[
        (transactions["event_id"] == "E1") & (transactions["tier_name"] == "Early Bird")
    ]
    assert set(e1_eb["buyer_price"]) == {9.0}


def test_fee_treatment_survives_the_parquet_write(built, tmp_path):
    """The marker is useless if the allowlist strips it on the way out."""
    _, transactions, event_tier = built
    for frame, name in [(transactions, "transactions"), (event_tier, "event_tier")]:
        written = pd.read_parquet(write_table(frame, name, out_dir=tmp_path))
        assert set(written["fee_treatment"]) == {"UNKNOWN"}


def test_event_tier_price_is_the_opening_price_not_the_modal_one():
    """SPEC 4.3/8.2. A tier corrected from £10 to £14 mid-window is exactly the price
    variation at a given lead time that identification needs. The modal price buried it
    (and used the whole window's outcome to describe its own opening), so `price` is the
    first price observed and `n_distinct_prices` shows the change happened."""
    events = build_events(fake_costs().iloc[[0]])
    tx = pd.DataFrame(
        {
            "order_id": [f"FAKE-ORD-{i:03d}" for i in range(5)],
            "event_key": "E1",
            "tier_name": "Tier 1",
            # One ticket at the opening price, then four after the correction: the mode is
            # 14.0 and would have hidden the £10 the tier actually opened at.
            "price_paid": [10.0, 14.0, 14.0, 14.0, 14.0],
            "booking_fee": 0.0,
            "purchased_at": pd.to_datetime(
                ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
            ),
            "quantity": 1,
            "platform": "fixr",
        }
    )
    panel = build_event_tier(build_transactions(tx, events))
    row = panel.iloc[0]
    assert row["price"] == 10.0
    assert row["face_price"] == 10.0
    assert row["n_distinct_prices"] == 2
    assert row["units_sold"] == 5


def test_opening_price_is_the_first_ticket_sold_not_the_first_row_in_the_file():
    """Rows arrive in export order, which is not sale order."""
    events = build_events(fake_costs().iloc[[0]])
    tx = pd.DataFrame(
        {
            "order_id": ["FAKE-ORD-001", "FAKE-ORD-002"],
            "event_key": "E1",
            "tier_name": "Tier 1",
            "price_paid": [14.0, 10.0],  # the £10 sale came first in TIME
            "booking_fee": 0.0,
            "purchased_at": pd.to_datetime(["2026-01-09", "2026-01-05"]),
            "quantity": 1,
            "platform": "fixr",
        }
    )
    panel = build_event_tier(build_transactions(tx, events))
    assert panel.iloc[0]["price"] == 10.0


def test_a_stable_tier_reports_one_price(built):
    """n_distinct_prices is only a signal if it stays 1 when nothing moved."""
    _, _, event_tier = built
    paid = event_tier[event_tier["units_sold"] > 0]
    assert set(paid["n_distinct_prices"]) == {1}


def test_write_table_round_trips_parquet(built, tmp_path):
    _, transactions, _ = built
    path = write_table(transactions, "transactions", out_dir=tmp_path)
    assert path.exists()
    assert len(pd.read_parquet(path)) == len(transactions)


def test_write_table_refuses_to_write_pii(tmp_path):
    leaky = pd.DataFrame({"order_id": ["FAKE-ORD-001"], "buyer_email": ["x@example.invalid"]})
    with pytest.raises(ValueError, match="buyer PII"):
        write_table(leaky, "transactions", out_dir=tmp_path)
    assert not (tmp_path / "transactions.parquet").exists()


def test_write_table_refuses_a_column_no_pattern_would_have_caught(tmp_path):
    """Fail-closed: "Lead Booker" beats a PII pattern list, but not an allowlist."""
    leaky = pd.DataFrame({"order_id": ["FAKE-ORD-001"], "lead_booker": ["FAKE Buyer 001"]})
    with pytest.raises(ValueError, match="allowlist"):
        write_table(leaky, "transactions", out_dir=tmp_path)
    assert not (tmp_path / "transactions.parquet").exists()


def test_derived_dir_default_is_inside_the_repo():
    """Running ingest from another directory must not write real tables outside the
    gitignored data/ tree."""
    from pricing.tables import DERIVED_DIR, REPO_ROOT

    assert DERIVED_DIR.is_absolute()
    assert DERIVED_DIR == REPO_ROOT / "data" / "derived"
    assert (REPO_ROOT / "SPEC.md").exists()
