"""Validation report smoke test, plus the ingest CLI wiring."""

from __future__ import annotations

import pandas as pd

from pricing import ingest
from pricing.tables import build_event_tier, build_events, build_transactions
from pricing.validate import format_report, validate
from tests.synthetic import fake_costs, fake_transactions


def _report():
    events = build_events(fake_costs())
    transactions = build_transactions(fake_transactions(), events)
    event_tier = build_event_tier(transactions)
    return validate(events, transactions, event_tier)


def test_report_has_every_section():
    report = _report()
    assert set(report) == {
        "row_counts",
        "date_ranges",
        "date_parsing",
        "null_rates",
        "duplicates",
        "price_parsing",
        "cancelled",
        "tier_coverage",
        "fee_treatment",
        "join_health",
        "comps",
        "price_variation",
        "unexpected_columns",
    }


def test_row_counts_and_date_ranges():
    report = _report()
    assert report["row_counts"] == {"events": 2, "transactions": 17, "event_tier": 8}
    assert report["date_ranges"]["events.event_date"] == ("2026-03-01", "2026-04-15")
    assert report["date_ranges"]["transactions.purchased_at"][0] == "2026-01-10"


def test_duplicate_order_ids_detected():
    events = build_events(fake_costs())
    tx = fake_transactions()
    doubled = pd.concat([tx, tx.iloc[[0]]], ignore_index=True)
    report = validate(
        events,
        build_transactions(doubled, events),
        build_event_tier(build_transactions(doubled, events)),
    )
    assert report["duplicates"]["fully_duplicated_transaction_rows"] == 1
    assert report["duplicates"]["order_ids_appearing_more_than_once"] == 1


def test_tier_coverage_is_complete_when_every_paid_tier_is_recognised():
    """Comps have no release ordinal by design and must not count as a miss."""
    report = _report()
    assert report["tier_coverage"]["transactions_with_ordinal"] == 1.0
    assert report["tier_coverage"]["unmapped_tier_labels"] == []
    assert report["tier_coverage"]["distinct_ordinals"] == [0, 1, 99]


def test_tier_coverage_reports_unmapped_labels():
    events = build_events(fake_costs())
    tx = fake_transactions()
    tx.loc[0, "tier_name"] = "Mystery Bundle"
    transactions = build_transactions(tx, events)
    report = validate(events, transactions, build_event_tier(transactions))
    assert "Mystery Bundle" in report["tier_coverage"]["unmapped_tier_labels"]
    assert report["tier_coverage"]["transactions_with_ordinal"] < 1.0
    assert "UNMAPPED LABELS" in format_report(report)


def test_fee_treatment_reported_as_unknown():
    report = _report()
    assert report["fee_treatment"] == {"fixr": "UNKNOWN"}
    assert "RESOLVE THIS" in format_report(report)


def test_no_unexpected_columns_in_any_table():
    report = _report()
    assert report["unexpected_columns"] == {"events": [], "transactions": [], "event_tier": []}
    assert "every derived column is on the canonical allowlist" in format_report(report)


def test_report_flags_unparsed_prices_instead_of_treating_them_as_free():
    events = build_events(fake_costs())
    tx = fake_transactions()
    tx["price_paid"] = ["on the door innit", *tx["price_paid"][1:].astype(str)]
    transactions = build_transactions(tx, events)
    report = validate(events, transactions, build_event_tier(transactions))

    assert report["price_parsing"]["transactions_with_unparsed_price"] == 1
    assert "PRICE PARSE FAILURE" in format_report(report)
    assert not transactions.loc[0, "is_comp"]  # an unreadable price is not a free ticket


def test_report_counts_cancelled_events_and_ambiguous_dates():
    report = _report()
    assert report["cancelled"]["cancelled_events"] == 0
    # 2026-03-01 and 2026-04-15: one of the two event dates has a day <= 12.
    assert report["date_parsing"]["events.event_date_ambiguous_share"] == 0.5
    assert "could also read as mm/dd" in format_report(report)


def test_report_flags_tiers_that_share_an_ordinal():
    """Tier 1 and Release 2 both map to ordinal 1 — two prices, one ordinal."""
    events = build_events(fake_costs())
    tx = fake_transactions()
    tx.loc[tx["tier_name"] == "Door", "tier_name"] = "Release 2"  # collides with "Tier 1" on E1
    transactions = build_transactions(tx, events)
    report = validate(events, transactions, build_event_tier(transactions))
    assert report["duplicates"]["event_tiers_sharing_an_ordinal"] == 1


def test_comps_are_reported_by_tier_name_so_a_bad_flag_is_visible():
    """SPEC 8.7. Stripping a comp is right; stripping a paid tier because its name holds
    the word "free" is an undercount nothing downstream catches. The labels being stripped
    are printed, so a human can see a paid tier in the list."""
    report = _report()
    assert report["comps"]["comp_rows_by_tier_name"] == {"Guestlist": 1, "Comp": 1}

    text = format_report(report)
    assert "'Guestlist'" in text
    assert "PAID tier listed here is a deleted sale" in text


def test_a_paid_tier_named_free_stays_out_of_the_comp_count():
    events = build_events(fake_costs())
    tx = fake_transactions()
    tx.loc[tx["tier_name"] == "Early Bird", "tier_name"] = "Tier 1 + Free Drink"
    transactions = build_transactions(tx, events)
    report = validate(events, transactions, build_event_tier(transactions))

    assert "Tier 1 + Free Drink" not in report["comps"]["comp_rows_by_tier_name"]
    assert report["comps"]["comp_ticket_rows"] == 2  # Guestlist and Comp, nothing else


def test_report_counts_tiers_whose_price_moved_mid_window():
    """SPEC 4.3 — price variation at a given lead time is the gold dust. It must surface
    in the report, not sit in a parquet column nobody opens."""
    events = build_events(fake_costs())
    tx = fake_transactions()
    # The last two Early Bird tickets on E1 were sold after a correction to £12.
    eb = tx.index[(tx["event_key"] == "E1") & (tx["tier_name"] == "Early Bird")][-2:]
    tx.loc[eb, "price_paid"] = 12.0
    transactions = build_transactions(tx, events)
    report = validate(events, transactions, build_event_tier(transactions))

    assert report["price_variation"]["event_tiers_with_more_than_one_price"] == 1
    assert report["price_variation"]["max_distinct_prices_in_one_tier"] == 2
    assert "price changed mid-window" in format_report(report)


def test_no_price_movement_reported_when_every_tier_held_its_price():
    report = _report()
    assert report["price_variation"] == {
        "event_tiers_with_more_than_one_price": 0,
        "max_distinct_prices_in_one_tier": 1,
    }


def test_format_report_is_readable_text():
    text = format_report(_report())
    for heading in [
        "VALIDATION REPORT",
        "Row counts",
        "Date ranges",
        "Tier ordinal coverage",
        "Comps / guestlist",
        "Join health",
        "Null rates",
    ]:
        assert heading in text
    # The report must never quote buyer values.
    assert "FAKE Buyer" not in text


def test_repulled_rows_dropped_but_real_repeat_tickets_kept():
    """One export pulled twice must not double revenue; two tickets on one order must
    not be deleted. Both look identical, so occurrence-within-file decides."""
    ticket = {
        "order_id": "FAKE-ORD-001",
        "tier_name": "Early Bird",
        "purchased_at": pd.Timestamp("2026-01-10"),
        "price_paid": 8.0,
    }
    first_pull = pd.DataFrame([ticket, ticket]).assign(source_file="a.csv")  # 2 real tickets
    second_pull = pd.DataFrame([ticket]).assign(source_file="b.csv")  # same ticket again

    deduped, dropped = ingest._drop_repulled_rows(
        pd.concat([first_pull, second_pull], ignore_index=True)
    )
    assert dropped == 1
    assert len(deduped) == 2
    assert "source_file" not in deduped.columns


def test_ingest_sniff_mode_runs_on_empty_raw_dir(tmp_path, capsys):
    assert ingest.main(["--sniff", "--raw-dir", str(tmp_path)]) == 0
    assert "No raw exports found" in capsys.readouterr().out


def test_ingest_full_run_stops_until_adapters_are_mapped(tmp_path, capsys):
    """The real exports have not landed; a full run must fail loudly, not silently."""
    (tmp_path / "fixr").mkdir()
    pd.DataFrame({"Order Ref": ["FAKE-ORD-001"]}).to_csv(
        tmp_path / "fixr" / "export.csv", index=False
    )
    assert ingest.main(["--raw-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")]) == 1
    assert "COLUMN_MAP is empty" in capsys.readouterr().err
