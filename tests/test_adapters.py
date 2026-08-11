"""Adapter stubs and the --sniff schema summary."""

from __future__ import annotations

import pandas as pd
import pytest

from pricing import adapters


def test_unmapped_adapters_raise_loudly():
    """Until the real exports are mapped, every adapter must refuse to guess."""
    raw = pd.DataFrame({"whatever": [1]})
    for adapt in (
        adapters.adapt_fixr,
        adapters.adapt_tbc,
        adapters.adapt_youni,
        adapters.adapt_costs,
    ):
        with pytest.raises(NotImplementedError, match="COLUMN_MAP"):
            adapt(raw)


def test_apply_map_produces_canonical_columns_and_strips_pii():
    """Simulates a filled-in COLUMN_MAP without touching the real (empty) one."""
    raw = pd.DataFrame(
        {
            "Order Ref": ["FAKE-ORD-001"],
            "Event": ["FAKE Event One"],
            "Ticket Type": ["Early Bird"],
            "Price": [8.0],
            "Purchased": ["2026-01-10"],
            "Customer Name": ["FAKE Buyer 001"],
        }
    )
    column_map = {
        "Order Ref": "order_id",
        "Event": "event_key",
        "Ticket Type": "tier_name",
        "Price": "price_paid",
        "Purchased": "purchased_at",
        "Customer Name": "buyer_name",
    }
    out = adapters._apply_map(raw, column_map, adapters.TRANSACTION_REQUIRED, "fixr")
    assert set(out.columns) == set(adapters.TRANSACTION_REQUIRED)
    assert "buyer_name" not in out.columns


def test_apply_map_rejects_two_raw_columns_mapped_to_one_field():
    """Renaming both "Price" and "Total Paid" to price_paid keeps whichever lands last and
    drops the other without a word. The map is hand-typed from a --sniff printout, so this
    is a plausible typo and it silently changes every price downstream."""
    raw = pd.DataFrame(
        {
            "Order Ref": ["FAKE-ORD-001"],
            "Event": ["FAKE Event One"],
            "Ticket Type": ["Early Bird"],
            "Price": [8.0],
            "Total Paid": [9.0],
            "Purchased": ["2026-01-10"],
        }
    )
    column_map = {
        "Order Ref": "order_id",
        "Event": "event_key",
        "Ticket Type": "tier_name",
        "Price": "price_paid",
        "Total Paid": "price_paid",
        "Purchased": "purchased_at",
    }
    with pytest.raises(ValueError, match="more than one raw column to \\['price_paid'\\]"):
        adapters._apply_map(raw, column_map, adapters.TRANSACTION_REQUIRED, "fixr")


def test_apply_map_rejects_a_stale_map():
    with pytest.raises(ValueError, match="absent from the file"):
        adapters._apply_map(pd.DataFrame({"A": [1]}), {"B": "order_id"}, ["order_id"], "fixr")


def test_fee_treatment_is_unknown_until_real_data_confirms_it():
    """SPEC 8.6 — no platform may claim a fee treatment before it is checked."""
    assert set(adapters.FEE_TREATMENT) == {"fixr", "tbc", "youni"}
    assert all(v == "UNKNOWN" for v in adapters.FEE_TREATMENT.values())


def _write_fake_export(tmp_path):
    fixr_dir = tmp_path / "fixr"
    fixr_dir.mkdir()
    pd.DataFrame(
        {
            "Order Ref": ["FAKE-ORD-001", "FAKE-ORD-002"],
            "Ticket Type": ["Early Bird", "Door"],
            "Customer Email": ["fake001@example.invalid", "fake002@example.invalid"],
            "Date of Birth": ["1998-04-02", "2003-11-30"],
            "Purchased": ["2026-01-10", "2026-03-01"],
            "Price": [8.0, 12.0],
            "Artist Guarantee": [1500, 7250],
            "Capacity": ["1,200", "500"],  # a number written as text
        }
    ).to_csv(fixr_dir / "BrandName_Halloween_2023_export.csv", index=False)
    return fixr_dir / "BrandName_Halloween_2023_export.csv"


def test_sniff_reports_shape_only(tmp_path):
    _write_fake_export(tmp_path)
    summaries = adapters.sniff_raw(tmp_path)

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["platform"] == "fixr"
    assert summary["rows"] == 2
    assert [c["name"] for c in summary["columns"]] == [
        "Order Ref",
        "Ticket Type",
        "Customer Email",
        "Date of Birth",
        "Purchased",
        "Price",
        "Artist Guarantee",
        "Capacity",
    ]
    assert summary["suspected_pii_columns"] == ["Customer Email", "Date of Birth"]

    purchased = next(c for c in summary["columns"] if c["name"] == "Purchased")
    assert purchased["date_range"] == ("2026-01-10", "2026-03-01")

    text = adapters.format_sniff(summaries)
    assert "Ticket Type" in text  # column names appear
    for value in ["FAKE-ORD-001", "fake001@example.invalid", "Early Bird"]:
        assert value not in text  # cell values never do


def test_sniff_never_prints_a_number_or_a_pii_date_range(tmp_path):
    """pd.to_datetime reads a number as epoch nanoseconds, so money columns used to print
    their real min and max in a summary that claims to hold no values."""
    _write_fake_export(tmp_path)
    summaries = adapters.sniff_raw(tmp_path)
    columns = {c["name"]: c for c in summaries[0]["columns"]}

    assert "date_range" not in columns["Artist Guarantee"]  # int64: skipped outright
    assert "date_range" not in columns["Price"]  # float64: skipped outright
    assert "date_range" not in columns["Capacity"]  # "1,200": money as text, still skipped
    assert "date_range" not in columns["Date of Birth"]  # PII: never gets a range

    text = adapters.format_sniff(summaries)
    for value in ["1500", "7250", "8.0", "12.0", "1,200", "500", "1998", "2003"]:
        assert value not in text


def test_sniff_does_not_print_filenames(tmp_path):
    """Filenames carry real brand and event names (DECISIONS.md)."""
    path = _write_fake_export(tmp_path)
    text = adapters.format_sniff(adapters.sniff_raw(tmp_path))
    assert path.name not in text
    assert "BrandName" not in text
    assert "fixr/file 1 of 1" in text


def test_sniff_reports_only_the_error_type(tmp_path):
    """A pandas parse error can quote the file's own content back at you."""
    fixr_dir = tmp_path / "fixr"
    fixr_dir.mkdir()
    (fixr_dir / "broken.xlsx").write_text("FAKE Buyer 001 not really a workbook")
    summary = adapters.sniff_raw(tmp_path)[0]
    assert set(summary) == {"folder", "error", "label"}
    assert "FAKE Buyer 001" not in adapters.format_sniff([summary])


def test_sniff_reads_excel(tmp_path):
    costs_dir = tmp_path / "costs"
    costs_dir.mkdir()
    pd.DataFrame({"Event": ["FAKE Event One"], "Guarantee": [1500.0]}).to_excel(
        costs_dir / "brand_a_master.xlsx", index=False
    )
    summaries = adapters.sniff_raw(tmp_path)
    assert summaries[0]["platform"] == "costs"
    assert [c["name"] for c in summaries[0]["columns"]] == ["Event", "Guarantee"]
    text = adapters.format_sniff(summaries)
    assert "FAKE Event One" not in text
    assert "1500" not in text


def test_sniff_on_empty_directory_is_a_friendly_message(tmp_path):
    assert "No raw exports found" in adapters.format_sniff(adapters.sniff_raw(tmp_path))


def test_adapter_for_routes_by_folder_then_filename(tmp_path):
    assert adapters.adapter_for(tmp_path / "fixr" / "a.csv")[0] == "fixr"
    assert adapters.adapter_for(tmp_path / "costs" / "a.xlsx")[0] == "costs"
    assert adapters.adapter_for(tmp_path / "other" / "youni_export.csv")[0] == "youni"
    assert adapters.adapter_for(tmp_path / "other" / "tbc_export.csv")[0] == "tbc"
    assert adapters.adapter_for(tmp_path / "other" / "mystery.csv") == ("unknown", None)
