"""Tier normalisation, comp flagging, PII strip."""

from __future__ import annotations

import pandas as pd
import pytest

from pricing.adapters import COST_FIELDS, TRANSACTION_FIELDS
from pricing.normalize import (
    ALLOWED_COLUMNS,
    DOOR_ORDINAL,
    add_comp_flag,
    add_tier_ordinal,
    assert_no_pii,
    clean_label,
    is_comp_name,
    parse_number,
    parse_uk_datetime,
    pii_columns,
    strip_pii,
    tier_ordinal,
)
from tests.synthetic import fake_transactions


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Early Bird", 0),
        ("EARLY BIRD", 0),
        ("  early   bird  ", 0),
        ("Early-Bird", 0),
        ("earlybird", 0),
        ("EB", 0),
        ("First Release", 0),
        ("Tier 0", 0),
        ("Release 1", 0),
        ("1st Release", 0),
        ("Tier 1", 1),
        ("TIER 1", 1),
        ("Release 2", 1),
        ("Second Release", 1),
        ("2nd Release", 1),
        ("Tier 2", 2),
        ("Release 3", 2),
        ("Third Release", 2),
        ("Tier 4", 4),
        ("Door", DOOR_ORDINAL),
        ("DOOR", DOOR_ORDINAL),
        ("On The Door", DOOR_ORDINAL),
        ("OTD", DOOR_ORDINAL),
        ("Walk Up", DOOR_ORDINAL),
    ],
)
def test_tier_ordinal_known_labels(raw, expected):
    assert tier_ordinal(raw) == expected


@pytest.mark.parametrize("raw", ["", None, "Something Weird", "VIP Table", "Coach + Ticket"])
def test_tier_ordinal_unknown_returns_none(raw):
    """Unknown labels return None so validate.py can report them, not a wrong guess."""
    assert tier_ordinal(raw) is None


@pytest.mark.parametrize("raw", ["Final Release", "Last Release", "Final Tier", "Last Tier"])
def test_last_release_has_no_ordinal_and_no_sentinel(raw):
    """A "final release" label says the tier is last, not WHICH position it is: release 3
    on a three-release event, release 5 on a five-release one. It used to be parked on a
    second sentinel (98) — a guess dressed as data, and one that two spellings shared. It
    is unmapped now; window_open orders it."""
    assert tier_ordinal(raw) is None
    assert DOOR_ORDINAL - 1 not in {tier_ordinal(x) for x in [raw, "Tier 1", "Door"]}


def test_tier_ladder_orders_correctly():
    ladder = ["Early Bird", "Tier 1", "Tier 2", "Release 4", "Door"]
    ordinals = [tier_ordinal(t) for t in ladder]
    assert ordinals == sorted(ordinals)
    assert None not in ordinals


def test_clean_label():
    assert clean_label("EARLY  BIRD!!") == "early bird"


def test_add_tier_ordinal_column():
    df = pd.DataFrame({"tier_name": ["Early Bird", "Tier 2", "Mystery"]})
    out = add_tier_ordinal(df)
    assert list(out["tier_ordinal"][:2]) == [0, 2]
    assert pd.isna(out["tier_ordinal"].iloc[2])


@pytest.mark.parametrize(
    "raw",
    [
        "Comp",
        "COMPS",
        "Complimentary",
        "Guestlist",
        "Guest List",
        "GL",
        "Free Entry",
        "Artist Guest",
        "Staff",
        "Press",
    ],
)
def test_is_comp_name(raw):
    assert is_comp_name(raw)


@pytest.mark.parametrize("raw", ["Early Bird", "Tier 1", "Door", "VIP Booth"])
def test_is_not_comp_name(raw):
    assert not is_comp_name(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "Tier 1 + Free Drink",
        "Free Entry Before 11",
        "Girls Free B4 11",
        "Artist Package",
        "Plus Guest",
        "Guest DJ Ticket",
        "Free Drink Included",
    ],
)
def test_paid_tiers_containing_a_comp_word_are_not_comps(raw):
    """SPEC 8.7 says strip comps. A bare-word search for free/guest/artist stripped these
    PAID tiers out of demand instead, and an undercount is invisible downstream. The name
    test now matches the WHOLE cleaned label."""
    assert not is_comp_name(raw)


def test_a_paid_tier_named_like_a_comp_survives_the_full_flag():
    """End to end: the tier is sold for money and must stay in demand."""
    df = pd.DataFrame(
        {
            "tier_name": ["Tier 1 + Free Drink", "Girls Free B4 11", "Guestlist"],
            "price_paid": [12.0, 5.0, 0.0],
        }
    )
    assert list(add_comp_flag(df)["is_comp"]) == [False, False, True]


def test_an_unlisted_comp_label_is_still_caught_by_its_price():
    """Exact-name matching misses a comp spelling nobody listed. Free is free."""
    df = pd.DataFrame({"tier_name": ["Artist Comp x2"], "price_paid": [0.0]})
    assert add_comp_flag(df)["is_comp"].iloc[0]


def test_comp_flag_uses_name_or_zero_price():
    df = pd.DataFrame(
        {
            "tier_name": ["Early Bird", "Guestlist", "VIP", "Tier 1"],
            "price_paid": [8.0, 12.0, 0.0, 10.0],
        }
    )
    out = add_comp_flag(df)
    # Guestlist by name, VIP by zero price.
    assert list(out["is_comp"]) == [False, True, True, False]


def test_pii_columns_found_and_stripped():
    tx = fake_transactions()
    found = pii_columns(tx)
    assert set(found) == {"buyer_name", "buyer_email", "phone", "address", "postcode"}

    clean = strip_pii(tx)
    for col in ["buyer_name", "buyer_email", "phone", "address", "postcode"]:
        assert col not in clean.columns
    # No fake buyer values survive anywhere.
    assert not clean.astype(str).apply(lambda s: s.str.contains("FAKE Buyer")).any().any()


def test_business_name_columns_are_not_treated_as_pii():
    df = pd.DataFrame(
        columns=["event_name", "tier_name", "venue_name", "brand_name", "artist_name", "city"]
    )
    assert pii_columns(df) == []


@pytest.mark.parametrize(
    "column",
    [
        "Booking Name",
        "Ticket Holder",
        "Username",
        "User Name",
        "Recipient Name",
        "Lead Booker",
        "Order Name",
        "Payer",
        "Contact Number",
        "Emergency Contact",
        "Student ID",
        "Nickname",
        "Instagram Handle",
    ],
)
def test_real_export_pii_column_names_are_caught(column):
    """Names real ticketing exports use. --sniff must warn about all of them."""
    assert pii_columns(pd.DataFrame(columns=["order_id", column])) == [column]


@pytest.mark.parametrize(
    "column",
    [
        "Mobile_Phone",  # \b patterns only fire once "_" becomes a space
        "mobile phone",
        "city_of_buyer",  # the one pattern that used to be written with underscores
        "City Of Buyer",
        "customerEmail",  # camelCase lowercases to one word, no separator at all
        "BUYER-NAME",
    ],
)
def test_pii_names_are_caught_in_one_pass_whatever_the_separator(column):
    """The column name is normalised to spaced words and searched ONCE, not searched in
    both the spaced and the underscored form. Every separator style must still be
    caught, in particular the two shapes that motivated the double search."""
    assert pii_columns(pd.DataFrame(columns=["order_id", column])) == [column]


def test_assert_no_pii_raises():
    with pytest.raises(ValueError, match="buyer PII"):
        assert_no_pii(pd.DataFrame(columns=["order_id", "customer_email"]), where="test")


def test_assert_no_pii_is_fail_closed():
    """The last line of defence is an allowlist, not a pattern hunt: a column nobody
    predicted is refused because it is unknown, not because it looks person-shaped."""
    with pytest.raises(ValueError, match="allowlist"):
        assert_no_pii(pd.DataFrame(columns=["order_id", "who_bought_it"]), where="test")


def test_allowlist_covers_every_canonical_field():
    """ALLOWED_COLUMNS must not drift from the adapters' canonical schemas."""
    assert set(TRANSACTION_FIELDS) <= ALLOWED_COLUMNS
    assert set(COST_FIELDS) <= ALLOWED_COLUMNS


def test_assert_no_pii_passes_on_clean_frame():
    clean = strip_pii(fake_transactions())
    assert_no_pii(clean, where="test")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("03/04/2024 21:30", pd.Timestamp("2024-04-03 21:30")),  # UK: 3 April
        ("01/02/2024", pd.Timestamp("2024-02-01")),  # UK: 1 February
        ("2024-03-04", pd.Timestamp("2024-03-04")),  # ISO: 4 March, never flipped
        ("15/04/2024", pd.Timestamp("2024-04-15")),
        ("not a date", pd.NaT),
    ],
)
def test_parse_uk_datetime(raw, expected):
    parsed = parse_uk_datetime(pd.Series([raw])).iloc[0]
    assert parsed is pd.NaT if expected is pd.NaT else parsed == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("£10.00", 10.0), ("1,250", 1250.0), ("  12.5 ", 12.5), (8.0, 8.0), ("free", None)],
)
def test_parse_number_handles_money_formatting(raw, expected):
    parsed = parse_number(pd.Series([raw])).iloc[0]
    assert pd.isna(parsed) if expected is None else parsed == expected


def test_unparseable_price_is_not_a_comp():
    df = pd.DataFrame({"tier_name": ["Tier 1"], "price_paid": ["ten pounds"]})
    assert not add_comp_flag(df)["is_comp"].iloc[0]
