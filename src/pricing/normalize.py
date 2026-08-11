"""Normalisation: date/number parsing, tier names -> ordinal, comp flags, buyer-PII strip.

Four jobs, four sections. All plain functions on plain DataFrames.

SPEC.md 8.8  "Early Bird", "EARLY BIRD", "Tier 0", "First Release" are the same thing.
SPEC.md 8.7  Comps and guestlist are not demand. Flag them; never count them as sales.
DECISIONS.md Buyer PII is stripped at ingest. Non-negotiable, and it happens here.
"""

from __future__ import annotations

import re

import pandas as pd

# --------------------------------------------------------------------------------------
# 0. Parsing raw cells: UK dates and money
# --------------------------------------------------------------------------------------


def _to_datetime(values: pd.Series, **kwargs) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", **kwargs)


def parse_uk_datetime(values: pd.Series) -> pd.Series:
    """Parse a date column to tz-naive London wall-clock time. Used for EVERY timestamp.

    Two passes, because one rule cannot cover both shapes we will meet:

    1. ISO first ("2026-03-01"). ISO is unambiguous and must NEVER be read day-first —
       pandas with dayfirst=True turns 2026-03-01 into 3 January.
    2. Whatever ISO could not read is a UK export, so `dayfirst=True`: "03/04/2024" is
       3 April, not 4 March. `format="mixed"` parses each cell on its own terms, so one
       column holding both "01/02/2024" and "03/04/2024 21:30" survives.

    Getting this wrong silently corrupts event_date, purchased_at, lead_time_days, the
    whole sales curve and every calendar feature in preregistration.md, which is why it
    lives in one function that every call site uses. validate.py reports how many rows are
    ambiguous (day <= 12) so the assumption stays visible.

    Timezones: a column carrying UTC offsets — or a mix of GMT and BST, which otherwise
    raises "Mixed timezones detected" mid-ingest — is converted to London time and then
    made naive. Local wall-clock is what "Thursday event" and "bought 14 days out" mean.
    A column with no offsets is already local wall-clock and is left exactly as written.
    """
    try:
        parsed = _to_datetime(values, format="ISO8601").fillna(
            _to_datetime(values, dayfirst=True, format="mixed")
        )
    except ValueError:  # mixed GMT/BST offsets in one column
        parsed = _to_datetime(values, format="ISO8601", utc=True).fillna(
            _to_datetime(values, dayfirst=True, format="mixed", utc=True)
        )
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Europe/London").dt.tz_localize(None)
    return parsed


def parse_number(values: pd.Series) -> pd.Series:
    """Numbers out of a text column: strips currency symbols and thousands separators.

    Excel and ticketing exports write money as "£10.00" and "1,250". Feeding those
    straight to pd.to_numeric gives NaN, and a NaN price would otherwise read as a free
    ticket (SPEC 8.7) and delete the sale. Unparseable values stay NaN and validate.py
    reports them as a hard failure — never as zero.
    """
    if not pd.api.types.is_numeric_dtype(values):
        values = values.astype(str).str.replace(r"[£$€,\s]", "", regex=True)
    return pd.to_numeric(values, errors="coerce")


# --------------------------------------------------------------------------------------
# 1. Tier names -> ordinal
# --------------------------------------------------------------------------------------
#
# The ordinal is the release's position in the ladder: Early Bird = 0, then 1, 2, 3...
# and Door last of all.
#
# Two numbering conventions exist in the wild and they are off by one:
#   "Tier N"    is the platform's own 0-or-1-based label -> use N as-is.
#   "Release N" is human 1-based ("first release") -> use N - 1.
# That is why "Tier 1" and "Release 2" both land on ordinal 1. Early Bird, Tier 0 and
# First Release / Release 1 all land on ordinal 0.
#
# Door is parked at a large sentinel so it always sorts last regardless of how many
# numbered releases an event had.
DOOR_ORDINAL = 99

# Names that state their own position in the ladder. Exact match on the cleaned label, so
# "Early Bird Plus One" is NOT quietly read as a plain Early Bird -- unrecognised names
# are reported by validate.py rather than guessed at.
FIXED_TIER_ORDINALS: dict[str, int] = {
    "early bird": 0,
    "earlybird": 0,
    "eb": 0,
    "first release": 0,
    "advance": 0,
    "advanced": 0,
    "door": DOOR_ORDINAL,
    "otd": DOOR_ORDINAL,
    "on the door": DOOR_ORDINAL,
    "cash on door": DOOR_ORDINAL,
    "door entry": DOOR_ORDINAL,
    "walk up": DOOR_ORDINAL,
    "walkup": DOOR_ORDINAL,
    "walk in": DOOR_ORDINAL,
    "walkin": DOOR_ORDINAL,
}

# "Final Release" / "Last Release" are deliberately absent. The name says the tier is last
# but not WHICH position that is -- it is release 3 on a three-release event and release 5
# on a five-release one. They used to be parked on a second sentinel (98), which was a
# guess dressed as data and collided whenever an event used both spellings. They now come
# back unmapped, and their place in the ladder is settled by their first-sale timestamp
# (window_open in event_tier), which is the thing that actually orders releases.

# Word-number releases, spelled out. Kept as a flat lookup because that is the shortest
# thing that works; extend it when a real export shows something new.
_WORD_NUMBERS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
}

_ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")


def clean_label(name: object) -> str:
    """Lowercase, strip punctuation, collapse whitespace. 'EARLY  BIRD!' -> 'early bird'."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tier_ordinal(name: object) -> int | None:
    """Return the release ordinal for a raw tier name, or None if unrecognised.

    Three rules, in order: the fixed-name lookup, "Tier N", "Release N". Nothing else.

    None is a deliberate signal, not a failure: validate.py reports the unmapped labels
    so a human decides where they belong, rather than the code guessing.
    """
    label = clean_label(name)
    if not label:
        return None

    if label in FIXED_TIER_ORDINALS:
        return FIXED_TIER_ORDINALS[label]

    # "Second Release" and "2nd Release" are spellings of "Release 2". Rewrite, then let
    # the "release N" rule below do the arithmetic once.
    words = label.split()
    if len(words) == 2 and words[1] == "release":
        head = words[0]
        if head in _WORD_NUMBERS:
            label = f"release {_WORD_NUMBERS[head]}"
        elif head.endswith(_ORDINAL_SUFFIXES) and head[:-2].isdigit():
            label = f"release {head[:-2]}"

    # "Tier N" is the platform's own label: N as written.
    match = re.match(r"^tier (\d+)$", label)
    if match:
        return int(match.group(1))

    # "Release N" is human 1-based, so N - 1. "Tier 1" and "Release 2" both land on 1.
    match = re.match(r"^(?:release|rel) (\d+)$", label)
    if match:
        return max(int(match.group(1)) - 1, 0)

    return None


def add_tier_ordinal(df: pd.DataFrame, tier_col: str = "tier_name") -> pd.DataFrame:
    """Add a `tier_ordinal` column derived from `tier_col`."""
    out = df.copy()
    out["tier_ordinal"] = out[tier_col].map(tier_ordinal).astype("Int64")
    return out


# --------------------------------------------------------------------------------------
# 2. Comps and guestlist (SPEC.md 8.7)
# --------------------------------------------------------------------------------------

# A comp is recognised by its WHOLE name, from this list -- never by a word inside a
# longer name. A bare-word search for free/guest/artist marked these PAID tiers as comps
# and deleted them from demand: "Tier 1 + Free Drink", "Free Entry Before 11",
# "Girls Free B4 11", "Artist Package", "Plus Guest". SPEC 8.7 says strip comps, not
# sales, and an undercount here is invisible downstream.
#
# The cost of exact matching is that a comp called something unlisted ("Artist Comp x2")
# is missed by NAME. It is still caught by the price test below, because a comp is free;
# and validate.py prints comp counts per tier_name so both directions of error are
# visible in one place before anything is modelled.
COMP_LABELS: frozenset[str] = frozenset(
    {
        "comp",
        "comps",
        "comp ticket",
        "comp tickets",
        "complimentary",
        "guestlist",
        "guest list",
        "guest",
        "guests",
        "gl",
        "free",
        "free entry",
        "free ticket",
        "artist guest",
        "artist guestlist",
        "artist comp",
        "staff",
        "crew",
        "industry",
        "press",
        "photo pass",
        "photopass",
        "vip comp",
    }
)


def is_comp_name(name: object) -> bool:
    """True if the tier name IS a comp / guestlist label, matched whole."""
    return clean_label(name) in COMP_LABELS


def add_comp_flag(
    df: pd.DataFrame,
    tier_col: str = "tier_name",
    price_col: str = "price_paid",
) -> pd.DataFrame:
    """Add `is_comp`: True when the whole tier name is a comp label OR the buyer paid 0.

    Both tests are needed. A "VIP" tier issued at zero price is a comp whatever it is
    called, and a comp label the list has not seen is caught by the price. The name test
    matches the FULL cleaned label (see COMP_LABELS), so a paid tier that merely contains
    the word "free" or "guest" stays in demand where it belongs.
    """
    out = df.copy()
    by_name = out[tier_col].map(is_comp_name)
    # An unparseable price is NOT free. NaN <= 0 is False, deliberately: a "£10.00" that
    # failed to parse must show up as a validation failure, not as a comp that quietly
    # drops the sale out of units_sold.
    by_price = parse_number(out[price_col]) <= 0
    out["is_comp"] = (by_name | by_price).astype(bool)
    return out


# --------------------------------------------------------------------------------------
# 3. Buyer PII strip (DECISIONS.md — non-negotiable)
# --------------------------------------------------------------------------------------
#
# Buyer names, emails, phones and addresses are dropped here, at ingest, and are never
# written to any derived artifact. Two mechanisms with different jobs:
#
#   pii_columns()   pattern hunt. Best-effort, used to DROP raw columns and to warn in
#                   --sniff. A pattern list can always be beaten by a column name nobody
#                   predicted, so it is never the last line of defence.
#   assert_no_pii() fail-CLOSED allowlist. Refuses to write any column that is not a
#                   known canonical or derived field, whatever it is called. An
#                   unrecognised column is treated as PII until a human adds it below.
#
# _NEVER_DROP keeps business columns whose names look person-ish (`event_name`,
# `venue_name`) out of the pattern hunt's jaws.

# One concern per line, matched against the column name lowercased with every separator
# turned into a single space ("Buyer_Email" and "buyerEmail" both read as one string).
# This list is NOT redundant with the allowlist and must not be deleted: the allowlist
# only guards derived tables, whose columns are known names, whereas these run against
# RAW export columns, whose names are arbitrary. It is what drops them before anything
# else sees the frame, and what --sniff warns on.
_PII_PATTERNS: list[str] = [
    r"e-? ?mail",
    r"\bphone\b|\bmobile\b|\btelephone\b|\btel\b|\bmsisdn\b",
    r"address|postcode|post ?code|\bzip\b|city of buyer",
    r"\bdob\b|date ?of ?birth|birthday",
    r"\bip\b|ip ?address",
    r"card|last ?4|iban|sort ?code|account ?number",
    # Person-name columns: an explicit buyer-ish prefix, or a bare `name`. The prefixes
    # already covered by the bare-word line below are not repeated here.
    (
        r"(first|last|full|sur|fore|given|customer|buyer|purchaser|member|student|"
        r"booking|order|recipient|lead) ?name"
    ),
    r"^name$",
    # A column named for the person alone, with no "name" in it at all.
    r"^(customer|buyer|attendee|purchaser|booker|contact)$",
    r"ticket ?holder|\bholder\b|\bbooker\b|\bpayer\b|\bpayee\b|\battendee\b|\bguest\b",
    r"\bcontact\b",  # contact number, emergency contact, contact details
    # Names real ticketing exports actually use for the person on the ticket.
    r"user ?name|\bnickname\b|\bhandle\b|screen ?name|\bdisplay ?name\b",
    r"student ?(id|number)|\bmatric\b",
    r"instagram|twitter|snapchat|tiktok|facebook|social",
]

_NEVER_DROP: set[str] = {
    "event_name",
    "tier_name",
    "ticket_type_name",
    "venue_name",
    "brand_name",
    "artist_name",
    "promoter_name",
    "platform_name",
}

_PII_RE = re.compile("|".join(_PII_PATTERNS))


def pii_columns(df: pd.DataFrame) -> list[str]:
    """List the columns that look like buyer PII.

    The column name is normalised to lowercase words separated by single spaces, then
    searched ONCE. It used to be searched twice, in both the spaced and the underscored
    form, because one pattern (`city_of_buyer`) was written with underscores while the
    `\\b`-anchored ones only fire on the spaced form -- `_` is a word character, so
    `\\bphone\\b` never matched `mobile_phone`. Spaces everywhere, one pass, no exceptions.
    """
    found = []
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()
        if key.replace(" ", "_") in _NEVER_DROP:
            continue
        if _PII_RE.search(key):
            found.append(col)
    return found


def strip_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every buyer-PII column. Call this the moment raw data enters memory."""
    return df.drop(columns=pii_columns(df))


# The complete list of columns allowed to exist in a derived table. Anything else is
# refused at write time. Keep in sync with adapters.TRANSACTION_FIELDS / COST_FIELDS —
# tests/test_normalize.py asserts they are all present here.
ALLOWED_COLUMNS: set[str] = {
    # transactions, straight from the platform export
    "order_id",
    "event_key",
    "tier_name",
    "price_paid",
    "booking_fee",
    "purchased_at",
    "quantity",
    "promo_code",
    # events, from the brand finance workbooks
    "event_name",
    "event_date",
    "venue",
    "city",
    "brand",
    "capacity",
    "artist_guarantee",
    "deal_structure",
    "venue_cost",
    "security_cost",
    "production_cost",
    "staffing_cost",
    "marketing_spend",
    "cancelled",
    # derived here
    "platform",
    "event_id",
    "tier_ordinal",
    "is_comp",
    "buyer_price",
    "fee_treatment",  # travels with buyer_price: "UNKNOWN" means the number is a guess
    "lead_time_days",
    "fixed_cost_stack",
    # event_tier panel
    "price",
    "face_price",
    "n_distinct_prices",
    "units_sold",
    "units_comp",
    "n_orders",
    "window_open",
    "window_close",
    "lead_time_open_days",
}


def unexpected_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are not on the canonical allowlist. Treated as PII until proven not."""
    return [c for c in df.columns if str(c) not in ALLOWED_COLUMNS]


def assert_no_pii(df: pd.DataFrame, where: str = "dataframe") -> None:
    """Fail-closed guard on every parquet write.

    Refuses any column that is not on ALLOWED_COLUMNS, rather than hunting for column
    names that look like buyer PII. A pattern hunt can be beaten by a name nobody thought
    of ("Lead Booker", "Instagram Handle"); an allowlist cannot. If a legitimate new field
    is refused, add it to ALLOWED_COLUMNS on purpose.
    """
    unexpected = unexpected_columns(df)
    if unexpected:
        raise ValueError(
            f"refusing to write {where}: columns not on the canonical allowlist, so they "
            f"are treated as possible buyer PII: {sorted(unexpected)}. "
            f"If they are legitimate, add them to normalize.ALLOWED_COLUMNS."
        )
