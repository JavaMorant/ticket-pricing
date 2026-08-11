"""Build the three derived tables (SPEC.md 3.3) and write them to data/derived/.

    events        one row per event: metadata + the reconstructed cost stack
    transactions  one row per ticket, joined to events
    event_tier    one row per (event x tier): price, units sold, window open/close
                  ^ this is the panel Phase 4 runs on

Input is adapter output (canonical column names), not raw files. These functions do not
know what a Fixr column is called; adapters.py does.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pricing.adapters import FEE_TREATMENT
from pricing.normalize import (
    add_comp_flag,
    add_tier_ordinal,
    assert_no_pii,
    parse_number,
    parse_uk_datetime,
    strip_pii,
)

# Anchored to the repo, not the shell's working directory: running ingest from another
# folder must not write real event and finance tables somewhere outside the gitignore.
REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = REPO_ROOT / "data" / "derived"

FIXED_COST_COLUMNS = [
    "artist_guarantee",
    "venue_cost",
    "security_cost",
    "production_cost",
    "staffing_cost",
    "marketing_spend",
]


# Values in a workbook's `cancelled` column that mean "yes, this one did not happen".
# Everything else is False. The naive `.astype(bool)` this replaces marked "N" as True,
# which would have inverted SPEC 8.4 the moment the column was populated.
CANCELLED_TRUE = {"y", "yes", "true", "1", "1.0", "cancelled", "cancelled event", "pulled"}


def build_events(costs: pd.DataFrame) -> pd.DataFrame:
    """One row per event from the brand finance workbooks.

    Adds `fixed_cost_stack` — the denominator of break-even attendance (SPEC.md 6.3).
    Missing cost components are treated as 0 here and reported as null rates by
    validate.py, so a hole in the workbook shows up as a data problem rather than
    silently deleting the event.
    """
    out = strip_pii(costs).copy()
    out["event_date"] = parse_uk_datetime(out["event_date"])

    for col in FIXED_COST_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = parse_number(out[col])
    if "capacity" in out.columns:
        out["capacity"] = parse_number(out["capacity"])

    out["fixed_cost_stack"] = out[FIXED_COST_COLUMNS].fillna(0).sum(axis=1)
    if "cancelled" not in out.columns:
        out["cancelled"] = False
    out["cancelled"] = out["cancelled"].astype(str).str.strip().str.lower().isin(CANCELLED_TRUE)

    out["event_id"] = out["event_key"].astype(str).str.strip()

    # A duplicate event row fans out the transaction join and doubles units_sold silently.
    # The finance workbooks are the likely source: one sheet per year, a totals row, or an
    # event entered twice. Stop here rather than publish a table with doubled sales.
    repeated = sorted(out.loc[out["event_id"].duplicated(), "event_id"].unique())
    if repeated:
        raise ValueError(
            f"events: event_id must be unique, but {len(repeated)} repeat: {repeated[:10]}. "
            f"Check the costs workbook for totals rows, per-year sheets, or a re-entered "
            f"event, and filter them in adapt_costs."
        )

    return out.sort_values("event_date").reset_index(drop=True)


def build_transactions(tx: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per ticket, normalised and joined to events.

    Adds: tier_ordinal, is_comp, fee_treatment, buyer_price, lead_time_days.
    """
    out = strip_pii(tx).copy()
    out["event_id"] = out["event_key"].astype(str).str.strip()
    out["purchased_at"] = parse_uk_datetime(out["purchased_at"])
    out["price_paid"] = parse_number(out["price_paid"])

    if "booking_fee" not in out.columns:
        out["booking_fee"] = 0.0
    out["booking_fee"] = parse_number(out["booking_fee"]).fillna(0.0)

    if "quantity" not in out.columns:
        out["quantity"] = 1
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").fillna(1).astype(int)

    out = add_tier_ordinal(out)
    out = add_comp_flag(out)
    # fee_treatment is written to the table alongside buyer_price, not just printed in the
    # report, so the provisional-ness travels with the number into every downstream join.
    out["fee_treatment"] = _fee_treatment(out)
    out["buyer_price"] = out["price_paid"].where(
        out["fee_treatment"] == "inclusive", out["price_paid"] + out["booking_fee"]
    )

    meta = [c for c in ["event_date", "brand", "venue", "city", "capacity"] if c in events.columns]
    out = out.merge(events[["event_id", *meta]], on="event_id", how="left")

    if "event_date" in out.columns:
        out["lead_time_days"] = _lead_time_days(out["event_date"], out["purchased_at"])

    return out.sort_values(["event_id", "purchased_at"]).reset_index(drop=True)


def _fee_treatment(tx: pd.DataFrame) -> pd.Series:
    """Per-row fee treatment (SPEC.md 8.6), from the platform the ticket was sold on.

    "inclusive"     -> price_paid already contains the fee
    "face_plus_fee" -> the buyer paid price_paid + booking_fee
    "UNKNOWN"       -> buyer_price is computed as face_plus_fee, because that is the
                       common platform default, but it is a GUESS. It is carried as a
                       column on transactions and on event_tier so nothing downstream can
                       read a price without also seeing that it is unconfirmed. Every
                       elasticity number is provisional until FEE_TREATMENT is settled.
    """
    platform = tx["platform"] if "platform" in tx.columns else pd.Series("fixr", index=tx.index)
    return platform.map(lambda p: FEE_TREATMENT.get(str(p), "UNKNOWN"))


def _lead_time_days(event_date: pd.Series, stamp: pd.Series) -> pd.Series:
    """Whole calendar days from a timestamp's DATE to the event's date.

    Both sides are floored to midnight first. event_date is a date, so subtracting a raw
    timestamp gave -1 for a door sale at 23:30 on the night of the event: bought 30
    minutes before midnight, "one day after" the event that was still happening. A ticket
    bought on the day is 0 days out, and lead time is never negative.
    """
    return (event_date.dt.normalize() - stamp.dt.normalize()).dt.days


def build_event_tier(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per (event x tier). The Phase 4 panel.

    Grouped by (event_id, TIER NAME), not by tier ordinal. The tier is the thing that has
    a price and a sales window; the ordinal is a label derived from its name and it is not
    unique within an event — "Tier 1" and "Release 2" both map to 1, and every tier the
    normaliser does not recognise maps to NA. Grouping on the ordinal collapses tiers at
    different prices into one row and destroys exactly the within-event price variation
    SPEC 4.2 identifies beta off. `tier_ordinal` is carried through as the group's min,
    and validate.py reports any (event_id, tier_ordinal) pair that ends up shared.

    Comps are excluded from `units_sold` and counted separately in `units_comp`
    (SPEC.md 8.7 — comps are not demand).

    `price` is the FIRST buyer price observed in the tier — the price it opened at — not
    the modal realised price. The mode is an outcome of how the tier sold, so it both
    leaks the end of the window into a number used to explain it (SPEC 8.2) and averages
    away mid-window corrections, which are exactly the price-at-a-given-lead-time
    variation SPEC 4.3 says to go looking for. `n_distinct_prices` makes any such change
    visible instead: > 1 means the tier's price moved while it was on sale.

    "first" skips nulls, so a tier whose earliest row has an unparsed price takes the next
    price that parsed — never NaN. Those rows are a hard failure in validate.py anyway.
    """
    keys = ["event_id", "tier_name"]
    # Sorted so "first" means the first ticket SOLD, not the first row in the file.
    paid = transactions[~transactions["is_comp"]].sort_values("purchased_at")

    panel = (
        paid.groupby(keys, dropna=False)
        .agg(
            tier_ordinal=("tier_ordinal", "min"),
            price=("buyer_price", "first"),
            face_price=("price_paid", "first"),
            n_distinct_prices=("buyer_price", "nunique"),
            units_sold=("quantity", "sum"),
            n_orders=("order_id", "nunique"),
            window_open=("purchased_at", "min"),
            window_close=("purchased_at", "max"),
        )
        .reset_index()
    )

    comps = (
        transactions[transactions["is_comp"]]
        .groupby(keys, dropna=False)
        .agg(units_comp=("quantity", "sum"), comp_ordinal=("tier_ordinal", "min"))
        .reset_index()
    )
    panel = panel.merge(comps, on=keys, how="outer")
    # A comp-only tier has no paid rows, so its ordinal comes from the comp side.
    panel["tier_ordinal"] = panel["tier_ordinal"].fillna(panel["comp_ordinal"])
    panel = panel.drop(columns=["comp_ordinal"])
    panel["units_comp"] = panel["units_comp"].fillna(0).astype(int)
    panel["units_sold"] = panel["units_sold"].fillna(0).astype(int)
    panel["n_distinct_prices"] = panel["n_distinct_prices"].fillna(0).astype(int)

    meta = [
        c
        for c in ["event_date", "brand", "platform", "fee_treatment", "capacity"]
        if c in transactions.columns
    ]
    if meta:
        first = transactions.groupby("event_id")[meta].first().reset_index()
        panel = panel.merge(first, on="event_id", how="left")

    if "event_date" in panel.columns:
        # Same date-floored rule as transactions.lead_time_days.
        panel["lead_time_open_days"] = _lead_time_days(panel["event_date"], panel["window_open"])

    return panel.sort_values(["event_id", "tier_ordinal", "tier_name"]).reset_index(drop=True)


def write_table(df: pd.DataFrame, name: str, out_dir: Path = DERIVED_DIR) -> Path:
    """Write a derived table to parquet. Refuses to write a column not on the allowlist."""
    assert_no_pii(df, where=name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path.resolve()
