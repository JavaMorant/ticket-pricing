"""Validation report over the three derived tables.

Answers, in one printout: how much data is there, does it cover the years it should,
what is missing, what is duplicated, which tier names did the normaliser fail to
recognise, and which per-platform unknowns are still unresolved.

The report contains no buyer data — counts, rates, dates and tier labels only. It DOES
contain real tier labels (you need them to write the normalisation rules) and real
absolute event counts, so it is LOCAL USE ONLY: `run_ingest` writes it to
data/derived/validation_report.txt, which is gitignored. Do not paste it anywhere public.
"""

from __future__ import annotations

import pandas as pd

from pricing.adapters import FEE_TREATMENT
from pricing.normalize import unexpected_columns


def _null_rates(df: pd.DataFrame) -> dict[str, float]:
    """Null rate per column, worst first, non-zero only."""
    rates = df.isna().mean().sort_values(ascending=False)
    return {str(k): round(float(v), 4) for k, v in rates.items() if v > 0}


def _date_span(series: pd.Series) -> tuple[str | None, str | None]:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().sum() == 0:
        return None, None
    return str(parsed.min().date()), str(parsed.max().date())


def _ambiguous_day_share(series: pd.Series) -> float:
    """Share of dates whose day-of-month is <= 12, i.e. dd/mm and mm/dd both look valid.

    Every timestamp is parsed dayfirst (UK exports are dd/mm/yyyy). This number makes the
    assumption visible: it is the fraction of rows that would move to a different date if
    the assumption is wrong. Spot-check a few of those against the platform before
    trusting a sales curve.
    """
    days = pd.to_datetime(series, errors="coerce").dt.day.dropna()
    return float((days <= 12).mean()) if len(days) else 0.0


def validate(
    events: pd.DataFrame,
    transactions: pd.DataFrame,
    event_tier: pd.DataFrame,
) -> dict:
    """Run every check and return the results as a plain dict."""
    report: dict = {}

    report["row_counts"] = {
        "events": len(events),
        "transactions": len(transactions),
        "event_tier": len(event_tier),
    }

    report["date_ranges"] = {
        "events.event_date": _date_span(events["event_date"]),
        "transactions.purchased_at": _date_span(transactions["purchased_at"]),
    }

    report["date_parsing"] = {
        "assumed_format": "dd/mm/yyyy (dayfirst=True), London wall-clock",
        "events.event_date_ambiguous_share": _ambiguous_day_share(events["event_date"]),
        "transactions.purchased_at_ambiguous_share": _ambiguous_day_share(
            transactions["purchased_at"]
        ),
    }

    report["null_rates"] = {
        "events": _null_rates(events),
        "transactions": _null_rates(transactions),
        "event_tier": _null_rates(event_tier),
    }

    # Duplicate order ids. One order can legitimately hold several tickets, so a repeat
    # order_id is only suspicious when the whole row repeats.
    dup_rows = int(transactions.duplicated().sum())
    order_counts = transactions["order_id"].value_counts()
    # A repeated event_id fans out the transaction join; build_events refuses to build one,
    # so this should always be 0 and is here to prove it. A shared (event_id, tier_ordinal)
    # means two tier NAMES in one event mapped to the same ordinal (e.g. "Tier 1" and
    # "Release 2", or two casings of one tier) — fix the labels or the rules before Phase 4.
    with_ordinal = event_tier.dropna(subset=["tier_ordinal"])
    report["duplicates"] = {
        "fully_duplicated_transaction_rows": dup_rows,
        "distinct_order_ids": int(transactions["order_id"].nunique()),
        "order_ids_appearing_more_than_once": int((order_counts > 1).sum()),
        "duplicate_event_ids": int(events["event_id"].duplicated().sum()),
        "event_tiers_sharing_an_ordinal": int(
            with_ordinal.duplicated(subset=["event_id", "tier_ordinal"]).sum()
        ),
    }

    # An unparseable price must never reach a model. It is not a free ticket and it is not
    # a zero: it is a mapping or formatting failure ("£10.00", "1,250") and it invalidates
    # every price number downstream until it is fixed.
    report["price_parsing"] = {
        "transactions_with_unparsed_price": int(transactions["price_paid"].isna().sum()),
        "transactions_with_unparsed_buyer_price": int(transactions["buyer_price"].isna().sum()),
    }

    # SPEC 8.4 survivorship: cancelled events are conditioning-on-success rows if dropped
    # silently. Print the count so a workbook column of "N"/"Y" cannot flip unnoticed.
    report["cancelled"] = {"cancelled_events": int(events["cancelled"].sum())}

    # Tier ordinal coverage: what fraction of tickets got a recognised tier, and which
    # labels defeated the normaliser (SPEC.md 8.8). Comps are excluded — they are not
    # releases and are not expected to have an ordinal, so counting them here would bury
    # the real misses in noise.
    sold = transactions[~transactions["is_comp"]]
    unmapped = sold.loc[sold["tier_ordinal"].isna(), "tier_name"]
    report["tier_coverage"] = {
        "transactions_with_ordinal": float(sold["tier_ordinal"].notna().mean())
        if len(sold)
        else 0.0,
        "unmapped_tier_labels": sorted({str(x) for x in unmapped.dropna().unique()}),
        "distinct_ordinals": sorted(int(x) for x in sold["tier_ordinal"].dropna().unique()),
    }

    # Face vs fee, per platform (SPEC.md 8.6). Stays UNKNOWN until confirmed from a real
    # order confirmation; everything downstream of price is provisional while it is.
    platforms = (
        sorted(transactions["platform"].dropna().unique())
        if "platform" in transactions.columns
        else []
    )
    report["fee_treatment"] = {
        str(p): FEE_TREATMENT.get(str(p), "UNKNOWN") for p in platforms
    } or dict(FEE_TREATMENT)

    # Events with no tickets, and tickets with no event: both are join failures.
    tx_events = set(transactions["event_id"].dropna().unique())
    ev_events = set(events["event_id"].dropna().unique())
    report["join_health"] = {
        "events_without_transactions": len(ev_events - tx_events),
        "transaction_events_missing_from_events_table": len(tx_events - ev_events),
    }

    # Comps broken out BY TIER NAME (SPEC 8.7). Stripping a comp is right; stripping a
    # paid tier because its name contains the word "free" or "guest" is an undercount
    # nothing downstream would catch, so the labels being stripped are printed and a
    # human reads them. A paid tier in this list means normalize.COMP_LABELS is wrong.
    comp_rows = transactions[transactions["is_comp"]]
    report["comps"] = {
        "comp_ticket_rows": int(transactions["is_comp"].sum()),
        "comp_share_of_rows": float(transactions["is_comp"].mean()) if len(transactions) else 0.0,
        "comp_rows_by_tier_name": {
            str(label): int(n) for label, n in comp_rows["tier_name"].value_counts().items()
        },
    }

    # Within-tier price movement (SPEC 4.3). event_tier.price is the price the tier OPENED
    # at, and n_distinct_prices says whether it stayed there. A tier whose price changed
    # mid-window is price variation at a given lead time — the one thing that separates
    # the price effect from the timing effect — so it gets counted, not averaged away.
    n_prices = event_tier["n_distinct_prices"]
    report["price_variation"] = {
        "event_tiers_with_more_than_one_price": int((n_prices > 1).sum()),
        "max_distinct_prices_in_one_tier": int(n_prices.max()) if len(n_prices) else 0,
    }

    # Privacy guard, fail-closed: any column that is not on the canonical allowlist is
    # treated as possible buyer PII. Must be empty in every table, always.
    report["unexpected_columns"] = {
        "events": sorted(str(c) for c in unexpected_columns(events)),
        "transactions": sorted(str(c) for c in unexpected_columns(transactions)),
        "event_tier": sorted(str(c) for c in unexpected_columns(event_tier)),
    }

    return report


def format_report(report: dict) -> str:
    """Render the report as readable text."""
    lines = ["VALIDATION REPORT", ""]

    lines.append("Row counts")
    for name, n in report["row_counts"].items():
        lines.append(f"    {name:<14} {n:>8,}")

    lines.append("")
    lines.append("Date ranges")
    for name, (lo, hi) in report["date_ranges"].items():
        lines.append(f"    {name:<28} {lo} .. {hi}")

    lines.append("")
    lines.append(f"Date parsing — {report['date_parsing']['assumed_format']}")
    for name, share in report["date_parsing"].items():
        if name == "assumed_format":
            continue
        lines.append(f"    {name:<45} {share:.1%} of rows could also read as mm/dd")

    lines.append("")
    lines.append("Duplicates")
    for k, v in report["duplicates"].items():
        lines.append(f"    {k:<40} {v:>8,}")

    lines.append("")
    tc = report["tier_coverage"]
    lines.append("Tier ordinal coverage (SPEC 8.8, comps excluded)")
    lines.append(f"    paid tickets with an ordinal          {tc['transactions_with_ordinal']:.1%}")
    lines.append(f"    ordinals seen                         {tc['distinct_ordinals']}")
    if tc["unmapped_tier_labels"]:
        lines.append("    UNMAPPED LABELS — add them to normalize.FIXED_TIER_ORDINALS if the")
        lines.append("    name states its place in the ladder. If it does not ('Final Release'")
        lines.append("    is release 3 on one event and release 5 on another), leave it unmapped")
        lines.append("    and let window_open order it. Never guess a position:")
        for label in tc["unmapped_tier_labels"]:
            lines.append(f"        {label!r}")

    lines.append("")
    lines.append("Cancelled events (SPEC 8.4)")
    lines.append(
        f"    events flagged cancelled              {report['cancelled']['cancelled_events']:,}"
    )

    lines.append("")
    lines.append("Comps / guestlist (SPEC 8.7 — excluded from units_sold)")
    lines.append(
        f"    comp rows                             {report['comps']['comp_ticket_rows']:,}"
    )
    lines.append(
        f"    share of rows                         {report['comps']['comp_share_of_rows']:.1%}"
    )
    if report["comps"]["comp_rows_by_tier_name"]:
        lines.append("    by tier name — a PAID tier listed here is a deleted sale, not a comp:")
        for label, n in report["comps"]["comp_rows_by_tier_name"].items():
            lines.append(f"        {label!r:<38} {n:,}")

    lines.append("")
    pv = report["price_variation"]
    lines.append("Within-tier price movement (SPEC 4.3 — price = the price the tier OPENED at)")
    lines.append(
        f"    tiers whose price changed mid-window  {pv['event_tiers_with_more_than_one_price']:,}"
    )
    lines.append(
        f"    most prices seen in one tier          {pv['max_distinct_prices_in_one_tier']}"
    )
    if pv["event_tiers_with_more_than_one_price"]:
        lines.append("    ^ price variation at a given lead time. Check each one is a real")
        lines.append("      correction and not a refund or a mapping error — if real, these are")
        lines.append("      the rows Phase 4 identification leans on.")

    lines.append("")
    lines.append("Face value vs booking fee, per platform (SPEC 8.6)")
    for platform, treatment in report["fee_treatment"].items():
        marker = "  <-- RESOLVE THIS" if treatment == "UNKNOWN" else ""
        lines.append(f"    {platform:<12} {treatment}{marker}")
    if any(v == "UNKNOWN" for v in report["fee_treatment"].values()):
        lines.append("    UNKNOWN means buyer_price is provisional and so is every price result")
        lines.append("    downstream. Confirm against one real order confirmation per platform,")
        lines.append("    then set FEE_TREATMENT in src/pricing/adapters.py.")

    lines.append("")
    lines.append("Join health")
    for k, v in report["join_health"].items():
        lines.append(f"    {k:<45} {v:>6,}")

    lines.append("")
    lines.append("Null rates (non-zero only)")
    for table, rates in report["null_rates"].items():
        lines.append(f"    {table}")
        if not rates:
            lines.append("        (none)")
        for col, rate in rates.items():
            lines.append(f"        {col:<32} {rate:.1%}")

    lines.append("")
    unparsed = report["price_parsing"]["transactions_with_unparsed_price"]
    if unparsed:
        lines.append(
            f"PRICE PARSE FAILURE — {unparsed:,} transactions have an unreadable price_paid."
        )
        lines.append("    These are NOT free tickets. Fix the column mapping or the source")
        lines.append("    formatting before believing any price, revenue or elasticity number.")
    else:
        lines.append("Prices: every transaction has a numeric price. OK.")

    lines.append("")
    leaked = {k: v for k, v in report["unexpected_columns"].items() if v}
    if leaked:
        lines.append(f"PRIVACY FAILURE — columns not on the canonical allowlist: {leaked}")
    else:
        lines.append("Privacy: every derived column is on the canonical allowlist. OK.")

    return "\n".join(lines)
