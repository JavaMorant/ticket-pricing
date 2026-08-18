"""Phase 1 — descriptives. "Look at it before you model it" (SPEC.md 2).

    uv run python -m pricing.phase1 --synthetic

What this produces: counts, distributions and shapes. Events per brand / venue / year,
capacity utilisation, revenue per event, tier-ladder shape, comp rates, sellout rate.
Tables and five figures, written to results/synthetic/ (or results/ for real data).

WHAT THIS PHASE DELIBERATELY DOES NOT DO
----------------------------------------
No modelling, and — this is the important one — **no hypothesis is TESTED here**. No
p-value, no FDR correction, no significance claim, no out-of-sample confirmation, no
statement that one group differs from another. Per SPEC.md 5.3 a pre-registered pattern
may only be tested against a FROZEN `preregistration.md`, with the FDR correction and the
final-year confirmation that go with it, and `preregistration.md` still says DRAFT.

Be precise about what that does and does not rule out, because the loose version of the
claim is not true of this report. Descriptive cross-tabs by **academic year**, **brand**
and **venue** ARE shown — median tickets by venue, median sell-through by brand, sellout
rate by year — and every one of those is also a line on the SPEC.md 5.4 list. They are
here because SPEC.md 8.9 requires knowing the growth trend and the programme's composition
BEFORE any later calendar effect can be believed: a night you only started running
recently proxies for "recent" in any model, and a brand that moved into bigger rooms shows
falling sell-through while growing. You cannot judge either without these tables. What
makes them descriptive rather than results is that nothing here compares them to a null,
attaches a p-value, or claims a difference is real.

What is genuinely absent is the CALENDAR crossed with an outcome: no tickets by term week,
no freshers'-week effect, no exam-period effect, no loan-instalment effect, no
tickets-by-day-of-week. Day of week and academic year appear only as counts of EVENTS —
programme composition, never demand.

The trap is subtle and worth stating plainly, because Phase 1 is exactly where it springs:
if you plot tickets by academic week now, you will see something, you will remember it, and
the "pre-registered" list you write afterwards will be a list of what you remember seeing.
That is SPEC.md 5.2 — generating and confirming a hypothesis on the same data — and it is
not recoverable by being careful later.

This module also owns the per-event fact table (`event_outcomes`) that Phases 2 and 3
import: one row per event with tickets, revenue, sell-through and the sellout flag. It is
defined once, here, so "tickets sold" means the same thing in every later phase.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing import phaseio, report

# Paid units / capacity at or above this counts as a sellout. A judgement call, not a
# fact: real rooms run a few tickets short of licensed capacity for comps, guestlist and
# no-shows, so demanding 1.00 would report zero sellouts on nights that were visibly full.
SELLOUT_THRESHOLD = 0.95


# --------------------------------------------------------------------------------------
# The per-event fact table (imported by Phases 2 and 3)
# --------------------------------------------------------------------------------------


def event_outcomes(events: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per event: metadata, plus what actually happened.

    Definitions, fixed here and reused by every later phase:

      tickets       PAID units. Comps and guestlist are excluded (SPEC.md 8.7 — a comp is
                    not demand; counting them inflates sell-through and, worse, makes a
                    poorly-selling night look like a healthy one).
      comps         the excluded units, counted separately so they are visible.
      revenue_buyer sum of what buyers paid (face + fee). PROVISIONAL: `fee_treatment` is
                    still UNKNOWN for every platform (SPEC.md 8.6), so this is the
                    face-plus-fee reading, which is the common platform default and a
                    guess. It travels as a column so nothing downstream loses the caveat.
      revenue_face  sum of face value only — the operator's side of the same tickets.
      sell_through  tickets / capacity. The demand measure.
      room_fill     (tickets + comps) / capacity. The fire-marshal measure. They differ,
                    and mixing them up is how a 60%-sold night becomes a "sellout".
      sellout       sell_through >= SELLOUT_THRESHOLD.

    Cancelled events are KEPT here and flagged, never dropped (SPEC.md 8.4). They are the
    most informative rows in the dataset — the nights that were selling so badly they got
    pulled — and a table that silently drops them is conditioning on success. Later phases
    each make their own explicit decision about them, and say what it was.
    """
    paid = transactions[~transactions["is_comp"]]
    comps = transactions[transactions["is_comp"]]

    per_event = paid.groupby("event_id").apply(
        lambda g: pd.Series(
            {
                "tickets": int(g["quantity"].sum()),
                "revenue_buyer": float((g["buyer_price"] * g["quantity"]).sum()),
                "revenue_face": float((g["price_paid"] * g["quantity"]).sum()),
                "n_orders": int(g["order_id"].nunique()),
                "platform": g["platform"].iloc[0],
            }
        ),
        include_groups=False,
    )

    out = events.copy()
    out = out.merge(per_event, left_on="event_id", right_index=True, how="left")
    out = out.merge(
        comps.groupby("event_id")["quantity"].sum().rename("comps"),
        left_on="event_id",
        right_index=True,
        how="left",
    )
    # An event with no paid rows at all is a real possibility (a pulled event that never
    # went on sale). Zero is the honest value; NaN would quietly drop it from every mean.
    for col in ["tickets", "comps", "revenue_buyer", "revenue_face", "n_orders"]:
        out[col] = out[col].fillna(0)
    out[["tickets", "comps", "n_orders"]] = out[["tickets", "comps", "n_orders"]].astype(int)

    out["academic_year"] = phaseio.academic_year(out["event_date"])
    out["dow"] = out["event_date"].dt.day_name()
    out["sell_through"] = out["tickets"] / out["capacity"]
    out["room_fill"] = (out["tickets"] + out["comps"]) / out["capacity"]
    out["sellout"] = out["sell_through"] >= SELLOUT_THRESHOLD
    out["comp_rate"] = out["comps"] / (out["comps"] + out["tickets"]).replace(0, np.nan)
    out["comp_rate"] = out["comp_rate"].fillna(0.0)
    out["revenue_per_head"] = out["revenue_buyer"] / out["tickets"].replace(0, np.nan)
    return out.sort_values("event_date").reset_index(drop=True)


def live(ev: pd.DataFrame) -> pd.DataFrame:
    """The events that actually happened. Cancelled ones are reported separately."""
    return ev[~ev["cancelled"]]


# --------------------------------------------------------------------------------------
# Ladder shape
# --------------------------------------------------------------------------------------


def ladder(event_tier: pd.DataFrame) -> pd.DataFrame:
    """Paid tiers with a `rung` — their position in the on-sale order within the event.

    Position comes from `window_open` (the first sale in the tier), not from the tier
    NAME's ordinal. Names lie: "Release 2" and "Tier 1" are the same rung under two
    conventions, "Final Release" states no position at all, and normalize.py deliberately
    leaves labels it does not recognise unmapped rather than guessing. Timing does not lie.

    Comp-only tiers (Guestlist, Press) have no price and no paid units, so they carry no
    ladder information and are dropped here.
    """
    paid = event_tier[(event_tier["units_sold"] > 0) & event_tier["price"].notna()].copy()
    paid = paid.sort_values(["event_id", "window_open"])
    paid["rung"] = paid.groupby("event_id").cumcount() + 1
    return paid.reset_index(drop=True)


def ladder_shape(rungs: pd.DataFrame) -> pd.DataFrame:
    """One row per event: how many rungs, what the ladder spans in price and in time."""
    grouped = rungs.groupby("event_id")
    shape = pd.DataFrame(
        {
            "n_rungs": grouped.size(),
            "price_first": grouped["price"].first(),
            "price_last": grouped["price"].last(),
            "first_onsale_lead_days": grouped["lead_time_open_days"].max(),
            "last_onsale_lead_days": grouped["lead_time_open_days"].min(),
            "units": grouped["units_sold"].sum(),
        }
    )
    shape["price_ratio"] = shape["price_last"] / shape["price_first"]
    shape["ladder_span_days"] = shape["first_onsale_lead_days"] - shape["last_onsale_lead_days"]
    return shape.reset_index()


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------


def counts_by_year_brand(ev: pd.DataFrame) -> pd.DataFrame:
    """Events per academic year x brand, plus the row totals."""
    table = pd.crosstab(ev["academic_year"], ev["brand"])
    table["total"] = table.sum(axis=1)
    return table.reset_index()


def counts_by_venue(ev: pd.DataFrame) -> pd.DataFrame:
    """Events per venue, with the room size and which brands used it."""
    out = (
        ev.groupby(["venue", "city"])
        .agg(
            events=("event_id", "size"),
            capacity=("capacity", "max"),
            brands=("brand", lambda s: ", ".join(sorted(s.unique()))),
            median_tickets=("tickets", "median"),
        )
        .reset_index()
    )
    return out.sort_values("capacity").reset_index(drop=True)


def distribution(series: pd.Series, name: str) -> pd.DataFrame:
    """min / p10 / p25 / median / p75 / p90 / max / mean for one column.

    Percentiles rather than mean +/- sd because none of these distributions is symmetric:
    revenue per event is right-skewed by a handful of big rooms, and quoting a mean and an
    sd would describe a bell curve that is not there.
    """
    q = series.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    return pd.DataFrame(
        [
            {
                "measure": name,
                "n": int(series.notna().sum()),
                "min": series.min(),
                "p10": q.loc[0.10],
                "p25": q.loc[0.25],
                "median": q.loc[0.50],
                "p75": q.loc[0.75],
                "p90": q.loc[0.90],
                "max": series.max(),
                "mean": series.mean(),
            }
        ]
    )


def by_brand(ev: pd.DataFrame) -> pd.DataFrame:
    """The headline per-brand summary."""
    out = (
        ev.groupby("brand")
        .agg(
            events=("event_id", "size"),
            median_capacity=("capacity", "median"),
            median_tickets=("tickets", "median"),
            median_sell_through=("sell_through", "median"),
            sellout_rate=("sellout", "mean"),
            median_revenue=("revenue_buyer", "median"),
            median_comp_rate=("comp_rate", "median"),
        )
        .reset_index()
    )
    return out


def sellout_by_year(ev: pd.DataFrame) -> pd.DataFrame:
    """Sellout rate and median sell-through per academic year.

    Read this next to the event counts, not on its own (SPEC.md 8.9): if the programme
    moved into bigger rooms over time, sell-through falls even as the brand grows.
    """
    out = (
        ev.groupby("academic_year")
        .agg(
            events=("event_id", "size"),
            median_capacity=("capacity", "median"),
            median_tickets=("tickets", "median"),
            median_sell_through=("sell_through", "median"),
            sellouts=("sellout", "sum"),
            sellout_rate=("sellout", "mean"),
        )
        .reset_index()
    )
    return out


def units_by_rung(rungs: pd.DataFrame) -> pd.DataFrame:
    """Where the tickets are sold on the ladder, and at what price."""
    out = (
        rungs.groupby("rung")
        .agg(
            tiers=("event_id", "size"),
            events=("event_id", "nunique"),
            median_price=("price", "median"),
            median_lead_days=("lead_time_open_days", "median"),
            units=("units_sold", "sum"),
        )
        .reset_index()
    )
    out["share_of_units"] = out["units"] / out["units"].sum()
    return out


def comp_summary(ev: pd.DataFrame) -> pd.DataFrame:
    """Comps by brand: how many events issue them, and how big they get."""
    out = (
        ev.groupby("brand")
        .agg(
            events=("event_id", "size"),
            events_with_comps=("comps", lambda s: int((s > 0).sum())),
            comps_total=("comps", "sum"),
            median_comp_rate=("comp_rate", lambda s: s[s > 0].median()),
            max_comp_rate=("comp_rate", "max"),
        )
        .reset_index()
    )
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_programme(ev: pd.DataFrame, out_dir: Path) -> Path:
    """Programme composition: events per year per brand, and per venue."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    table = pd.crosstab(ev["academic_year"], ev["brand"])
    x = np.arange(len(table.index))
    width = 0.38
    for i, brand in enumerate(table.columns):
        ax1.bar(
            x + (i - 0.5) * width,
            table[brand],
            width,
            label=brand,
            color=report.BRAND_COLOURS.get(brand, report.GREY),
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(table.index, fontsize=8)
    ax1.legend(fontsize=8, frameon=False)
    report.style(ax1, "Events per academic year", "academic year", "events")

    venues = ev.groupby(["venue", "capacity"]).size().reset_index(name="events")
    venues = venues.sort_values("capacity")
    labels = [f"{v}\n({c:,} cap)" for v, c in zip(venues["venue"], venues["capacity"], strict=True)]
    ax2.bar(labels, venues["events"], color=report.GREY)
    report.style(ax2, "Events per venue", "venue", "events")
    ax2.tick_params(axis="x", labelsize=8)

    return report.save_fig(fig, out_dir, "phase1_programme.png")


def fig_sell_through(ev: pd.DataFrame, out_dir: Path) -> Path:
    """The distribution that matters most: how full the rooms actually got."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.hist(ev["sell_through"], bins=20, color=report.GREY, edgecolor="white")
    ax1.axvline(
        SELLOUT_THRESHOLD, color="#b03030", linestyle="--", linewidth=1.2, label="sellout line"
    )
    ax1.legend(fontsize=8, frameon=False)
    report.style(ax1, "Capacity sell-through (paid units / capacity)", "sell-through", "events")

    brands = sorted(ev["brand"].unique())
    ax2.boxplot(
        [ev.loc[ev["brand"] == b, "sell_through"] for b in brands],
        tick_labels=brands,
        widths=0.5,
        medianprops={"color": "#b03030"},
    )
    ax2.axhline(SELLOUT_THRESHOLD, color="#b03030", linestyle="--", linewidth=1.0)
    report.style(ax2, "Sell-through by brand", "", "sell-through")

    return report.save_fig(fig, out_dir, "phase1_sell_through.png")


def fig_revenue(ev: pd.DataFrame, out_dir: Path) -> Path:
    """Revenue per event against room size, and how it moved across the five years."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for brand in sorted(ev["brand"].unique()):
        sub = ev[ev["brand"] == brand]
        ax1.scatter(
            sub["capacity"],
            sub["revenue_buyer"],
            s=18,
            alpha=0.75,
            label=brand,
            color=report.BRAND_COLOURS.get(brand, report.GREY),
        )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8, frameon=False)
    report.style(ax1, "Revenue per event vs capacity (log-log)", "capacity", "buyer revenue (GBP)")

    years = sorted(ev["academic_year"].unique())
    ax2.boxplot(
        [ev.loc[ev["academic_year"] == y, "revenue_buyer"] for y in years],
        tick_labels=years,
        widths=0.55,
        medianprops={"color": "#b03030"},
    )
    ax2.tick_params(axis="x", labelsize=8)
    report.style(ax2, "Revenue per event by academic year", "", "buyer revenue (GBP)")

    return report.save_fig(fig, out_dir, "phase1_revenue.png")


def fig_ladder(rungs: pd.DataFrame, shape: pd.DataFrame, out_dir: Path) -> Path:
    """Ladder shape: price by rung, and how many rungs events actually use."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    positions = sorted(rungs["rung"].unique())
    ax1.boxplot(
        [rungs.loc[rungs["rung"] == r, "price"] for r in positions],
        tick_labels=[str(r) for r in positions],
        widths=0.5,
        medianprops={"color": "#b03030"},
    )
    report.style(ax1, "Buyer price by rung (on-sale order)", "rung", "buyer price (GBP)")

    counts = shape["n_rungs"].value_counts().sort_index()
    ax2.bar([str(i) for i in counts.index], counts.to_numpy(), color=report.GREY)
    report.style(ax2, "Rungs per event", "paid tiers in the ladder", "events")

    return report.save_fig(fig, out_dir, "phase1_ladder.png")


def fig_comps_and_sellouts(ev: pd.DataFrame, out_dir: Path) -> Path:
    """Comp rate distribution, and the sellout rate year by year."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.hist(ev["comp_rate"], bins=20, color=report.GREY, edgecolor="white")
    report.style(ax1, "Comp rate (comps / all admissions)", "comp rate", "events")

    per_year = ev.groupby("academic_year")["sellout"].mean()
    ax2.bar(per_year.index, per_year.to_numpy(), color=report.GREY)
    ax2.tick_params(axis="x", labelsize=8)
    report.style(ax2, f"Sellout rate (sell-through >= {SELLOUT_THRESHOLD:.0%})", "", "share")

    return report.save_fig(fig, out_dir, "phase1_comps_sellouts.png")


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def build(run: phaseio.Run) -> list[Path]:
    """Compute every table, draw every figure, write the markdown. Returns paths written."""
    out_dir = run.out_dir
    ev_all = event_outcomes(run.events, run.transactions)
    ev = live(ev_all)
    rungs = ladder(run.event_tier)
    rungs = rungs[rungs["event_id"].isin(ev["event_id"])]
    shape = ladder_shape(rungs)

    figures = [
        fig_programme(ev, out_dir),
        fig_sell_through(ev, out_dir),
        fig_revenue(ev, out_dir),
        fig_ladder(rungs, shape, out_dir),
        fig_comps_and_sellouts(ev, out_dir),
    ]

    cancelled = ev_all[ev_all["cancelled"]]
    spans = f"{ev['event_date'].min().date()} to {ev['event_date'].max().date()}"

    lines = [
        phaseio.provenance_header(
            "Phase 1 — Descriptives",
            run.source,
            run.seed,
            phaseio.regenerate_command("phase1", run.source, run.seed),
        ),
        "",
    ]
    lines += [
        "Counts, distributions and shapes only. **No model is fitted here and no hypothesis is",
        "tested**: there is no p-value in this report, no multiple-comparison correction, no",
        "significance claim and no out-of-sample confirmation. `preregistration.md` has been",
        "FROZEN since commit `1c7196d` (SPEC.md §5.3), so the §5.4 list is now fixed and the",
        "ordering it protects has been honoured — but nothing on that list has been tested,",
        "because there is no real data to test it on. The platform exports are parked, and",
        "every number below comes from the seeded synthetic fixture. Testing a pattern you",
        "noticed while making this report is SPEC.md §5.2, the cardinal sin; freezing the list",
        "first is what makes that a checkable rule rather than a good intention.",
        "",
        "**What that does not mean.** Descriptive cross-tabs by academic year, brand and venue",
        "are below, and each of those is also a line on the §5.4 list. They are here because",
        "SPEC.md §8.9 says you must know the growth trend and the programme's composition before",
        "you can believe any later calendar effect — a night you only started running recently",
        "proxies for 'recent', and a brand that moved into bigger rooms shows falling",
        "sell-through while growing. Reading a median off a table is not testing a hypothesis;",
        "what would be is comparing it to a null and claiming the difference is real, and",
        "nothing here does that.",
        "",
        "**What is genuinely absent is the calendar crossed with an outcome:** no tickets by term",
        "week, no freshers' week, no exam period, no loan instalment, no tickets by day of week.",
        "Day of week and academic year appear below only as counts of *events*.",
        "",
        "---",
        "",
        "## 1. The programme",
        "",
        f"- **{len(ev):,} events that ran**, {spans}",
        (
            f"- **{len(cancelled)} cancelled/pulled events**, kept in the dataset and flagged "
            "(SPEC.md §8.4), excluded from every table below"
        ),
        (
            f"- **{int(ev['tickets'].sum()):,} paid tickets**, "
            f"{int(ev['comps'].sum()):,} comps/guestlist (excluded from demand, SPEC.md §8.7)"
        ),
        f"- **{len(rungs):,} paid tiers** across those events",
        f"- Platforms: {_platform_line(ev)}",
        "",
        report.md_table(counts_by_year_brand(ev)),
        "",
        "### Per venue",
        "",
        report.md_table(counts_by_venue(ev)),
        "",
        "### Per day of week",
        "",
        "Counts of **events**, not of tickets. This is programme composition — which nights",
        "the brands actually ran — and it is here because SPEC.md §8.9 warns that a night",
        "you only started running recently will proxy for 'recent' in any later model. The",
        "tickets-by-day-of-week comparison is a §5.4 hypothesis and is not tested here.",
        "",
        report.md_table(ev.groupby("dow").size().reset_index(name="events")),
        "",
        "---",
        "",
        "## 2. Capacity utilisation",
        "",
        report.md_table(
            pd.concat(
                [
                    distribution(ev["sell_through"], "sell_through (paid / capacity)"),
                    distribution(ev["room_fill"], "room_fill (paid + comps / capacity)"),
                    distribution(ev["tickets"].astype(float), "tickets sold"),
                ]
            ),
            floatfmt="{:,.3f}",
        ),
        "",
        (
            f"**Sellout rate: {ev['sellout'].mean():.1%}** "
            f"({int(ev['sellout'].sum())} of {len(ev)} events at or above "
            f"{SELLOUT_THRESHOLD:.0%} sell-through)."
        ),
        "",
        report.md_table(sellout_by_year(ev), floatfmt="{:,.3f}"),
        "",
        "---",
        "",
        "## 3. Revenue per event",
        "",
        "`revenue_buyer` is what buyers paid (face + fee); `revenue_face` is face value only.",
        "**Both are provisional**: `fee_treatment` is `UNKNOWN` for every platform",
        "(SPEC.md §8.6), so face-plus-fee is the assumed reading and not a confirmed one.",
        "",
        report.md_table(
            pd.concat(
                [
                    distribution(ev["revenue_buyer"], "revenue_buyer (GBP)"),
                    distribution(ev["revenue_face"], "revenue_face (GBP)"),
                    distribution(ev["revenue_per_head"], "revenue per paid ticket (GBP)"),
                    distribution(ev["fixed_cost_stack"], "fixed cost stack (GBP)"),
                ]
            ),
        ),
        "",
        "### By brand",
        "",
        report.md_table(by_brand(ev), floatfmt="{:,.3f}"),
        "",
        "---",
        "",
        "## 4. Tier-ladder shapes",
        "",
        "`rung` is position in **on-sale order** (from each tier's first sale), not the tier",
        "name's ordinal — names disagree across events and some state no position at all.",
        "",
        report.md_table(units_by_rung(rungs), floatfmt="{:,.3f}"),
        "",
        "### Ladder span per event",
        "",
        report.md_table(
            pd.concat(
                [
                    distribution(shape["n_rungs"].astype(float), "rungs per event"),
                    distribution(shape["price_ratio"], "top price / opening price"),
                    distribution(shape["first_onsale_lead_days"], "first on-sale (days out)"),
                    distribution(shape["ladder_span_days"], "ladder span (days)"),
                ]
            ),
            floatfmt="{:,.2f}",
        ),
        "",
        "The last row is the one Phase 4 lives or dies on. If every event opened its ladder",
        "on the same schedule, tier price and lead time would be the same variable and the",
        "elasticity would not be identified (SPEC.md §4.3). Spread in `first on-sale` and",
        "`ladder span` across events is the variation that separates them.",
        "",
        "---",
        "",
        "## 5. Comps and guestlist",
        "",
        report.md_table(comp_summary(ev), floatfmt="{:,.3f}"),
        "",
        "---",
        "",
        "## 6. Cancelled events (SPEC.md §8.4)",
        "",
        (
            report.md_table(
                cancelled[
                    [
                        "event_id",
                        "event_date",
                        "brand",
                        "venue",
                        "capacity",
                        "tickets",
                        "sell_through",
                    ]
                ],
                floatfmt="{:,.3f}",
            )
            if len(cancelled)
            else "_None in this dataset._"
        ),
        "",
        "These are the nights that did not happen. They are in the table above and excluded",
        "from every other table in this report, which is a choice, not an accident: their",
        "sales histories are truncated at the moment they were pulled, so their sell-through",
        "is not comparable to an event that ran. Excluding them means every model below is",
        "**conditioned on the event having gone ahead**, and that limitation belongs in the",
        "README (DECISIONS.md, survivorship).",
        "",
        "---",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{p.name}`" for p in figures]
    lines += [
        "",
        "".join(f"![{p.stem}]({p.name})\n" for p in figures),
    ]

    print(
        f"events {len(ev):,} that ran ({len(cancelled)} cancelled)   "
        f"tickets {int(ev['tickets'].sum()):,}   comps {int(ev['comps'].sum()):,}   "
        f"tiers {len(rungs):,}"
    )
    print(
        f"sell-through median {ev['sell_through'].median():.1%}   "
        f"sellout rate {ev['sellout'].mean():.1%}   "
        f"revenue/event median GBP {ev['revenue_buyer'].median():,.0f}   "
        f"comp rate median {ev['comp_rate'].median():.1%}"
    )

    return [phaseio.write_report(lines, "phase1_descriptives.md", out_dir), *figures]


def _platform_line(ev: pd.DataFrame) -> str:
    counts = ev["platform"].value_counts()
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def main(argv: list[str] | None = None) -> int:
    return phaseio.run_phase(build, "pricing.phase1", __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
