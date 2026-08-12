"""Phase 3 — sales curve (SPEC.md 6.2). How tickets accumulate, and what that forecasts.

    uv run python -m pricing.phase3 --synthetic

Two questions, one object:

  1. **What shape is the presale?** Cumulative share of final sales against days-to-event,
     per event, then the median curve per segment.
  2. **Given the presale so far, how many will this event sell?** At day -7, -3 and -1,
     scale what has already sold by the segment's median share at that day:

         forecast = tickets_so_far(d) / median_share_at(d)

     That is "presale velocity" (SPEC.md 6.2) and it is the most operationally useful
     number in the repo: it is what tells you on the Monday whether to spend on marketing.

WHY EMPIRICAL AND NOT BASS DIFFUSION
------------------------------------
SPEC.md 6.2 offers Bass diffusion "or something simpler that works". The empirical median
curve IS the simpler thing, and here it is also the better one:

  * Bass has three parameters (p, q, m) fitted per event by non-linear least squares, and
    `m` — the market potential — is the very thing being forecast. Fitting a curve whose
    parameter is the answer, from a partial series, at ~24 events per segment, is a lot of
    machinery to arrive at a ratio.
  * The empirical curve makes one assumption and states it plainly: **this event's presale
    is shaped like the median of previous events in its segment.** That assumption can be
    checked directly by looking at the spread of the curves, and the report shows it.
  * It cannot fail to converge, has nothing to tune, and can be explained in one sentence.

If the curves turn out to be wildly heterogeneous — the spread band in the figure is the
diagnostic — a parametric curve would not rescue that; it would hide it.

THE SEGMENTATION TRAP
---------------------
The obvious segmentation is brand x sold-out-or-not, and it is half illegal. **You do not
know on day -7 whether the event will sell out.** A forecast that uses the sellout flag is
using the answer. So the report carries both, clearly separated:

  * `brand`            — usable ex ante. This is the forecaster.
  * `brand x sellout`  — descriptive only. It shows whether the curve shape differs between
                         nights that sold out and nights that did not, which is worth
                         knowing, and it is NOT a forecast anyone could have made.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing import phaseio, report
from pricing.phase1 import event_outcomes, live
from pricing.phase2 import build_features, cross_validate, grouped_kfold, scores

# Days before the event that the curve is measured over. 90 covers the median ladder
# (first on-sale is typically ~70 days out); the handful of tickets sold earlier than that
# are counted in the day-90 bucket rather than dropped, so every curve still ends at 1.0.
MAX_LEAD = 90

# The days a decision actually gets made on: the week before, the last few days, the day
# before. Chosen because they are when marketing spend and tier drops are still possible.
FORECAST_DAYS = (7, 3, 1)

# A segment needs this many training events before its own median curve is trusted; below
# it, the pooled curve is used instead. Twelve is a judgement call — small enough to be
# usable on ~120 events, big enough that one weird night cannot define a segment's shape.
MIN_EVENTS_PER_SEGMENT = 12

# Floor on the denominator. Dividing observed sales by a near-zero share explodes: at 2%
# sold, a one-ticket error becomes a 50-ticket error in the forecast.
MIN_SHARE = 0.05


# --------------------------------------------------------------------------------------
# The curves
# --------------------------------------------------------------------------------------


def cumulative_sales(
    events: pd.DataFrame, transactions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(cumulative units, cumulative share, final units) by event x days-to-event.

    Columns run 90, 89, ... 1, 0 — descending lead time, i.e. left to right is forwards in
    time. Column `d` is "everything sold with at least d days to go".

    Comps are excluded (SPEC.md 8.7): a guestlist spot issued three days out is not a sale
    and would show up as a burst of late demand that no marketing decision could act on.
    Cancelled events are excluded too (SPEC.md 8.4) — their curves stop when the event was
    pulled, so their final total is not a final total.
    """
    live_ids = set(live(events)["event_id"]) if "cancelled" in events.columns else None
    paid = transactions[~transactions["is_comp"]]
    if live_ids is not None:
        paid = paid[paid["event_id"].isin(live_ids)]

    lead = paid["lead_time_days"].clip(lower=0, upper=MAX_LEAD)
    per_day = pd.crosstab(paid["event_id"], lead)
    per_day = per_day.reindex(columns=range(MAX_LEAD, -1, -1), fill_value=0)

    cum = per_day.cumsum(axis=1)
    final = cum.iloc[:, -1].rename("final")
    share = cum.div(final, axis=0)
    return cum, share, final


def segment_labels(ev: pd.DataFrame, by: list[str]) -> pd.Series:
    """One label per event, e.g. "Brand A" or "Brand A / sellout".

    Joined with " / " rather than " | " on purpose: these labels end up as cells in a
    markdown table, and a pipe inside a cell splits the row into extra columns.
    """
    frame = ev.set_index("event_id")[by].astype(str)
    if "sellout" in by:
        frame["sellout"] = np.where(ev.set_index("event_id")["sellout"], "sellout", "not sellout")
    return frame.agg(" / ".join, axis=1).rename("segment")


def median_curves(
    share: pd.DataFrame, segments: pd.Series, min_events: int = MIN_EVENTS_PER_SEGMENT
) -> tuple[pd.DataFrame, list[str]]:
    """Pointwise median curve per segment, with a pooled fallback for thin segments.

    Pointwise median, not a fitted curve. Worth knowing why that is safe: each event's
    cumulative share is non-decreasing, and the pointwise median of a set of non-decreasing
    functions is itself non-decreasing. So the median curve is a valid cumulative curve for
    free — no smoothing, no monotone constraint, nothing to tune.

    Returns (curves indexed by segment name with a "__pooled__" row, names that fell back).
    """
    pooled = share.median()
    curves: dict[str, pd.Series] = {"__pooled__": pooled}
    fell_back: list[str] = []
    for name, ids in segments.groupby(segments).groups.items():
        rows = share.loc[share.index.intersection(ids)]
        if len(rows) >= min_events:
            curves[str(name)] = rows.median()
        else:
            curves[str(name)] = pooled
            fell_back.append(f"{name} (n={len(rows)})")
    return pd.DataFrame(curves).T, fell_back


def forecast_final(
    cum: pd.DataFrame, curves: pd.DataFrame, segments: pd.Series, day: int
) -> pd.Series:
    """Scale up what has sold by day `day`, using each event's segment median curve.

    Two guards, both one-liners with a reason:
      * the share is floored at MIN_SHARE, because dividing by a near-zero denominator
        turns a one-ticket sampling wobble into a fifty-ticket forecast error;
      * the forecast is floored at what has ALREADY sold, because an event cannot sell
        fewer tickets than it has sold. Free accuracy, and it stops the forecast being
        visibly absurd on a fast starter.
    """
    keys = segments.reindex(cum.index).fillna("__pooled__")
    keys = keys.where(keys.isin(curves.index), "__pooled__")
    share_at_day = keys.map(curves[day]).clip(lower=MIN_SHARE)
    return np.maximum(cum[day] / share_at_day, cum[day]).rename(f"forecast_d{day}")


# --------------------------------------------------------------------------------------
# Held-out evaluation
# --------------------------------------------------------------------------------------


def evaluate(
    cum: pd.DataFrame, share: pd.DataFrame, final: pd.Series, segments: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grouped K-fold by event: median curves from training events only.

    This is the whole discipline of the phase in one function. Building the median curve on
    all events and then "forecasting" those same events would score the curve against data
    that helped draw it — the SPEC.md 8.1 mistake in its sales-curve costume. Each fold's
    curve therefore comes from the other four folds, and the forecast of an event is made
    by a curve that never saw it.

    Returns (metric rows, per-event held-out forecasts).
    """
    rows, held = [], []
    for train_idx, test_idx in grouped_kfold(pd.Series(cum.index)):
        train_ids = cum.index[train_idx]
        test_ids = cum.index[test_idx]

        curves, _ = median_curves(share.loc[train_ids], segments.reindex(train_ids))
        pooled_only = pd.DataFrame({"__pooled__": share.loc[train_ids].median()}).T

        for day in FORECAST_DAYS:
            segmented = forecast_final(cum.loc[test_ids], curves, segments, day)
            pooled = forecast_final(
                cum.loc[test_ids], pooled_only, pd.Series("__pooled__", index=test_ids), day
            )
            held.append(
                pd.DataFrame(
                    {
                        "event_id": test_ids,
                        "day": day,
                        "actual": final.loc[test_ids].to_numpy(),
                        "sold_by_day": cum.loc[test_ids, day].to_numpy(),
                        "forecast": segmented.to_numpy(),
                        "pooled_forecast": pooled.to_numpy(),
                    }
                )
            )

    held_df = pd.concat(held, ignore_index=True)
    for day in FORECAST_DAYS:  # chronological: -7 first, then -3, then -1
        group = held_df[held_df["day"] == day]
        rows.append(
            {
                "day": f"-{day}",
                "events": len(group),
                "median_share_sold": float((group["sold_by_day"] / group["actual"]).median()),
                **{
                    f"segment_{k}": v for k, v in scores(group["actual"], group["forecast"]).items()
                },
                **{
                    f"pooled_{k}": v
                    for k, v in scores(group["actual"], group["pooled_forecast"]).items()
                },
                # The floor of the whole exercise: assume nothing else sells. It is what you
                # would "forecast" with no model at all, and any curve must beat it.
                **{
                    f"nosale_{k}": v
                    for k, v in scores(group["actual"], group["sold_by_day"]).items()
                },
            }
        )
    return pd.DataFrame(rows), held_df


def spread_table(share: pd.DataFrame) -> pd.DataFrame:
    """How tightly the curves agree — the assumption behind the whole method, measured."""
    days = [60, 30, 21, 14, 7, 3, 1]
    out = pd.DataFrame(
        {
            "days_to_event": [f"-{d}" for d in days],
            "p10": [share[d].quantile(0.10) for d in days],
            "median": [share[d].median() for d in days],
            "p90": [share[d].quantile(0.90) for d in days],
        }
    )
    out["p90_minus_p10"] = out["p90"] - out["p10"]
    return out


def segment_curve_table(share: pd.DataFrame, segments: pd.Series) -> pd.DataFrame:
    """Median share sold at each decision day, per segment, with the segment's size."""
    days = [14, *FORECAST_DAYS]
    rows = []
    for name, ids in segments.groupby(segments).groups.items():
        rows_for_segment = share.loc[share.index.intersection(ids)]
        row = {"segment": str(name), "events": len(rows_for_segment)}
        row.update({f"share_at_-{d}": rows_for_segment[d].median() for d in days})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def _comparison(metrics: pd.DataFrame, phase2_scores: dict[str, float]) -> pd.DataFrame:
    """The Phase 2 forecast and the curve forecasts side by side, on the same events."""
    rows = [
        {
            "forecast": "Phase 2 model (pricing time)",
            "information used": "features only, no tickets sold yet",
            "MAE": phase2_scores["MAE"],
            "MAPE_pct": phase2_scores["MAPE_pct"],
        }
    ]
    for _, row in metrics.iterrows():
        rows.append(
            {
                "forecast": f"sales curve at day {row['day']}",
                "information used": f"presale to day {row['day']} "
                f"({row['median_share_sold']:.0%} of final sold)",
                "MAE": row["segment_MAE"],
                "MAPE_pct": row["segment_MAPE_pct"],
            }
        )
    return pd.DataFrame(rows)


def _segmentation_sentence(metrics: pd.DataFrame) -> str:
    """Say out loud whether segmenting by brand earned its keep. Usually it will not."""
    delta = float((metrics["segment_MAPE_pct"] - metrics["pooled_MAPE_pct"]).mean())
    if abs(delta) < 0.25:
        return (
            f"Averaged over the three decision days the brand-segmented curve scores "
            f"{delta:+.2f} MAPE points against the pooled one — that is, nothing. The two "
            f"brands' presales are the same shape here, so the honest conclusion is to use "
            f"the pooled curve. A split that buys nothing still costs degrees of freedom, "
            f"and on real data it would eventually pick up noise and call it a segment."
        )
    verdict = "worth having" if delta < 0 else "actively hurting"
    return (
        f"Averaged over the three decision days the brand-segmented curve scores "
        f"{delta:+.2f} MAPE points against the pooled one (negative is better), so the "
        f"split is {verdict}."
    )


def _overtake_sentence(metrics: pd.DataFrame, phase2_scores: dict[str, float]) -> str:
    """State plainly which day the presale forecast overtakes the pricing-time model.

    Written as a computation rather than a claim because the answer is not knowable in
    advance and will differ on real data. On the synthetic fixture the purchase-time
    process is generated independently of the event's demand shock, so an early presale
    carries less information than the features do — exactly the sort of thing that would be
    embarrassing to have asserted from memory.
    """
    better = metrics[metrics["segment_MAPE_pct"] < phase2_scores["MAPE_pct"]]
    if better.empty:
        return (
            f"On this dataset the presale forecast never beats the pricing-time model "
            f"(best {metrics['segment_MAPE_pct'].min():.1f}% vs "
            f"{phase2_scores['MAPE_pct']:.1f}%). The presale is still worth watching — it is "
            f"the only thing that reacts to a night going wrong — but on these numbers it "
            f"adds nothing to the forecast, and the two should be combined rather than "
            f"chosen between."
        )
    first = better.iloc[0]
    return (
        f"The presale forecast overtakes the pricing-time model at **day {first['day']}** "
        f"({first['segment_MAPE_pct']:.1f}% vs {phase2_scores['MAPE_pct']:.1f}% MAPE) and not "
        f"before. Earlier than that, what you already knew when you priced the event beats "
        f"what the presale has told you so far. The obvious next step is to use both — the "
        f"pricing-time forecast as the prior, the presale as the update — which is Phase 5's "
        f"job, not this one's."
    )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_curves(share: pd.DataFrame, segments: pd.Series, out_dir: Path) -> Path:
    """Every event's curve with the pooled median band, and the median per brand."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    days = share.columns.to_numpy()

    for _, row in share.iterrows():
        ax1.plot(days, row.to_numpy(), color=report.GREY, alpha=0.10, linewidth=0.8)
    ax1.fill_between(
        days,
        share.quantile(0.25).to_numpy(),
        share.quantile(0.75).to_numpy(),
        color="#3d6bb3",
        alpha=0.25,
        label="p25-p75",
    )
    ax1.plot(days, share.median().to_numpy(), color="#3d6bb3", linewidth=2, label="median")
    ax1.invert_xaxis()
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    report.style(ax1, "Cumulative share of final sales", "days to event", "share of final")

    for name, ids in segments.groupby(segments).groups.items():
        rows = share.loc[share.index.intersection(ids)]
        colour = report.BRAND_COLOURS.get(str(name).split(" / ")[0], report.GREY)
        ax2.plot(
            days,
            rows.median().to_numpy(),
            linewidth=1.8,
            label=f"{name} (n={len(rows)})",
            color=colour,
            linestyle="--" if "not sellout" in str(name) else "-",
        )
    for i, day in enumerate(FORECAST_DAYS):
        ax2.axvline(
            day,
            color="#b03030",
            linewidth=0.8,
            alpha=0.5,
            label="decision days (-7/-3/-1)" if i == 0 else None,
        )
    ax2.invert_xaxis()
    ax2.legend(fontsize=7, frameon=False, loc="upper left")
    report.style(ax2, "Median curve by segment", "days to event", "share of final")

    return report.save_fig(fig, out_dir, "phase3_curves.png")


def fig_forecast_error(metrics: pd.DataFrame, out_dir: Path) -> Path:
    """Held-out forecast error at each decision day, against both baselines."""
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.27, metrics["nosale_MAPE_pct"], 0.27, label="assume no more sales", color="#bbb")
    ax.bar(x, metrics["pooled_MAPE_pct"], 0.27, label="pooled median curve", color="#999")
    ax.bar(
        x + 0.27, metrics["segment_MAPE_pct"], 0.27, label="segment median curve", color="#3d6bb3"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"day {d}" for d in metrics["day"]])
    ax.legend(fontsize=8, frameon=False)
    report.style(ax, "Held-out forecast error from partial presale", "", "MAPE (%)")
    return report.save_fig(fig, out_dir, "phase3_forecast_error.png")


def fig_forecast_scatter(held: pd.DataFrame, out_dir: Path) -> Path:
    """Forecast vs actual at each decision day, held out."""
    fig, axes = plt.subplots(1, len(FORECAST_DAYS), figsize=(12, 4), sharex=True, sharey=True)
    for ax, day in zip(axes, FORECAST_DAYS, strict=True):
        sub = held[held["day"] == day]
        ax.scatter(sub["actual"], sub["forecast"], s=18, alpha=0.7, color="#3d6bb3")
        lo, hi = sub["actual"].min() * 0.7, sub["actual"].max() * 1.4
        ax.plot([lo, hi], [lo, hi], color="#b03030", linestyle="--", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        report.log_ticks(ax)
        report.style(ax, f"day -{day}", "actual final tickets", "forecast")
    return report.save_fig(fig, out_dir, "phase3_forecast_scatter.png")


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def build(run: phaseio.Run) -> list[Path]:
    out_dir = run.out_dir
    events, transactions = run.events, run.transactions
    ev = live(event_outcomes(events, transactions))

    cum, share, final = cumulative_sales(events, transactions)
    ev = ev[ev["event_id"].isin(share.index)]

    brand_segments = segment_labels(ev, ["brand"])
    sellout_segments = segment_labels(ev, ["brand", "sellout"])

    metrics, held = evaluate(cum, share, final, brand_segments)
    _, fell_back = median_curves(share, sellout_segments)

    # How much a partial presale is worth OVER the pricing-time model: same events, same
    # grouped-by-event discipline, so the two numbers are comparable.
    features = build_features(events, transactions)
    _, oof = cross_validate(features)
    phase2_scores = scores(oof["tickets"], oof["predicted"])

    figures = [
        fig_curves(share, sellout_segments, out_dir),
        fig_forecast_error(metrics, out_dir),
        fig_forecast_scatter(held, out_dir),
    ]

    day7 = metrics[metrics["day"] == "-7"].iloc[0]
    day1 = metrics[metrics["day"] == "-1"].iloc[0]

    lines = [
        phaseio.provenance_header(
            "Phase 3 — Sales curve",
            run.source,
            run.seed,
            phaseio.regenerate_command("phase3", run.source, run.seed),
        ),
        "",
    ]
    lines += [
        "How tickets accumulate against days-to-event, and what a partial presale forecasts.",
        "Empirical median curves — no Bass diffusion, no fitted parameters, nothing to tune.",
        "",
        "## Headline",
        "",
        (
            f"- Median event has sold **{day7['median_share_sold']:.0%} of its final total with a "
            f"week to go**, {day1['median_share_sold']:.0%} with one day to go."
        ),
        (
            f"- Forecasting from the day -7 presale: **MAE {day7['segment_MAE']:.1f} tickets, "
            f"MAPE {day7['segment_MAPE_pct']:.1f}%** on held-out events."
        ),
        (
            f"- By day -1 that tightens to MAE {day1['segment_MAE']:.1f} tickets, "
            f"MAPE {day1['segment_MAPE_pct']:.1f}%."
        ),
        "",
        "### Is watching the presale worth anything?",
        "",
        "Same events, same grouped-by-event discipline, so the rows are comparable:",
        "",
        report.md_table(_comparison(metrics, phase2_scores), floatfmt="{:,.1f}"),
        "",
        _overtake_sentence(metrics, phase2_scores),
        "",
        "---",
        "",
        "## 1. Curve shape",
        "",
        (
            f"Measured over {len(share)} events that ran, {MAX_LEAD} days out to the door. "
            "Comps excluded (SPEC.md §8.7); cancelled events excluded, because a pulled event's "
            "final total is not a final total (SPEC.md §8.4)."
        ),
        "",
        report.md_table(spread_table(share), floatfmt="{:,.3f}"),
        "",
        "`p90_minus_p10` is the assumption of this phase, measured. The method says an",
        "event's presale is shaped like the median of its segment; that column says how far",
        "from the median the middle 80% of events actually sit, and it is wide — the shape",
        "assumption is doing real work and is only roughly true.",
        "",
        "Note it does **not** shrink steadily as the event approaches. What improves is how",
        "much of the answer is already banked: with a week to go the median event has sold",
        (
            f"{day7['median_share_sold']:.0%} of its final total and by day -1 "
            f"{day1['median_share_sold']:.0%}, so the forecast is scaling up a smaller and"
        ),
        "smaller remainder. The accuracy comes from the numerator, not from the curve.",
        "",
        "## 2. Segments",
        "",
        report.md_table(segment_curve_table(share, sellout_segments), floatfmt="{:,.3f}"),
        "",
        "**The `sellout` split is descriptive only and must never be used to forecast.** On",
        "day -7 nobody knows which column an event belongs in — that is the answer, not an",
        "input. It is shown because whether a sold-out night's curve is shaped differently is",
        "worth knowing, and because writing down why it cannot be used is the whole lesson.",
        "",
        (
            f"Segments below {MIN_EVENTS_PER_SEGMENT} events fall back to the pooled curve: "
            f"{', '.join(fell_back)}."
            if fell_back
            else f"No segment fell below the {MIN_EVENTS_PER_SEGMENT}-event floor."
        ),
        "",
        "The forecaster below segments on **brand only**, which is known months in advance.",
        "",
        "## 3. Held-out forecast error",
        "",
        "Grouped 5-fold by event (SPEC.md §8.1): each fold's median curve is built from the",
        "other four folds, so no event is forecast by a curve it helped draw.",
        "",
        report.md_table(
            metrics[
                [
                    "day",
                    "events",
                    "median_share_sold",
                    "segment_MAE",
                    "segment_MAPE_pct",
                    "pooled_MAE",
                    "pooled_MAPE_pct",
                    "nosale_MAE",
                    "nosale_MAPE_pct",
                ]
            ],
            floatfmt="{:,.2f}",
        ),
        "",
        "- `segment` — brand median curve, the forecaster.",
        "- `pooled` — one curve for the whole programme. Read the gap honestly: if it is",
        "  small, brand segmentation is buying nothing and should be dropped.",
        "- `nosale` — assume not one more ticket sells. The floor any curve must clear.",
        "",
        _segmentation_sentence(metrics),
        "",
        "## 4. What can go wrong",
        "",
        "- **A tier drop after the forecast day breaks the shape.** The curve is a median",
        "  over events whose ladders opened on their own schedules; an event that releases a",
        "  new tier at day -5 gets a jump the median does not have. Ladder timing is in",
        "  `event_tier.lead_time_open_days` and conditioning the curve on it is the obvious",
        "  next refinement.",
        "- **Sellouts truncate.** A night that sells out at day -4 has a curve that flattens",
        "  because there is nothing left to sell, not because demand stopped. Scaling those",
        "  curves up estimates *sales*, never *demand* — the difference matters the moment",
        "  Phase 6 asks what a higher price would have earned.",
        "- **Small segments.** With 5 folds and 2 brands the thinnest training segment here",
        f"  is around {int(len(share) * 0.8 / 2)} events; the pooled fallback exists for real",
        "  data, where a brand-venue-year segmentation would be very thin indeed.",
        "- **The comparison with Phase 2 is not apples to apples.** Phase 2 forecasts before",
        "  a single ticket is sold; this forecasts with most of them already sold. The right",
        "  reading is 'how much does watching the presale add', not 'which model is better'.",
        "",
        "---",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{p.name}`" for p in figures]
    lines += ["", "".join(f"![{p.stem}]({p.name})\n" for p in figures)]

    for _, row in metrics.iterrows():
        print(
            f"day {row['day']:>3}  sold {row['median_share_sold']:.0%}  "
            f"segment MAE {row['segment_MAE']:6.1f} MAPE {row['segment_MAPE_pct']:5.1f}%   "
            f"pooled MAPE {row['pooled_MAPE_pct']:5.1f}%   "
            f"no-more-sales MAPE {row['nosale_MAPE_pct']:5.1f}%"
        )
    print(f"phase 2 (no presale) MAPE {phase2_scores['MAPE_pct']:.1f}% on the same events")

    return [phaseio.write_report(lines, "phase3_sales_curve.md", out_dir), *figures]


def main(argv: list[str] | None = None) -> int:
    return phaseio.run_phase(build, "pricing.phase3", __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
