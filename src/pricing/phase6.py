"""Phase 6 — the counterfactual: what should I have charged? SPEC.md 6.4.

    uv run python -m pricing.phase6 --synthetic
    uv run python -m pricing.phase6 --real          <- refused while the prereg is DRAFT

The headline number of the whole project — **and the one most likely to be a lie.**

What it does
------------
Take each event's actual tier ladder. Re-price it two ways at once:

    level      multiply every rung by the same factor        (0.70 .. 1.30)
    spread     stretch or flatten the gaps between the rungs (0.0 .. 1.50, 1.0 = as it was)

Walk a **visible grid** over those two knobs — no optimiser, no solver, nothing you cannot
print. At each grid point, move quantity with the elasticity from Phase 4:

    units_new = units_actual * exp(beta * change in log price)

cap the event's total at its capacity (you cannot sell a room you do not have), and add up
the contribution. Each event keeps its own best grid point. Compare with what it actually
made. That difference is the projected uplift.

The honesty rule that governs this file
---------------------------------------
**If Phase 4's verdict is NOT-IDENTIFIED, the headline uplift number does not exist.**

Not "is uncertain". Does not exist. An uplift is beta multiplied by a price change; if beta
is not identified then the number is an assumption wearing a percentage sign, and putting
it on a CV is the same sin as a fabricated Sharpe ratio (SPEC.md 6.4, 8.3). So in that case
this module reports an **uplift curve over assumed elasticities** instead, every row of it
labelled with the assumption it rests on, and points at `experiment/DESIGN.md` — the
running randomised experiment — as the thing that pins beta.

When beta IS identified, the point uplift is reported, still labelled **projected** and
never realised, and on synthetic data it is checked against the generator's true demand
process.

Two results this method produces that are arithmetic, not discoveries
---------------------------------------------------------------------
Both are stated in the report rather than dressed up, because a reader who spots them
before you do will assume you did not understand your own model:

1. **With |beta| < 1 the optimum is always the top of the grid.** Constant-elasticity
   revenue is `price^(1+beta)`, so if demand is inelastic, revenue rises with price without
   limit and the model wants infinity. The grid bound is therefore an assumption about
   what is commercially sane, not a result. It is stated as one.
2. **The ladder's shape is chosen by the same single number.** With `|beta| < 1` the
   objective is concave in log price, so the model wants to FLATTEN the ladder to one
   price; with `|beta| > 1` it is convex and the model wants to STRETCH it as far as the
   grid allows. Neither is advice. One beta for every rung means tiers have no separate
   willingness to pay, and early-bird buyers and door buyers are different people that
   this model cannot tell apart.

Method policy (DECISIONS.md, 2026-08-11 amendment): a printed grid, plain pandas, no
optimiser.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pricing import phase4, phase5, phaseio

REPORT_FILENAME = "phase6_counterfactual.md"

# --------------------------------------------------------------------------------------
# The grid. Coarse and visible on purpose — you can read every candidate off this page.
# --------------------------------------------------------------------------------------
#
# LEVEL_GRID  multiplies every rung of the ladder. +/-30% is the bound, and it is a
#             judgement about what a student night can be re-priced by without becoming a
#             different product, NOT something the data implies. See honesty note 1 above.
# SPREAD_GRID multiplies each rung's distance from the ladder's own average, in logs.
#             1.0 leaves the ladder as it was; 0.0 collapses it to a single price; 1.5
#             stretches Early Bird further below the door price.
LEVEL_GRID: tuple[float, ...] = tuple(round(0.70 + 0.05 * i, 2) for i in range(13))
SPREAD_GRID: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)

# The elasticities the uplift curve is reported over when beta is not identified — and as
# a sensitivity band when it is. -0.4 is "raising prices is nearly free"; -2.0 is "the room
# empties". Real live-event estimates in the literature mostly sit between -0.5 and -1.5.
ASSUMED_BETAS: tuple[float, ...] = (-0.4, -0.6, -0.8, -1.0, -1.2, -1.5, -2.0)

# The experiment's dose (experiment/DESIGN.md): +/-15% on every rung. Reported separately
# because it is a price change the business has already decided it can live with, so the
# uplift at that dose is a number with a use rather than a corner solution.
EXPERIMENT_DOSE = 1.15

# Capacity is a hard constraint: a licensed capacity is a legal number, not a soft target.
MAX_SELL_THROUGH = 1.00


# --------------------------------------------------------------------------------------
# 1. The panel and the counterfactual arithmetic
# --------------------------------------------------------------------------------------


def build_panel(event_tier: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Tier-level rows with everything the counterfactual needs.

    Same filters as Phase 4 (`phase4.build_panel`) so the ladder being re-priced is the
    ladder beta was estimated from — cancelled events out, comps out, zero-sale tiers out,
    single-tier events out.

    `face_price` is what the promoter banks per ticket; `price` is what the buyer sees.
    Demand responds to the buyer price and the till receives the face price, so the
    counterfactual moves both by the same proportion and keeps them apart in the
    arithmetic. While `fee_treatment` is UNKNOWN (SPEC.md 8.6) that split is a guess.

    Contribution means what **Phase 5** says it means (`contribution_per_ticket`): face
    value less any per-head cost. The constant is 0.00 today, so subtracting it changes
    nothing — and that is exactly why it has to be here rather than assumed away. The day
    a real per-head levy lands in the cost stack, two phases using the same word for two
    different numbers is the kind of disagreement nobody notices (review finding,
    2026-08-12).
    """
    panel = phase4.build_panel(event_tier, events)
    panel = panel[panel["face_price"] > 0].copy()
    panel["log_face"] = np.log(panel["face_price"])
    panel["mean_log_p"] = panel.groupby("event_id")["log_p"].transform("mean")
    panel["actual_contribution"] = panel["units_sold"] * _contribution_per_ticket(
        panel["face_price"].to_numpy()
    )
    return panel.reset_index(drop=True)


def _contribution_per_ticket(face_price: np.ndarray) -> np.ndarray:
    """Phase 5's definition, imported rather than retyped: face value less per-head cost."""
    return face_price - phase5.VARIABLE_COST_PER_TICKET


def ladder_change(panel: pd.DataFrame, level: float, spread: float) -> np.ndarray:
    """Change in log price for every tier, for one grid point.

        delta = log(level) + (spread - 1) * (log P_t - mean log P_e)

    `level` moves the whole ladder; `spread` re-scales each rung's distance from its own
    ladder's average. At (1.0, 1.0) every delta is zero and the counterfactual is the
    thing that actually happened, which is the arithmetic check that the machinery is not
    quietly adding something.
    """
    deviation = panel["log_p"].to_numpy() - panel["mean_log_p"].to_numpy()
    return np.log(level) + (spread - 1.0) * deviation


def counterfactual_contribution(
    panel: pd.DataFrame, delta: np.ndarray, beta: float, max_sell_through: float = MAX_SELL_THROUGH
) -> pd.Series:
    """Contribution per event under a re-priced ladder. The model side of Phase 6.

    Three lines of economics:

    1. **Quantity moves with the elasticity.** `units_new = units * exp(beta * delta)`.
       Constant elasticity, applied tier by tier. This is the only place beta is used.
    2. **The till moves with the price.** The face value scales by the same proportion as
       the buyer price, so contribution per ticket is `face * exp(delta)` less Phase 5's
       per-head cost — the same definition of "contribution" as Phase 5, by import.
    3. **Capacity binds.** If the re-priced event would sell more than the room holds,
       every tier is scaled back proportionally. Rationing, and it is what actually
       happens: the last tier stops selling.

    What it deliberately does NOT model: buyers moving between tiers when the gaps change
    (that needs a tier-specific elasticity, which this data cannot support), the sales
    curve running out of time, or anyone deciding the night is now too expensive to be
    the night they go to. See the report's limitations section.
    """
    units_new = panel["units_sold"].to_numpy() * np.exp(beta * delta)
    face_new = panel["face_price"].to_numpy() * np.exp(delta)

    frame = pd.DataFrame(
        {
            "event_id": panel["event_id"].to_numpy(),
            "units_new": units_new,
            "face_new": face_new,
            "capacity": panel["capacity"].to_numpy(dtype=float),
        }
    )
    total = frame.groupby("event_id")["units_new"].transform("sum")
    room = frame["capacity"] * max_sell_through
    frame["units_new"] = frame["units_new"] * np.minimum(1.0, room / total.replace(0, np.nan))
    frame["contribution"] = frame["units_new"] * _contribution_per_ticket(
        frame["face_new"].to_numpy()
    )
    return frame.groupby("event_id")["contribution"].sum()


def grid_search(panel: pd.DataFrame, beta: float) -> pd.DataFrame:
    """Every grid point, evaluated per event. The visible search.

    Returns one row per (event x level x spread). Nothing is hidden inside an optimiser:
    the whole search space is a table you can sort, print or argue with, and at this size
    (13 levels x 7 spreads) it costs a second.
    """
    frames = []
    for level in LEVEL_GRID:
        for spread in SPREAD_GRID:
            contribution = counterfactual_contribution(
                panel, ladder_change(panel, level, spread), beta
            )
            frames.append(
                pd.DataFrame(
                    {
                        "event_id": contribution.index,
                        "level": level,
                        "spread": spread,
                        "contribution": contribution.to_numpy(),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def optimise(panel: pd.DataFrame, beta: float) -> dict:
    """Each event's best grid point, and the programme-level uplift that follows.

    Each event is optimised on its own: the answer is a schedule per night, not one
    multiplier for the business. The uplift is the ratio of the two totals, which is a
    weighted average that lets the big rooms count for more — as they should, since that
    is where the money is.
    """
    actual = panel.groupby("event_id")["actual_contribution"].sum()
    grid = grid_search(panel, beta)
    best_pos = grid.groupby("event_id")["contribution"].idxmax()
    best = grid.loc[best_pos].set_index("event_id")

    joined = pd.DataFrame(
        {
            "actual": actual,
            "best": best["contribution"],
            "level": best["level"],
            "spread": best["spread"],
        }
    ).dropna()
    joined["uplift"] = joined["best"] / joined["actual"] - 1.0

    total_actual = float(joined["actual"].sum())
    total_best = float(joined["best"].sum())
    at_boundary = float(
        joined["level"].isin([min(LEVEL_GRID), max(LEVEL_GRID)]).mean()
        if len(joined)
        else float("nan")
    )
    return {
        "beta": beta,
        "per_event": joined,
        "total_actual": total_actual,
        "total_best": total_best,
        "uplift": total_best / total_actual - 1.0 if total_actual > 0 else float("nan"),
        "median_event_uplift": float(joined["uplift"].median()),
        "share_at_level_boundary": at_boundary,
        "modal_level": float(joined["level"].mode().iloc[0]) if len(joined) else float("nan"),
        "modal_spread": float(joined["spread"].mode().iloc[0]) if len(joined) else float("nan"),
    }


def uplift_at(panel: pd.DataFrame, beta: float, level: float, spread: float = 1.0) -> float:
    """Uplift from one stated ladder change applied to every event. No optimisation.

    This is the number that survives the honesty rules: a price change decided in advance
    (the experiment's +15%, say), evaluated at a stated beta. No search, so no corner
    solution, so nothing to argue about except beta.
    """
    actual = float(panel["actual_contribution"].sum())
    new = float(counterfactual_contribution(panel, ladder_change(panel, level, spread), beta).sum())
    return new / actual - 1.0 if actual > 0 else float("nan")


def uplift_curve(panel: pd.DataFrame, betas=ASSUMED_BETAS) -> pd.DataFrame:
    """The uplift as a function of the elasticity you are willing to assume.

    This is the honest deliverable when beta is not identified. It is not a hedge: it says
    exactly what the counterfactual is worth under each assumption, and leaves the reader
    to hold the assumption rather than swallowing one silently.
    """
    rows = []
    for beta in betas:
        result = optimise(panel, beta)
        rows.append(
            {
                "assumed_beta": beta,
                "uplift": result["uplift"],
                "median_event_uplift": result["median_event_uplift"],
                "modal_level": result["modal_level"],
                "modal_spread": result["modal_spread"],
                "share_at_level_boundary": result["share_at_level_boundary"],
                "uplift_at_experiment_dose": uplift_at(panel, beta, EXPERIMENT_DOSE),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 2. The ground-truth check (synthetic only)
# --------------------------------------------------------------------------------------


def true_process_contribution(
    panel: pd.DataFrame, delta: np.ndarray, truth: dict, max_sell_through: float | None = None
) -> pd.Series:
    """Re-run the counterfactual ladder through the GENERATOR'S OWN demand process.

    This is the check that the model's simplification of the world does not change the
    answer. The generator does not move each tier independently: it allocates the event's
    total across tiers with a softmax of the tier terms, and moves the total with the
    event's AVERAGE log price. So:

        new share  ∝ old share * exp(beta_true * delta_t)      (softmax, renormalised)
        new total   = old total * exp(beta_true * mean(delta)) (capped at the room)
        new units   = max(3, round(new total * new share))     (the generator's floor)

    The tier-level noise never has to be recovered, because it is already inside the
    observed shares — which is why this is three lines rather than a re-simulation.

    For a UNIFORM price change the two processes agree exactly, and that is worth knowing:
    the check only has teeth where the ladder's SHAPE changes, which is precisely where a
    per-tier elasticity is the strongest assumption the model makes.
    """
    beta_true = float(truth["beta_true"])
    if max_sell_through is None:
        max_sell_through = float(truth["params"].get("max_sellthrough", MAX_SELL_THROUGH))

    frame = pd.DataFrame(
        {
            "event_id": panel["event_id"].to_numpy(),
            "units": panel["units_sold"].to_numpy(dtype=float),
            "face_new": panel["face_price"].to_numpy() * np.exp(delta),
            "capacity": panel["capacity"].to_numpy(dtype=float),
            "delta": delta,
        }
    )
    total = frame.groupby("event_id")["units"].transform("sum")
    share = frame["units"] / total
    raw = share * np.exp(beta_true * frame["delta"])
    new_share = raw / raw.groupby(frame["event_id"]).transform("sum")

    mean_delta = frame.groupby("event_id")["delta"].transform("mean")
    new_total = np.minimum(
        total * np.exp(beta_true * mean_delta), frame["capacity"] * max_sell_through
    )
    frame["units_true"] = np.maximum(3.0, np.round(new_total * new_share))
    frame["contribution"] = frame["units_true"] * frame["face_new"]
    return frame.groupby("event_id")["contribution"].sum()


def ground_truth_check(panel: pd.DataFrame, truth: dict, best: dict) -> dict:
    """Model-implied uplift vs the uplift the true process would actually have delivered.

    Applies each event's OWN chosen ladder (the grid point the model picked) and asks what
    the generator's demand process would have done with it. Agreement in sign and rough
    magnitude is the pass condition — not agreement to three decimal places, which would
    only prove that two nearly identical formulas are nearly identical.

    What it proves: the constant-elasticity, tier-by-tier approximation does not throw the
    answer away. What it does not prove: that the real world is a constant-elasticity
    world. Nothing run on this fixture can prove that.
    """
    per_event = best["per_event"]
    lookup = per_event[["level", "spread"]]
    joined = panel.merge(lookup, left_on="event_id", right_index=True, how="inner")

    delta = np.log(joined["level"].to_numpy()) + (joined["spread"].to_numpy() - 1.0) * (
        joined["log_p"].to_numpy() - joined["mean_log_p"].to_numpy()
    )
    true_contribution = true_process_contribution(joined, delta, truth)
    actual = joined.groupby("event_id")["actual_contribution"].sum()

    true_total = float(true_contribution.sum())
    actual_total = float(actual.sum())
    true_uplift = true_total / actual_total - 1.0
    model_uplift = best["uplift"]
    return {
        "model_uplift": model_uplift,
        "true_uplift": true_uplift,
        "same_sign": bool(np.sign(model_uplift) == np.sign(true_uplift)),
        "ratio": model_uplift / true_uplift if true_uplift != 0 else float("nan"),
        "beta_used": best["beta"],
        "beta_true": float(truth["beta_true"]),
    }


# --------------------------------------------------------------------------------------
# 3. Run everything
# --------------------------------------------------------------------------------------


def run(run_data: phaseio.Run) -> dict:
    """Phase 6, which begins by re-running Phase 4 because it cannot proceed without it."""
    elasticity = phase4.run(run_data)
    panel = build_panel(run_data.event_tier, run_data.events)

    beta_hat = float(elasticity["beta"])
    identified = bool(elasticity["identified"])

    best = optimise(panel, beta_hat)
    curve = uplift_curve(panel)

    check = None
    if run_data.truth is not None:
        check = ground_truth_check(panel, run_data.truth, best)

    return {
        "source": run_data.source,
        "seed": run_data.seed,
        "panel": panel,
        "elasticity": elasticity,
        "beta_hat": beta_hat,
        "identified": identified,
        "verdict": elasticity["verdict"].verdict,
        "best": best,
        "curve": curve,
        "uplift_at_experiment_dose": uplift_at(panel, beta_hat, EXPERIMENT_DOSE),
        "ground_truth_check": check,
        "n_events": int(panel["event_id"].nunique()),
        "n_tiers": len(panel),
    }


# --------------------------------------------------------------------------------------
# 4. The report
# --------------------------------------------------------------------------------------


def _wants(modal_level: float) -> str:
    """What the model does with prices at one assumed beta. Used in the table and the prose."""
    if modal_level > 1.0:
        return "raise prices"
    return "cut prices" if modal_level < 1.0 else "leave them"


def _contradiction_sentence(curve: pd.DataFrame) -> str:
    """The 'these two rows are contradictory advice' sentence, read OFF the curve.

    This used to be two percentages typed into the prose by hand. They were right on one
    seed and wrong on the next — the report printed a number that its own table two
    paragraphs above disagreed with (review finding, 2026-08-12). In a report whose entire
    selling point is that every number regenerates, a hand-typed number in prose is the
    worst defect available, so the two rows are pulled out of the curve like everything
    else.

    The rows chosen are the extremes of the assumed range: the most elastic beta and the
    least elastic one. They are the pair that makes the point hardest, and they are chosen
    by a rule rather than by which pair looked most striking.
    """
    elastic = curve.loc[curve["assumed_beta"].idxmin()]
    inelastic = curve.loc[curve["assumed_beta"].idxmax()]
    return (
        f"So the {elastic['uplift']:+.1%} uplift at beta = {elastic['assumed_beta']:+.1f} "
        f"(which gets there by telling you to {_wants(elastic['modal_level'])}) and the "
        f"{inelastic['uplift']:+.1%} uplift at beta = {inelastic['assumed_beta']:+.1f} "
        f"(which gets there by telling you to {_wants(inelastic['modal_level'])}) are not the "
        "same claim with different confidence: they are contradictory pieces of advice that "
        "happen to carry a similar-looking number."
    )


def _curve_table(curve: pd.DataFrame) -> str:
    lines = [
        (
            "| assumed beta | optimised uplift | median event uplift | at the experiment's +15% | "
            "modal level | what it wants | modal spread | at the grid edge |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in curve.iterrows():
        want = _wants(r["modal_level"])
        lines.append(
            f"| {r['assumed_beta']:+.1f} | **{r['uplift']:+.1%}** | "
            f"{r['median_event_uplift']:+.1%} | {r['uplift_at_experiment_dose']:+.1%} | "
            f"x{r['modal_level']:.2f} | {want} | x{r['modal_spread']:.2f} | "
            f"{r['share_at_level_boundary']:.0%} |"
        )
    return "\n".join(lines)


def _grid_optimum_block(results: dict) -> list[str]:
    """The grid optimum, demoted to what it is: a bound set by where the grid stops.

    Review finding, 2026-08-12: this number used to be the `## Headline:` line, and on the
    fixture 100% of events pick the top corner of the level grid — so the headline was a
    restatement of "the grid stops at x1.30", not a result. The argument was already in the
    report body; the headline just did not match it. Now the pre-committed dose leads and
    this sits underneath, worded as a bound whenever the corner binds.
    """
    best = results["best"]
    share, modal = best["share_at_level_boundary"], best["modal_level"]
    tag = "" if results["identified"] else "PROVISIONAL — "
    if share > 0:
        reading = (
            f"{tag}**Corner solution — this is the grid bound, not a result.** "
            f"{share:.0%} of events pick a level at the edge of the grid (modal level "
            f"x{modal:.2f}), which is what `price^(1+beta)` does when demand is inelastic: "
            f"revenue rises with price without limit and the model stops only where the grid "
            f"stops. Read it as *at least {best['uplift']:+.1%} within the "
            f"x{min(LEVEL_GRID):.2f}–x{max(LEVEL_GRID):.2f} ladder range considered*, and note "
            f"that the range is a judgement about what a student night can be re-priced by, not "
            f"something the data implies. Widening the grid moves this number and nothing else."
        )
    else:
        reading = (
            f"{tag}every event's chosen level is interior to the grid (modal level x{modal:.2f}), "
            f"so the bound is not what produced this number."
        )
    return [
        "### Sensitivity: what an unconstrained search over the grid would have picked",
        "",
        (
            f"Optimising each event over all {len(LEVEL_GRID) * len(SPREAD_GRID)} candidate "
            f"ladders gives {tag}{best['uplift']:+.1%}."
        ),
        "",
        reading,
    ]


def _headline(results: dict) -> list[str]:
    if results["identified"]:
        return [
            (
                f"## Headline: **{results['uplift_at_experiment_dose']:+.1%} projected uplift in "
                f"contribution margin** at the experiment's pre-committed +"
                f"{EXPERIMENT_DOSE * 100 - 100:.0f}% dose"
            ),
            "",
            (
                f"At the Phase 4 elasticity of **{results['beta_hat']:+.3f}** (verdict: "
                f"{results['verdict']}), moving every rung of every ladder by "
                f"+{EXPERIMENT_DOSE * 100 - 100:.0f}% — the dose the business has already agreed "
                f"it can live with (`experiment/DESIGN.md`) — would have changed contribution by "
                f"{results['uplift_at_experiment_dose']:+.1%} across {results['n_events']} events."
            ),
            "",
            (
                "**This is the headline because it is the only uplift here with no search in it.** "
                "One price change, decided in advance, evaluated once. The optimised number below "
                "is larger and means less: with an inelastic estimate the optimiser walks to "
                "whatever bound the grid sets, so it reports the bound."
            ),
            "",
            (
                "**Projected, not realised.** Nobody was charged these prices. It is what a model "
                "says would have happened, and the difference between that and a realised number "
                "is the difference between a research finding and a fabrication (SPEC.md 6.4). The "
                "CV wording is *projected*, and it says so before anyone asks."
            ),
            "",
            *_grid_optimum_block(results),
        ]
    best = results["best"]
    return [
        "## Headline: **there is no headline uplift number**",
        "",
        (
            f"Phase 4's verdict is **{results['verdict']}**. An uplift is an elasticity multiplied "
            "by a price change, so an uplift computed from an unidentified elasticity is an "
            "assumption wearing a percentage sign. Reporting it as a finding would be the same sin "
            "as a fabricated Sharpe ratio."
        ),
        "",
        (
            "What exists instead is a **curve**: the uplift, conditional on each elasticity you "
            "might be willing to assume. Every row below is labelled with the assumption it rests "
            "on. **The running randomised experiment (`experiment/DESIGN.md`) is what pins beta** — "
            "whole events assigned a HIGH (x1.15) or LOW (x0.85) ladder by a pre-committed "
            "cryptographic coin, which produces price variation the operator's demand expectations "
            "did not cause. When it reads out, one row of this curve becomes the answer."
        ),
        "",
        (
            f"The point estimate at the fixed-effects beta of {results['beta_hat']:+.3f} is "
            f"**{best['uplift']:+.1%}** — reported here as **PROVISIONAL**, conditional on an "
            "observational estimate that did not clear its own identification criteria, and not to "
            "be quoted as a finding."
        ),
    ]


def _schedule_block(results: dict) -> list[str]:
    """The per-schedule numbers — and the qualifier that has to travel with them.

    Review finding, 2026-08-12: the NOT-IDENTIFIED report correctly refused a headline
    ("there is no headline uplift number"), correctly labelled the point estimate
    PROVISIONAL — and then, forty lines later, printed `programme uplift: **+5.3%**` in
    bold with no qualifier at all. A number in bold survives a copy-paste; the section
    heading it sat under does not. So in that branch the bold comes off and the word
    PROVISIONAL goes on the same line as every uplift percentage, and there is a test that
    scans the rendered report for a bold percentage on an unlabelled line.
    """
    best = results["best"]
    identified = results["identified"]
    tag = "" if identified else "PROVISIONAL — "
    heading = f"## What the schedule actually looks like at beta = {results['beta_hat']:+.3f}"
    if not identified:
        heading += " (PROVISIONAL — this beta did not clear its identification criteria)"

    def number(value: float) -> str:
        return f"**{value:+.1%}**" if identified else f"{value:+.1%}"

    return [
        heading,
        "",
        f"- total actual contribution: £{best['total_actual']:,.0f}",
        f"- total counterfactual contribution: £{best['total_best']:,.0f}",
        (
            f"- {tag}programme uplift at the grid optimum — a bound, see the headline: "
            f"{number(best['uplift'])} (median event {best['median_event_uplift']:+.1%})"
        ),
        (
            f"- modal chosen ladder: level x{best['modal_level']:.2f}, spread "
            f"x{best['modal_spread']:.2f}"
            f" ({best['share_at_level_boundary']:.0%} of events at the grid edge)"
        ),
        (
            f"- {tag}THE HEADLINE NUMBER — uplift from the experiment's +15% dose alone, no "
            f"optimisation: {number(results['uplift_at_experiment_dose'])}"
        ),
        "",
    ]


def report(results: dict, run_data: phaseio.Run) -> str:
    regenerate = phaseio.regenerate_command("phase6", results["source"], results["seed"])
    best = results["best"]
    check = results["ground_truth_check"]

    parts = [
        phaseio.provenance_header(
            "Phase 6 — Counterfactual pricing backtest", run_data.source, run_data.seed, regenerate
        ),
        "",
        (
            "**The question:** what would an optimal tiered release schedule have earned across "
            "these events, versus what was actually charged?"
        ),
        "",
        *_headline(results),
        "",
        "## The grid (there is no optimiser here)",
        "",
        (
            f"- **level** — multiply every rung by the same factor: "
            f"`{LEVEL_GRID[0]:.2f} .. {LEVEL_GRID[-1]:.2f}` in steps of 0.05 "
            f"({len(LEVEL_GRID)} values)"
        ),
        (
            f"- **spread** — multiply each rung's distance from its own ladder's average, in logs: "
            f"`{', '.join(f'{s:.2f}' for s in SPREAD_GRID)}` (1.00 leaves the ladder as it was, "
            "0.00 collapses it to a single price)"
        ),
        (
            f"- {len(LEVEL_GRID) * len(SPREAD_GRID)} candidate schedules per event, "
            f"{results['n_events']} events, {results['n_tiers']} tier rows. Every candidate is a "
            "row in a table you can print."
        ),
        "",
        (
            "At each candidate: quantity moves as `units * exp(beta * change in log price)`, the "
            "face value moves by the same proportion as the buyer price, and the event's total is "
            "capped at its licensed capacity with every tier scaled back proportionally. At "
            "(level 1.00, spread 1.00) every change is zero and the counterfactual reproduces what "
            "actually happened — which is the arithmetic check that the machinery is not quietly "
            "adding something."
        ),
        "",
        "## The uplift curve",
        "",
        _curve_table(results["curve"]),
        "",
        (
            "**Read the 'what it wants' column before the uplift column.** The curve is "
            "U-shaped, and it touches zero at beta = -1.0 for a reason that is pure arithmetic: at "
            "unit elasticity, revenue does not respond to the level of prices at all, so there is "
            "nothing to gain. Either side of that point the model wants to move prices in "
            "**opposite directions** — raise them if demand is inelastic, cut them if it is "
            "elastic. " + _contradiction_sentence(results["curve"]) + " That is exactly why an "
            "unidentified beta cannot produce a headline number, and why the experiment matters "
            "more than any estimator choice."
        ),
        "",
        "## Two things in this table that are arithmetic, not discoveries",
        "",
        (
            f"**1. The optimum sits on the edge of the grid.** At the Phase 4 beta, "
            f"{best['share_at_level_boundary']:.0%} of events pick a level at the grid boundary. "
            "That is not a discovery, it is `price^(1+beta)`: when demand is inelastic "
            "(`|beta| < 1`) revenue rises with price without limit, so the model wants infinity "
            "and stops only where the grid stops. **The +/-30% bound is an assumption about what a "
            "student night can be re-priced by without becoming a different product — it is not "
            "something the data implies.** The model has no mechanism for a brand becoming known as "
            "expensive, for a competitor undercutting, or for the bar spend of the people who "
            f"stopped coming. That is why the '+15%' column exists: it is a dose the business has "
            "already agreed it can live with (`experiment/DESIGN.md`), so the uplift there is a "
            "number with a use rather than a corner solution."
        ),
        "",
        (
            f"**2. The ladder's shape is chosen by the same single number** — here the model wants "
            f"to {'flatten the ladder' if best['modal_spread'] < 1 else 'stretch the ladder'} "
            f"(modal spread x{best['modal_spread']:.2f}). Also arithmetic: with `|beta| < 1` the "
            "objective is concave in log price, so spreading prices apart can only lose money and "
            "the model collapses the ladder to one price; with `|beta| > 1` it is convex and the "
            "model stretches the ladder as far as the grid allows. Neither is advice. One beta for "
            "every rung means tiers have no separate willingness to pay, and early-bird buyers and "
            "door buyers are different people this model cannot tell apart. **Do not read the "
            "flat-ladder result as advice to abolish early-bird pricing.** Read it as the model "
            "naming the question it cannot answer — separating tier-level elasticities needs more "
            "within-event price variation than exists (SPEC.md 4.3, 5.6)."
        ),
        "",
        *_schedule_block(results),
    ]

    if check is not None:
        agree = "agree" if check["same_sign"] else "**DISAGREE**"
        parts += [
            "## Ground-truth check (synthetic only)",
            "",
            (
                "The chosen ladders were pushed back through the generator's **own** demand "
                "process — softmax reallocation across tiers, event total responding to the "
                "average log price, the generator's floor and its sell-out cap — using the true "
                f"elasticity of {check['beta_true']:+.2f} rather than the estimated "
                f"{check['beta_used']:+.3f}."
            ),
            "",
            "| | uplift |",
            "|---|---|",
            f"| model-implied (constant elasticity, tier by tier) | {check['model_uplift']:+.1%} |",
            f"| the generator's true demand process | {check['true_uplift']:+.1%} |",
            f"| ratio | {check['ratio']:.2f} |",
            "",
            (
                f"The two {agree} in sign, and the gap between them has a specific cause worth "
                "knowing: the model moves each tier's quantity by that tier's own price change, so "
                "it is implicitly quantity-weighted, while the generator moves the event's total by "
                "the **unweighted average** change across the ladder. Flattening a ladder raises the "
                "cheap rungs (which carry most of the units) and cuts the expensive ones, so the "
                "quantity-weighted view sees a bigger average price rise than the unweighted one and "
                "predicts a bigger loss of sales. The model is the conservative one of the two here, "
                "which is the direction to be wrong in."
            ),
            "",
            (
                "The ratio is what the model's simplification costs. "
                "There is a pinned test on both, with a wide tolerance on purpose: agreement to "
                "three decimal places would only prove that two nearly identical formulas are "
                "nearly identical."
            ),
            "",
            (
                "**What this proves:** applying one elasticity tier by tier does not throw the "
                "answer away. **What it does not prove:** that the real world is a "
                "constant-elasticity world, that ladder shape has no effect, or that beta is what "
                "this fixture says. Nothing run on a fixture can prove any of those."
            ),
            "",
        ]

    parts += [
        "## What this counterfactual cannot see",
        "",
        (
            "- **Buyers moving between tiers.** Change the gaps and some Early Bird buyers become "
            "door buyers, or stop coming. That needs a tier-specific elasticity and the data "
            "cannot support one."
        ),
        (
            "- **Time.** A ladder is a schedule, not just a set of prices, and the sales curve "
            "(Phase 3) is what says whether a tier had time to sell. This grid re-prices the "
            "rungs; it does not re-time them."
        ),
        (
            "- **The competition, and the brand.** Nothing here knows that a rival promoter runs "
            "the same night, or that a night which gets a reputation for being expensive sells "
            "worse next term at any price."
        ),
        (
            "- **Anything past the room.** Bar spend, repeat custom and the value of a full room to "
            "the next booking are all outside the objective, and all three argue for lower prices "
            "than this model wants."
        ),
        (
            "- **The events that did not happen.** Cancelled events are excluded (SPEC.md 8.4), so "
            "this is an uplift conditional on the night going ahead."
        ),
        "",
        "## Provisional while `fee_treatment` is UNKNOWN (SPEC.md 8.6)",
        "",
        (
            "Demand responds to the price the buyer sees; the till receives the face value. The "
            "split between the two is a guess until one real export row is compared with one real "
            "order confirmation, and every pound in this report rests on it."
        ),
        "",
        "## Reproduce",
        "",
        "```",
        regenerate,
        "```",
    ]
    return "\n".join(parts) + "\n"


def summary_lines(results: dict) -> list[str]:
    best = results["best"]
    lines = [
        f"PHASE 6  counterfactual  [{results['source']}]",
        "",
        f"  Phase 4 verdict                {results['verdict']}",
        f"  beta used                      {results['beta_hat']:+.3f}",
        (
            f"  grid                           {len(LEVEL_GRID)} levels x {len(SPREAD_GRID)} spreads"
            f" = {len(LEVEL_GRID) * len(SPREAD_GRID)} schedules per event"
        ),
        "",
    ]
    if results["identified"]:
        lines.append(
            f"  PROJECTED UPLIFT               {results['uplift_at_experiment_dose']:+.1%}"
            f"  (contribution margin, at the pre-committed +15% dose)"
        )
    else:
        lines.append("  PROJECTED UPLIFT               none — beta is NOT identified")
        lines.append(
            f"    at the +15% dose (PROVISIONAL) {results['uplift_at_experiment_dose']:+.1%}"
        )
    lines += [
        (
            f"  grid optimum (a BOUND)         {best['uplift']:+.1%}"
            f"  — {best['share_at_level_boundary']:.0%} of events at the grid edge"
        ),
        (
            f"  modal ladder chosen            level x{best['modal_level']:.2f}, "
            f"spread x{best['modal_spread']:.2f}"
        ),
        "",
        "  uplift curve over assumed beta:",
    ]
    for _, r in results["curve"].iterrows():
        lines.append(
            f"    beta {r['assumed_beta']:+.1f}   optimised {r['uplift']:+7.1%}"
            f"   at +15% {r['uplift_at_experiment_dose']:+7.1%}"
        )
    check = results["ground_truth_check"]
    if check is not None:
        lines += [
            "",
            (
                f"  ground truth check   model {check['model_uplift']:+.1%}"
                f"   true process {check['true_uplift']:+.1%}"
                f"   ratio {check['ratio']:.2f}"
                f"   {'SIGNS AGREE' if check['same_sign'] else 'SIGNS DISAGREE'}"
            ),
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = phaseio.phase_parser("pricing.phase6", "Phase 6 — counterfactual (SPEC.md 6.4)")
    parser.add_argument("--no-write", action="store_true", help="print it, write nothing")
    args = parser.parse_args(argv)
    try:
        run_data = phaseio.resolve(args)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    results = run(run_data)
    print("\n".join(summary_lines(results)))
    if not args.no_write:
        path = phaseio.write_report(report(results, run_data), REPORT_FILENAME, run_data.out_dir)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
