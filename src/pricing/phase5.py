"""Phase 5 — break-even, P(clear the guarantee), and which deal to sign. SPEC.md 6.3.

    uv run python -m pricing.phase5 --synthetic
    uv run python -m pricing.phase5 --real          <- refused while the prereg is DRAFT

The question a promoter actually asks
-------------------------------------
An agent offers you an artist for a fixed fee. Three things decide whether you take it:

    contribution per ticket   = what you keep from one paid ticket
    break-even attendance     = fixed cost stack / contribution per ticket
    P(clear)                  = P(tickets sold > break-even attendance)

and then, because the same artist can be booked three different ways, which structure you
sign (SPEC.md 6.3):

    fixed guarantee            profit = Net - F            you carry all the demand risk
    door split                 profit = (1 - s) * Net      the risk is shared both ways
    guarantee OR s% of door,   profit = Net - max(F, s*Net)
      whichever is greater                                 you get whichever is worse for you

where `Net = units * contribution_per_ticket - fixed costs other than the artist`.

The third one is the standard live-music deal and it is worth staring at: it pays you the
minimum of the other two, every single time. `payoff_guarantee_or_split` is exactly
`min(payoff_fixed_guarantee, payoff_door_split)` — there is a test that asserts it
pointwise. The artist takes the better of their two, so you take the worse of yours: you
are handing them the upside while keeping the downside, and the fee you charge for that
should be a lower F.

Where the distribution comes from
---------------------------------
**Phase 2**, and not from a parametric assumption. Phase 5 does not fit a demand model of
its own: it asks `phase2` for its grouped-K-fold out-of-fold predictions, takes the
residuals of those (so no event is ever scored by a model that had seen it — SPEC.md 8.1),
and resamples them with replacement.

That is a plain bootstrap, and the reason for it is that the spread of simulated outcomes
is then the spread of errors the forecast **actually made on events it had not seen**,
which is the honest answer to "how wrong is this usually?". A normal distribution fitted
to those residuals would be smoother and would understate the thing that matters most
here — the bad tail, which in this business is a wet Wednesday in exam season, not a
symmetric wobble.

Two things follow from taking Phase 2's numbers rather than making new ones:

* the centre of each event's distribution is its **out-of-fold** prediction, so the
  bootstrap is centred on what a model that had never seen the night would have said;
* "tickets sold" means exactly what it means in Phase 1 (`phase1.event_outcomes`) —
  paid units, comps stripped — everywhere in the repo, rather than being recomputed here
  with a slightly different filter and quietly disagreeing.

Method policy (DECISIONS.md, 2026-08-11 amendment): plain arithmetic on the cost stack,
plain bootstrap over an empirical residual distribution. No Bayesian machinery, no fitted
parametric demand distribution, no second demand model.
"""

from __future__ import annotations

import sys
from typing import NamedTuple

import numpy as np
import pandas as pd

from pricing import phase1, phase2, phaseio

REPORT_FILENAME = "phase5_risk.md"

# --------------------------------------------------------------------------------------
# Stated assumptions. Every one of these is a number someone chose; they are all here.
# --------------------------------------------------------------------------------------
#
# VARIABLE_COST_PER_TICKET
#     Cost that scales with the ticket rather than the night — per-head levy, wristband,
#     card fee we absorb. The reconstructed cost stack has no per-head line today, so it
#     is 0 and the contribution per ticket is the full face value we bank. If a real
#     per-head cost turns up in the finance workbooks, put it here: it moves every
#     break-even in the report.
# DOOR_SPLIT_SHARE
#     The artist's share of net door in a split deal. NOT in the data — it comes off the
#     contract, and until the contracts are read it is an assumption used identically for
#     every event so the three structures are compared on the same terms.
# MAX_SURVIVABLE_LOSS
#     The left tail the business can actually absorb on one night. The decision rule in
#     SPEC.md 6.3 is risk-adjusted, not expected-value: take the guarantee when it pays
#     more in expectation AND the 5th-percentile outcome is survivable.
# N_DRAWS
#     Bootstrap draws per event. There are two different uncertainties here and only one of
#     them is fixed by this number:
#       * Monte Carlo noise — the same residuals resampled under a different RNG seed give
#         slightly different percentiles. That shrinks like 1/sqrt(N_DRAWS), so it is the
#         cheap one, and `run()` MEASURES what is left of it at this setting and prints the
#         answer in the report rather than asserting a precision.
#       * Sampling noise — the residual distribution itself has only as many points in it
#         as there are events (~120). Resampling it ten million times does not add
#         information about the tail; the 5th percentile is still being read off a handful
#         of genuinely bad nights. No draw count fixes that, and it is the binding one.
#     4,000 was the old setting, and the report used to ASSERT that it gave a 5th
#     percentile "stable to about a pound". It did not. Re-running the fixture on a second
#     bootstrap seed at 4,000 draws moved the worst event's 5th-percentile night by £184
#     (`run(rd, n_draws=4000)` reproduces it). At 40,000 that worst case is about £30 and
#     the phase still runs in a couple of seconds. The report no longer asserts anything
#     here: `run()` measures the spread across two seeds on every run and prints it.
# P_CLEAR_TARGET
#     The odds a booking has to have before it is worth signing, used to answer the
#     question an agent actually asks: "what can you pay?" The largest fee that still
#     leaves this chance of clearing is the number to walk into the call with.
# BOOTSTRAP_SEED
#     The RNG the Monte Carlo runs on. It is a chosen number like every other one on this
#     list, so it is named here and printed in the report's assumptions table rather than
#     buried in a default argument (review finding, 2026-08-12). Note what `--seed` does
#     and does not do: it re-rolls the synthetic FIXTURE, not the bootstrap. The second
#     run inside `run()` uses BOOTSTRAP_SEED + 999 and the gap between the two is printed,
#     which is the honest version of "does the seed matter?".
VARIABLE_COST_PER_TICKET = 0.00
DOOR_SPLIT_SHARE = 0.70
MAX_SURVIVABLE_LOSS = 1500.00
P_CLEAR_TARGET = 0.50
N_DRAWS = 40000
BOOTSTRAP_SEED = 12345


# --------------------------------------------------------------------------------------
# 1. The cost stack, per event
# --------------------------------------------------------------------------------------


def build_event_panel(
    events: pd.DataFrame,
    transactions: pd.DataFrame,
    variable_cost_per_ticket: float = VARIABLE_COST_PER_TICKET,
) -> pd.DataFrame:
    """One row per event that happened: what it sold, what it cost, what it needed to sell.

    Money definitions, stated once and used everywhere below:

    * `promoter_revenue` — the FACE value of paid tickets. The booking fee is the
      platform's, not ours. While `fee_treatment` is UNKNOWN (SPEC.md 8.6) this split is a
      guess and the report says so; it is the single assumption every pound in this phase
      rests on.
    * `contribution_per_ticket` — promoter revenue per paid ticket, less any per-head cost.
    * `fixed_stack_ex_artist` — venue, security, production, staffing, marketing. The costs
      that happen whether two people come or two hundred.
    * `fixed_cost_stack` — the above plus the artist guarantee. This is SPEC.md 6.3's
      denominator input, and it is the right one for a guarantee deal only; a door split
      has no fixed artist cost, which is exactly why the structures differ.

    Units and revenue come from `phase1.event_outcomes`, not from a fresh groupby here, so
    "tickets sold" means one thing in every phase of this repo: paid units with comps
    stripped (SPEC.md 8.7 — a guestlist is not demand). Cancelled events are dropped by
    `phase1.live` and counted there (SPEC.md 8.4).
    """
    outcomes = phase1.live(phase1.event_outcomes(events, transactions))
    panel = outcomes.rename(
        columns={
            "tickets": "units_sold",
            "revenue_face": "promoter_revenue",
            "revenue_buyer": "buyer_revenue",
        }
    )
    panel = panel[panel["units_sold"] > 0].copy()

    other_fixed = ["venue_cost", "security_cost", "production_cost", "staffing_cost"]
    present = [c for c in other_fixed if c in panel.columns]
    panel["fixed_stack_ex_artist"] = panel[present].fillna(0).sum(axis=1)
    if "marketing_spend" in panel.columns:
        panel["fixed_stack_ex_artist"] += panel["marketing_spend"].fillna(0)
    if "fixed_cost_stack" not in panel.columns:
        panel["fixed_cost_stack"] = panel["fixed_stack_ex_artist"] + panel["artist_guarantee"]

    panel["contribution_per_ticket"] = (
        panel["promoter_revenue"] / panel["units_sold"] - variable_cost_per_ticket
    )
    panel["break_even_units"] = panel["fixed_cost_stack"] / panel["contribution_per_ticket"]
    panel["break_even_sell_through"] = panel["break_even_units"] / panel["capacity"]
    panel["sell_through"] = panel["units_sold"] / panel["capacity"]
    panel["mean_face_price"] = panel["promoter_revenue"] / panel["units_sold"]
    panel["mean_buyer_price"] = panel["buyer_revenue"] / panel["units_sold"]
    panel["log_units"] = np.log(panel["units_sold"])
    return panel.sort_values("event_date").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 2. The three payoff functions (SPEC.md 6.3)
# --------------------------------------------------------------------------------------
#
# All three take `units` (an array of simulated outcomes, or one number) and return the
# promoter's profit in pounds. They are pure arithmetic: no state, no fitting, no data.
# That is why they are the easiest thing in the repo to test and the easiest to argue
# about with an agent on the phone.


def net_before_artist(
    units: np.ndarray | float, contribution_per_ticket: float, fixed_stack_ex_artist: float
) -> np.ndarray | float:
    """`Net` — what the night makes before the artist is paid anything.

    Ticket contribution minus every fixed cost that is not the artist. This is the base
    all three structures are written against, and it is the number an agent means when
    they say "percent of the door".
    """
    return np.asarray(units) * contribution_per_ticket - fixed_stack_ex_artist


def payoff_fixed_guarantee(
    units: np.ndarray | float,
    contribution_per_ticket: float,
    fixed_stack_ex_artist: float,
    guarantee: float,
) -> np.ndarray | float:
    """Flat fee. `profit = Net - F`. You carry all of the demand risk, both ways.

    A quiet night costs you the whole shortfall; a sell-out is entirely yours.
    """
    return net_before_artist(units, contribution_per_ticket, fixed_stack_ex_artist) - guarantee


def payoff_door_split(
    units: np.ndarray | float,
    contribution_per_ticket: float,
    fixed_stack_ex_artist: float,
    share: float = DOOR_SPLIT_SHARE,
) -> np.ndarray | float:
    """Percentage of the door. `profit = (1 - s) * Net`. Risk shared in both directions.

    Written exactly as SPEC.md 6.3 writes it, which means a loss is shared too. Most real
    contracts floor the artist's share at zero so they never pay in; that variant is one
    line (`Net - s * max(Net, 0)`) and it is deliberately NOT what is implemented, because
    the spec's table is what the rest of the analysis is written against. If a real
    contract has the floor, this understates your loss on a bad night.
    """
    return (1.0 - share) * net_before_artist(units, contribution_per_ticket, fixed_stack_ex_artist)


def payoff_guarantee_or_split(
    units: np.ndarray | float,
    contribution_per_ticket: float,
    fixed_stack_ex_artist: float,
    guarantee: float,
    share: float = DOOR_SPLIT_SHARE,
) -> np.ndarray | float:
    """ "Guarantee OR s% of the door, whichever is greater" — the standard live-music deal.

    `profit = Net - max(F, s * Net)`.

    The artist takes the better of their two outcomes on every night, so you take the
    worse of yours on every night:

        payoff_guarantee_or_split == minimum(payoff_fixed_guarantee, payoff_door_split)

    exactly, for any units, any F > 0 and any 0 < s < 1. There is a test that asserts it
    pointwise. Below the crossover (`Net = F / s`) it behaves like the flat fee; above it,
    like the split. So it costs you the whole downside of the guarantee deal AND the
    capped upside of the split deal — the two worst halves. The compensation for that has
    to be a lower F, and if the F is the same as a straight guarantee you are being paid
    nothing for the difference.
    """
    net = net_before_artist(units, contribution_per_ticket, fixed_stack_ex_artist)
    return net - np.maximum(guarantee, share * net)


def crossover_net(guarantee: float, share: float = DOOR_SPLIT_SHARE) -> float:
    """The `Net` at which the artist stops taking the fee and starts taking the split."""
    return guarantee / share


# --------------------------------------------------------------------------------------
# 3. A demand distribution, by resampling out-of-sample errors
# --------------------------------------------------------------------------------------


def demand_distribution(events: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Phase 2's out-of-fold forecast and its errors. One row per event.

    This is the whole interface between Phase 2 and Phase 5, and it is deliberately three
    lines: build Phase 2's features, run its grouped K-fold (every event scored by a model
    that never saw it — SPEC.md 8.1), and take the log residual of each out-of-fold
    prediction.

    Columns returned: `event_id`, `predicted` (tickets), `tickets` (actual),
    `log_predicted`, `log_resid`.

    Phase 5 fits nothing of its own. Two demand models in one repo means two answers to
    "how many will come", and the day they disagree is the day the report stops being
    trustworthy. Note also what Phase 2 does NOT use as a feature: price. In a predictive
    model the price coefficient is the operator's own pricing rule, not a demand response
    (SPEC.md 4.1), so a forecast that "learns" higher prices sell more must never be the
    thing that chooses a price. Price enters this project once, as Phase 4's beta, applied
    on top of this by `simulate_event(log_price_change=..., beta=...)` and by Phase 6.
    """
    features = phase2.build_features(events, transactions)
    _folds, out_of_fold = phase2.cross_validate(features)
    out = out_of_fold[["event_id", "tickets", "predicted"]].copy()
    out["log_predicted"] = np.log(out["predicted"])
    out["log_resid"] = np.log(out["tickets"]) - out["log_predicted"]
    return out


def bootstrap_units(
    expected_log_units: float,
    residuals: np.ndarray,
    rng: np.random.Generator,
    n_draws: int,
    capacity: float,
    beta: float = 0.0,
    log_price_change: float = 0.0,
) -> np.ndarray:
    """The whole Monte Carlo, in three lines. Every simulated night is a night we had.

    `rng.choice(residuals, replace=True)` is the method: each draw is one of the forecast's
    ACTUAL out-of-sample misses, re-used. **Nothing is fitted.** No normal, no lognormal,
    no kernel — which matters because the number this phase exists to produce is a left
    tail, and a fitted symmetric distribution would smooth away the wet Wednesday in exam
    season that is the whole reason for asking.

    The consequence is testable, and there is a test that tests it: the simulated log
    demand can only ever be `expected_log_units + r` for an `r` that is literally an
    element of `residuals`, so the set of distinct simulated unit counts is a subset of the
    set the residual array can produce. Replace this line with a parametric draw and that
    stops being true immediately.

    `beta * log_price_change` is how a different ladder is simulated (Phase 6's hook); at
    the defaults it is zero. The result is rounded to whole tickets and capped at the room.
    """
    draws = rng.choice(residuals, size=n_draws, replace=True)
    units = np.exp(expected_log_units + beta * log_price_change + draws)
    return np.clip(np.round(units), 0.0, float(capacity))


class EventRisk(NamedTuple):
    """One event's break-even, its P(clear), and the three structures side by side."""

    event_id: str
    capacity: float
    guarantee: float
    contribution_per_ticket: float
    fixed_stack_ex_artist: float
    fixed_cost_stack: float
    break_even_units: float
    break_even_sell_through: float
    actual_units: float
    expected_units: float
    p_clear: float
    max_guarantee_at_target: float
    profit: dict  # structure -> {"mean", "p05", "p_positive"}
    recommendation: str


def simulate_event(
    row: pd.Series,
    expected_log_units: float,
    residuals: np.ndarray,
    rng: np.random.Generator,
    n_draws: int = N_DRAWS,
    share: float = DOOR_SPLIT_SHARE,
    log_price_change: float = 0.0,
    beta: float = 0.0,
    max_survivable_loss: float = MAX_SURVIVABLE_LOSS,
) -> EventRisk:
    """Bootstrap one event: resample residuals, price the three structures on each draw.

    `log_price_change` and `beta` are how a *different* ladder is simulated (Phase 6 uses
    them): the predicted log demand shifts by `beta * log_price_change`, and the
    contribution per ticket scales with the price. At their defaults nothing moves and
    this is the risk of the night as it was actually priced.

    Simulated units are capped at capacity — you cannot sell a room you do not have — and
    that cap is the reason the profit distribution is not symmetric even though the
    residual distribution nearly is.
    """
    units = bootstrap_units(
        expected_log_units,
        residuals,
        rng,
        n_draws,
        float(row["capacity"]),
        beta=beta,
        log_price_change=log_price_change,
    )

    contribution = float(row["contribution_per_ticket"]) * float(np.exp(log_price_change))
    fixed_ex_artist = float(row["fixed_stack_ex_artist"])
    guarantee = float(row["artist_guarantee"])
    break_even = (fixed_ex_artist + guarantee) / contribution

    payoffs = {
        "fixed_guarantee": payoff_fixed_guarantee(units, contribution, fixed_ex_artist, guarantee),
        "door_split": payoff_door_split(units, contribution, fixed_ex_artist, share),
        "guarantee_or_split": payoff_guarantee_or_split(
            units, contribution, fixed_ex_artist, guarantee, share
        ),
    }
    profit = {
        name: {
            "mean": float(np.mean(values)),
            "p05": float(np.percentile(values, 5)),
            "p_positive": float(np.mean(values > 0)),
        }
        for name, values in payoffs.items()
    }

    return EventRisk(
        event_id=str(row["event_id"]),
        capacity=float(row["capacity"]),
        guarantee=guarantee,
        contribution_per_ticket=contribution,
        fixed_stack_ex_artist=fixed_ex_artist,
        fixed_cost_stack=fixed_ex_artist + guarantee,
        break_even_units=break_even,
        break_even_sell_through=break_even / float(row["capacity"]),
        actual_units=float(row["units_sold"]),
        expected_units=float(np.mean(units)),
        # SPEC.md 6.3's definition, verbatim: P(Q > break-even attendance). It is the same
        # event as "the fixed-guarantee deal makes money", which is the arithmetic check
        # in the tests.
        p_clear=float(np.mean(units > break_even)),
        # The number to take into the phone call: the largest fee that still leaves a
        # P_CLEAR_TARGET chance of clearing. P(Net > F) = target means F is the
        # (1 - target) quantile of the simulated Net, which is one line and no solver.
        #
        # Clamped at zero. The quantile goes negative whenever the night is more likely
        # than not to lose money before the artist is paid anything, and "the largest fee
        # you can pay is minus £1,001" is not a number to take into a phone call (review
        # finding, 2026-08-12). Zero is the honest reading of that case and it has a
        # meaning: no fee, not even a free booking, clears the target at this room.
        max_guarantee_at_target=max(
            0.0,
            float(
                np.percentile(
                    net_before_artist(units, contribution, fixed_ex_artist),
                    100.0 * (1.0 - P_CLEAR_TARGET),
                )
            ),
        ),
        profit=profit,
        recommendation=recommend_structure(profit, max_survivable_loss),
    )


def p_clear_curve(
    row: pd.Series,
    expected_log_units: float,
    residuals: np.ndarray,
    rng: np.random.Generator,
    guarantees: np.ndarray,
    n_draws: int = N_DRAWS,
) -> pd.DataFrame:
    """P(clear) for one event across a range of artist fees. The negotiating table.

    One set of draws, then a simple count at each fee — `P(clear) = P(Net > F)` is just
    one minus the empirical distribution of `Net`, so the curve costs nothing extra and
    is guaranteed to be non-increasing in the fee, which is a property worth testing.
    """
    units = bootstrap_units(expected_log_units, residuals, rng, n_draws, float(row["capacity"]))
    net = net_before_artist(
        units, float(row["contribution_per_ticket"]), float(row["fixed_stack_ex_artist"])
    )
    return pd.DataFrame(
        {
            "guarantee": guarantees,
            "p_clear": [float(np.mean(net > f)) for f in guarantees],
            "mean_profit": [float(np.mean(net - f)) for f in guarantees],
        }
    )


def recommend_structure(profit: dict, max_survivable_loss: float = MAX_SURVIVABLE_LOSS) -> str:
    """SPEC.md 6.3's decision rule, written out: expectation first, then the tail.

    > Take the guarantee when E[profit | guarantee] > E[profit | split] **and** the left
    > tail is survivable. That is a risk-adjusted choice, not an expected-value one.

    So: the guarantee has to win on the mean AND its 5th-percentile night has to be a loss
    the business can actually absorb. If it wins on the mean but the bad night is bigger
    than `MAX_SURVIVABLE_LOSS`, the answer is the split — you are buying insurance and its
    price is the expected profit you give up.
    """
    fixed, split = profit["fixed_guarantee"], profit["door_split"]
    if fixed["mean"] <= split["mean"]:
        return "door split (higher expected profit)"
    if fixed["p05"] < -abs(max_survivable_loss):
        return (
            f"door split (guarantee pays more on average but its 5th-percentile night is "
            f"£{fixed['p05']:,.0f}, past the £{max_survivable_loss:,.0f} the business can absorb)"
        )
    return "fixed guarantee (higher expected profit, survivable tail)"


# --------------------------------------------------------------------------------------
# 4. Run everything
# --------------------------------------------------------------------------------------


def monte_carlo_spread(risks_a: list[EventRisk], risks_b: list[EventRisk]) -> dict:
    """How far the reported numbers move when only the RNG seed changes. Measured, not claimed.

    Two runs over the SAME events and the SAME residuals, differing only in the bootstrap
    seed. Whatever is left between them is Monte Carlo noise and nothing else — it is the
    error bar on the machinery, and the honest way to quote a simulated percentile is with
    it attached rather than with an adjective.

    It is the max across events, not the mean, because the number a reader will act on is
    one event's tail, not the programme's average tail.
    """

    def gap(get) -> dict:
        diffs = np.abs(np.array([get(a) - get(b) for a, b in zip(risks_a, risks_b, strict=True)]))
        return {"median": float(np.median(diffs)), "max": float(np.max(diffs))}

    return {
        "p05_fixed": gap(lambda r: r.profit["fixed_guarantee"]["p05"]),
        "mean_profit_fixed": gap(lambda r: r.profit["fixed_guarantee"]["mean"]),
        "p_clear": gap(lambda r: r.p_clear),
    }


def run(run_data: phaseio.Run, n_draws: int = N_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Every Phase 5 number, in one dict."""
    forecast = demand_distribution(run_data.events, run_data.transactions)
    panel = build_event_panel(run_data.events, run_data.transactions).merge(
        forecast[["event_id", "predicted", "log_predicted", "log_resid"]],
        on="event_id",
        how="inner",
    )
    residuals = panel["log_resid"].to_numpy(dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    rng = np.random.default_rng(seed)
    risks = [
        simulate_event(row, float(row["log_predicted"]), residuals, rng, n_draws=n_draws)
        for _, row in panel.iterrows()
    ]
    # The same events, the same residuals, a different bootstrap seed. The gap between the
    # two runs is the Monte Carlo error, and it goes in the report as a measured number
    # rather than as an adjective. It costs one more pass over the events.
    rng_b = np.random.default_rng(seed + 999)
    risks_b = [
        simulate_event(row, float(row["log_predicted"]), residuals, rng_b, n_draws=n_draws)
        for _, row in panel.iterrows()
    ]
    mc_spread = monte_carlo_spread(risks, risks_b)

    summary = pd.DataFrame(
        {
            "event_id": [r.event_id for r in risks],
            "capacity": [r.capacity for r in risks],
            "guarantee": [r.guarantee for r in risks],
            "contribution_per_ticket": [r.contribution_per_ticket for r in risks],
            "break_even_units": [r.break_even_units for r in risks],
            "break_even_sell_through": [r.break_even_sell_through for r in risks],
            "actual_units": [r.actual_units for r in risks],
            "p_clear": [r.p_clear for r in risks],
            "max_guarantee_at_target": [r.max_guarantee_at_target for r in risks],
            "e_profit_fixed": [r.profit["fixed_guarantee"]["mean"] for r in risks],
            "p05_fixed": [r.profit["fixed_guarantee"]["p05"] for r in risks],
            "e_profit_split": [r.profit["door_split"]["mean"] for r in risks],
            "p05_split": [r.profit["door_split"]["p05"] for r in risks],
            "e_profit_either": [r.profit["guarantee_or_split"]["mean"] for r in risks],
            "p05_either": [r.profit["guarantee_or_split"]["p05"] for r in risks],
            "recommendation": [r.recommendation for r in risks],
        }
    )

    # One event's whole negotiating curve, for the report. The event picked is the one
    # with the most HEADROOM — the biggest fee it could pay and still be a coin flip.
    # The rule is stated rather than "an illustrative event", because picking the example
    # that looks best and calling it illustrative is how reports start lying.
    example_pos = int(np.argmax(summary["max_guarantee_at_target"].to_numpy()))
    example_row = panel.iloc[example_pos]
    top_fee = max(2.0 * float(example_row["artist_guarantee"]), 100.0)
    curve = p_clear_curve(
        example_row,
        float(example_row["log_predicted"]),
        residuals,
        np.random.default_rng(seed + 1),
        guarantees=np.round(np.linspace(0.0, top_fee, 9), -1),
        n_draws=n_draws,
    )

    return {
        "source": run_data.source,
        "seed": run_data.seed,
        "panel": panel,
        "example_event": str(example_row["event_id"]),
        "example_curve": curve,
        "oos_residuals": residuals,
        # Phase 2's own headline error, recomputed here from the same out-of-fold
        # predictions so the two reports cannot drift apart.
        "oos_mae_tickets": float(np.mean(np.abs(panel["units_sold"] - panel["predicted"]))),
        "oos_mae_log": float(np.mean(np.abs(residuals))),
        "oos_mape": float(np.mean(np.abs(panel["predicted"] / panel["units_sold"] - 1.0))),
        "risks": risks,
        "summary": summary,
        "n_draws": n_draws,
        "bootstrap_seed": seed,
        "share": DOOR_SPLIT_SHARE,
        # Measured Monte Carlo error at this draw count (see `monte_carlo_spread`).
        "mc_spread": mc_spread,
        # Cancelled events are excluded from every distribution above (SPEC.md 8.4). The
        # count travels with the results so the report can state it instead of asserting a
        # number that was true of one fixture.
        "n_cancelled": int(run_data.events["cancelled"].sum())
        if "cancelled" in run_data.events.columns
        else 0,
    }


# --------------------------------------------------------------------------------------
# 5. The report
# --------------------------------------------------------------------------------------


def _money(x: float) -> str:
    return f"£{x:,.0f}"


def _payable_fee(x: float) -> str:
    """The largest payable fee, worded for a phone call. Zero is a sentence, not a number."""
    return _money(x) if x > 0 else "**£0 — no fee clears**"


def _calibration_note(results: dict) -> str:
    """Say it out loud when the P&L in front of you cannot be read as a P&L.

    The synthetic generator was built to make STATISTICAL properties visible — a
    wrong-signed pooled regression, a recoverable elasticity — and its cost stack was
    never calibrated against its ticket prices. On the fixture the stack comes to roughly
    twice ticket revenue, so nearly every simulated night loses money and P(clear) is
    ~0 everywhere. That is a fact about the fixture, not about the method, and hiding it
    behind a friendlier example would be the exact dishonesty this repo is about.
    """
    s = results["summary"]
    ratio = float(s["break_even_units"].median() / s["actual_units"].median())
    if results["source"] != "synthetic" or ratio < 1.2:
        return ""
    return (
        "> **Read the synthetic P(clear) numbers as machinery, not as a P&L.** The generator "
        "was built to make statistical properties visible (a wrong-signed pooled regression, a "
        "recoverable elasticity) and its cost stack was never calibrated against its ticket "
        f"prices: the median break-even is {ratio:.1f}x the median tickets actually sold, so "
        "almost every simulated night loses money and P(clear) sits at ~0. What is worth "
        "reading here is the *shape* — how the three structures order themselves, how the tail "
        "behaves, how the most-payable fee falls out of the draws. The levels wait for the real "
        "cost stack."
    )


def report(results: dict, run_data: phaseio.Run) -> str:
    s = results["summary"]
    regenerate = phaseio.regenerate_command("phase5", results["source"], results["seed"])
    biggest = s.sort_values("guarantee", ascending=False).head(6)

    example_rows = [
        (
            "| event | capacity | guarantee | contribution/ticket | break-even | break-even "
            f"sell-through | actually sold | **P(clear)** | most payable at "
            f"{P_CLEAR_TARGET:.0%} |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in biggest.iterrows():
        example_rows.append(
            f"| {r['event_id']} | {r['capacity']:.0f} | {_money(r['guarantee'])} | "
            f"£{r['contribution_per_ticket']:.2f} | {r['break_even_units']:.0f} | "
            f"{r['break_even_sell_through']:.0%} | {r['actual_units']:.0f} | "
            f"**{r['p_clear']:.0%}** | {_payable_fee(r['max_guarantee_at_target'])} |"
        )

    structure_rows = [
        (
            "| structure | who carries the risk | mean E[profit] | mean 5th-percentile night | "
            "share of events profitable in expectation |"
        ),
        "|---|---|---|---|---|",
        (
            f"| fixed guarantee | you, all of it | {_money(s['e_profit_fixed'].mean())} | "
            f"{_money(s['p05_fixed'].mean())} | {(s['e_profit_fixed'] > 0).mean():.0%} |"
        ),
        (
            f"| door split ({results['share']:.0%} to artist) | shared both ways | "
            f"{_money(s['e_profit_split'].mean())} | {_money(s['p05_split'].mean())} | "
            f"{(s['e_profit_split'] > 0).mean():.0%} |"
        ),
        (
            f"| guarantee OR {results['share']:.0%} of door, whichever is greater | "
            f"you keep the worse of the two | {_money(s['e_profit_either'].mean())} | "
            f"{_money(s['p05_either'].mean())} | {(s['e_profit_either'] > 0).mean():.0%} |"
        ),
    ]

    guarantee_wins = int((s["e_profit_fixed"] > s["e_profit_split"]).sum())
    recommended_guarantee = int(s["recommendation"].str.startswith("fixed").sum())

    parts = [
        phaseio.provenance_header(
            "Phase 5 — Break-even, P(clear) and deal structure",
            run_data.source,
            run_data.seed,
            regenerate,
        ),
        "",
        (
            "**The question:** an agent offers an artist for a fixed fee. What has to happen for "
            "that night to make money, how likely is it, and should the deal be a flat fee or a "
            "share of the door?"
        ),
        "",
        "## Method, in four steps",
        "",
        (
            "1. **Contribution per ticket** = face value banked per paid ticket, less any per-head "
            f"cost (£{VARIABLE_COST_PER_TICKET:.2f} — see assumptions). Comps are excluded: a "
            "guestlist is not demand (SPEC.md 8.7)."
        ),
        (
            "2. **Break-even attendance** = fixed cost stack ÷ contribution per ticket. The fixed "
            "stack is venue + security + production + staffing + marketing + the artist guarantee."
        ),
        (
            "3. **A demand distribution, not a point estimate — and it is Phase 2's, not a new "
            "one.** Phase 5 fits no model. It takes Phase 2's grouped-K-fold **out-of-fold** "
            "predictions (every event scored by a model that never saw it — SPEC.md 8.1) and "
            "resamples their residuals with replacement. No normal distribution is fitted to "
            "anything: the spread of simulated outcomes is the spread of mistakes the forecast "
            "actually made."
        ),
        (
            f"4. **{results['n_draws']:,} draws per event**, each one priced through all three deal "
            "structures."
        ),
        "",
        "### How much of this is just the random number generator?",
        "",
        (
            "A simulated percentile is worth nothing without knowing how much it wobbles when you "
            "re-run it, so the whole thing is run twice — same events, same residuals, different "
            "bootstrap seed — and the gap is printed rather than described:"
        ),
        "",
        "| number | median move across seeds | worst event |",
        "|---|---|---|",
        (
            f"| 5th-percentile night (fixed guarantee) | "
            f"{_money(results['mc_spread']['p05_fixed']['median'])} | "
            f"{_money(results['mc_spread']['p05_fixed']['max'])} |"
        ),
        (
            f"| expected profit (fixed guarantee) | "
            f"{_money(results['mc_spread']['mean_profit_fixed']['median'])} | "
            f"{_money(results['mc_spread']['mean_profit_fixed']['max'])} |"
        ),
        (
            f"| P(clear) | {results['mc_spread']['p_clear']['median']:.3f} | "
            f"{results['mc_spread']['p_clear']['max']:.3f} |"
        ),
        "",
        (
            "The median move is small and the worst event is not, and the reason is worth "
            f"knowing: the residual distribution is **atomic** — {len(results['oos_residuals'])} "
            "residuals, one per event — so a simulated 5th percentile can only ever land on one "
            "of that many values. Between seeds it is usually the same value and occasionally one "
            "step away, and the size of that step is set by the data, not by the draw count."
        ),
        "",
        (
            "**This is the smaller of the two uncertainties here, and the only one more draws can "
            "fix.** The one they cannot: that same atomic residual distribution is the entire "
            "evidence about the tail. The 5th percentile is being read off a handful of genuinely "
            "bad nights, and resampling them ten million times adds no information about how bad "
            "a night can get. Quote these percentiles as an order of magnitude."
        ),
        "",
        "## The demand distribution behind every probability here",
        "",
        (
            f"- source: **Phase 2** (`pricing.phase2`), `{len(results['oos_residuals'])}` out-of-fold "
            "predictions, one per event"
        ),
        (
            f"- out-of-fold mean absolute error: {results['oos_mae_tickets']:.1f} tickets "
            f"({results['oos_mape']:.0%} MAPE, {results['oos_mae_log']:.3f} log points)"
        ),
        (
            "- the centre of each event's distribution is its **out-of-fold** prediction, so the "
            "bootstrap is centred on what a model that had never seen the night would have said, "
            "not on a fit that already knew the answer"
        ),
        (
            "- **price is deliberately not a feature of that forecast.** In a predictive model its "
            "coefficient is our own pricing rule, not a demand response (SPEC.md 4.1), and a model "
            "that learns 'higher prices sell more' is precisely the model that must never choose a "
            "price. Price enters this project once, as Phase 4's beta, applied on top of this "
            "forecast by Phase 6."
        ),
        (
            "- two demand models in one repo means two answers to 'how many will come', and the day "
            "they disagree is the day the report stops being trustworthy. There is one."
        ),
        "",
        "## The biggest bookings",
        "",
        "\n".join(example_rows),
        "",
        (
            "Across all "
            f"{len(s)} events: median break-even is {s['break_even_units'].median():.0f} tickets "
            f"({s['break_even_sell_through'].median():.0%} of the room), median P(clear) is "
            f"**{s['p_clear'].median():.0%}**, and {int((s['p_clear'] < 0.5).sum())} events were "
            "more likely than not to lose money at the price they were sold at."
        ),
        "",
        (
            f"The last column is the one to take into the phone call: the largest fee that still "
            f"leaves a {P_CLEAR_TARGET:.0%} chance of clearing. It is the (1 − target) quantile of "
            "simulated `Net` — no solver, one line — **clamped at zero**, because the quantile "
            "goes negative on a night that is more likely than not to lose money before the "
            "artist is paid anything, and 'the most you can pay is minus a thousand pounds' is "
            "not a number you can take into a call. **£0 means no fee clears at this room**, not "
            "even a free booking. Median across the programme: "
            f"{_money(s['max_guarantee_at_target'].median())}"
            + (
                f" — no fee clears the {P_CLEAR_TARGET:.0%} target on at least half the programme"
                if s["max_guarantee_at_target"].median() <= 0
                else ""
            )
            + f", against a median guarantee actually paid of {_money(s['guarantee'].median())}."
        ),
        "",
        _calibration_note(results),
        "",
        f"### What the fee is worth — {results['example_event']}",
        "",
        (
            "The event with the most headroom in the programme — the one that could pay the largest "
            f"fee and still be a {P_CLEAR_TARGET:.0%} shot. Picked by that rule, not by which chart "
            "looked best."
        ),
        "",
        (
            "The same simulation, run across a range of artist fees. This is the table to have "
            'open during the call: it turns "can you do £2,000?" into a probability instead of a '
            "feeling. `P(clear)` is one minus the empirical distribution of `Net`, so it can only "
            "fall as the fee rises — a property worth checking, and there is a test that does."
        ),
        "",
        "| artist fee | P(clear) | expected profit |",
        "|---|---|---|",
        *[
            f"| {_money(r['guarantee'])} | {r['p_clear']:.0%} | {_money(r['mean_profit'])} |"
            for _, r in results["example_curve"].iterrows()
        ],
        "",
        "## The three deal structures (SPEC.md 6.3)",
        "",
        (
            "Writing `Net` for what the night makes before the artist is paid — ticket contribution "
            "minus every fixed cost except the artist:"
        ),
        "",
        "| structure | your profit |",
        "|---|---|",
        "| fixed guarantee | `Net − F` |",
        f"| door split | `(1 − s)·Net`, s = {results['share']:.0%} |",
        "| guarantee OR s% of door, whichever is greater | `Net − max(F, s·Net)` |",
        "",
        (
            "The third line is the standard live-music deal and it is worth reading twice: it pays "
            "**the minimum of the other two, on every single night**. Below the crossover "
            "(`Net = F / s`) it behaves exactly like the flat fee; above it, exactly like the "
            "split. The artist takes the better of their two outcomes, so you take the worse of "
            "yours — the full downside of the guarantee deal *and* the capped upside of the split "
            "deal. The only compensation for that is a lower F; if the F is the same as a straight "
            "guarantee, you are being paid nothing for the difference."
        ),
        "",
        "\n".join(structure_rows),
        "",
        (
            f"The fixed guarantee beats the split in expectation on **{guarantee_wins} of {len(s)}** "
            f"events, but after the tail test only **{recommended_guarantee}** are recommended as "
            "guarantees."
        ),
        "",
        "### The decision rule",
        "",
        (
            "> Take the guarantee when `E[profit | guarantee] > E[profit | split]` **and** the left "
            "tail is survivable."
        ),
        "",
        (
            f"Survivable is a number, not a feeling: `MAX_SURVIVABLE_LOSS = "
            f"{_money(MAX_SURVIVABLE_LOSS)}` on one night. Where the guarantee wins on the mean but "
            "its 5th-percentile night is worse than that, the recommendation flips to the split — "
            "you are buying insurance, and its price is the expected profit you hand over."
        ),
        "",
        "## Assumptions, all of them",
        "",
        "| assumption | value | where it comes from |",
        "|---|---|---|",
        (
            f"| variable cost per ticket | £{VARIABLE_COST_PER_TICKET:.2f} | no per-head line exists "
            "in the cost stack yet. If one appears, it moves every break-even here |"
        ),
        (
            f"| artist share of net door | {DOOR_SPLIT_SHARE:.0%} | **not in the data** — it is in "
            "the contracts. Used identically for every event so the structures compare like for "
            "like |"
        ),
        (
            f"| survivable one-night loss | {_money(MAX_SURVIVABLE_LOSS)} | a business judgement, "
            "written down so it can be argued with |"
        ),
        (
            f"| bootstrap draws | {results['n_draws']:,} | enough that re-running on a different "
            f"RNG seed moves an event's 5th-percentile night by at most "
            f"{_money(results['mc_spread']['p05_fixed']['max'])} (median "
            f"{_money(results['mc_spread']['p05_fixed']['median'])}) — measured, see below |"
        ),
        (
            f"| bootstrap RNG seed | {results['bootstrap_seed']} | a chosen number, like every "
            f"other row here. `--seed` re-rolls the synthetic fixture, NOT the Monte Carlo; the "
            f"second run at seed {results['bootstrap_seed'] + 999} is what measures the row above |"
        ),
        (
            "| promoter keeps face value, platform keeps the booking fee | `fee_treatment` is "
            "**UNKNOWN** | SPEC.md 8.6. This is the one assumption every pound in this report rests "
            "on, and it is cheap to resolve: compare one real export row against one order "
            "confirmation |"
        ),
        "",
        "## What this does not model",
        "",
        (
            "- **Bar spend.** For a promoter with a bar deal, tickets are not the whole margin, and "
            "a night that loses money on tickets can still be worth running. None of that is here."
        ),
        (
            f"- **Cancellation.** Cancelled events are excluded (SPEC.md 8.4), so every "
            f"distribution here is conditional on the night going ahead. "
            f"{results['n_cancelled']} events were pulled or downscaled. Why each one was pulled "
            "is **not something this dataset can check**: their presale histories were destroyed "
            "on-platform (DECISIONS.md, survivorship), so the rows that would say how badly they "
            "were selling do not exist. If poor sales are part of what gets a night pulled — and "
            "on the operator's own account they sometimes are — then the true left tail is worse "
            "than the one drawn here, and by an amount nobody can currently quantify. That is a "
            "known, unmeasured bias, not a rounding error."
        ),
        (
            "- **Correlation between events.** Each night is bootstrapped independently. Two events "
            "a week apart eat each other's audience (SPEC.md 8.10), so a portfolio built by adding "
            "these distributions up would understate a bad month."
        ),
        (
            "- **The residual distribution is the model's, not the world's.** It has as many draws "
            f"as there are events ({len(results['oos_residuals'])}). The 5th percentile is being "
            "read off a few dozen bad nights, and it should be quoted as an order of magnitude."
        ),
        "",
        "## Reproduce",
        "",
        "```",
        regenerate,
        "```",
        "",
        (
            f"`--seed` re-rolls the fixture. The bootstrap is separate and fixed at "
            f"`BOOTSTRAP_SEED = {results['bootstrap_seed']}`; to see how much of any number here "
            f"is the RNG, change it in `phase5.py` — or read the two-seed table above, which is "
            f"that experiment already run."
        ),
    ]
    return "\n".join(parts) + "\n"


def summary_lines(results: dict) -> list[str]:
    s = results["summary"]
    lines = [
        f"PHASE 5  risk  [{results['source']}]",
        "",
        f"  events priced            {len(s)}",
        (
            f"  demand model             Phase 2 out-of-fold, MAE "
            f"{results['oos_mae_tickets']:.1f} tickets ({results['oos_mape']:.0%})"
        ),
        (
            f"  median break-even        {s['break_even_units'].median():.0f} tickets"
            f"  ({s['break_even_sell_through'].median():.0%} of the room)"
        ),
        f"  median P(clear)          {s['p_clear'].median():.0%}",
        f"  events with P(clear)<50% {int((s['p_clear'] < 0.5).sum())}",
        (
            f"  median payable fee @{P_CLEAR_TARGET:.0%}  "
            f"{_money(s['max_guarantee_at_target'].median())}"
            + (
                "  (no fee clears at this room)"
                if s["max_guarantee_at_target"].median() <= 0
                else ""
            )
            + f"   (median fee actually paid {_money(s['guarantee'].median())})"
        ),
        "",
        "  mean profit by structure     E[profit]        5th percentile",
        (
            f"    fixed guarantee            {_money(s['e_profit_fixed'].mean()):>10s}"
            f"       {_money(s['p05_fixed'].mean()):>10s}"
        ),
        (
            f"    door split ({results['share']:.0%})            "
            f"{_money(s['e_profit_split'].mean()):>10s}       {_money(s['p05_split'].mean()):>10s}"
        ),
        (
            f"    guarantee OR split         {_money(s['e_profit_either'].mean()):>10s}"
            f"       {_money(s['p05_either'].mean()):>10s}"
        ),
    ]
    example = s.sort_values("guarantee", ascending=False).head(1)
    if len(example):
        r = example.iloc[0]
        lines += [
            "",
            (
                f"  biggest booking {r['event_id']}: break-even {r['break_even_units']:.0f} tickets"
                f" of {r['capacity']:.0f}, P(clear) {r['p_clear']:.0%}"
            ),
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = phaseio.phase_parser("pricing.phase5", "Phase 5 — risk / Monte Carlo (SPEC.md 6.3)")
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
