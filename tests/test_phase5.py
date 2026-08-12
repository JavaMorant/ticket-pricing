"""Tests for Phase 5 — break-even, P(clear) and deal structure (SPEC.md 6.3).

The payoff functions are pure arithmetic, so they are tested as arithmetic: against
closed-form answers, at their boundaries, and against each other. The one that matters
most is `test_guarantee_or_split_is_exactly_the_minimum_of_the_other_two` — that identity
is the entire argument for why the standard live-music deal is worse than it looks, and if
it ever stops holding, the report's central claim is wrong.
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from pricing import phase5, phaseio, synthetic

CONTRIBUTION = 8.0
FIXED = 3000.0
GUARANTEE = 2000.0
SHARE = 0.7


@lru_cache(maxsize=4)
def _run(seed: int) -> dict:
    args = argparse.Namespace(real=False, synthetic=True, seed=seed, no_write=True)
    return phase5.run(phaseio.resolve(args), n_draws=1500)


@lru_cache(maxsize=4)
def _run_with_draws(n_draws: int) -> dict:
    """The same fixture at a different draw count — used to show the MC error is measured."""
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    return phase5.run(phaseio.resolve(args), n_draws=n_draws)


@pytest.fixture(scope="module")
def results() -> dict:
    return _run(synthetic.DEFAULT_SEED)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    data = synthetic.generate(seed=synthetic.DEFAULT_SEED)
    return phase5.build_event_panel(data.events, data.transactions)


# --------------------------------------------------------------------------------------
# The payoff functions, as algebra
# --------------------------------------------------------------------------------------


def test_net_before_artist_is_contribution_times_units_minus_fixed_costs():
    assert phase5.net_before_artist(500, CONTRIBUTION, FIXED) == pytest.approx(500 * 8.0 - 3000)
    assert phase5.net_before_artist(0, CONTRIBUTION, FIXED) == pytest.approx(-3000)


def test_fixed_guarantee_payoff_is_net_minus_the_fee():
    units = np.array([0, 100, 500, 1000])
    net = phase5.net_before_artist(units, CONTRIBUTION, FIXED)
    got = phase5.payoff_fixed_guarantee(units, CONTRIBUTION, FIXED, GUARANTEE)
    assert np.allclose(got, net - GUARANTEE)


def test_door_split_payoff_is_your_share_of_net():
    units = np.array([0, 100, 500, 1000])
    net = phase5.net_before_artist(units, CONTRIBUTION, FIXED)
    got = phase5.payoff_door_split(units, CONTRIBUTION, FIXED, SHARE)
    assert np.allclose(got, (1 - SHARE) * net)


def test_guarantee_or_split_is_exactly_the_minimum_of_the_other_two():
    """The identity the report's central claim rests on, over the whole range.

    The artist takes whichever of their two outcomes is greater on every night, so the
    promoter takes whichever of theirs is smaller on every night. If this ever fails, the
    "you keep the worse of the two" argument in the report is wrong.
    """
    units = np.linspace(0, 2000, 2001)
    fixed = phase5.payoff_fixed_guarantee(units, CONTRIBUTION, FIXED, GUARANTEE)
    split = phase5.payoff_door_split(units, CONTRIBUTION, FIXED, SHARE)
    either = phase5.payoff_guarantee_or_split(units, CONTRIBUTION, FIXED, GUARANTEE, SHARE)
    assert np.allclose(either, np.minimum(fixed, split))


@pytest.mark.parametrize("guarantee", [500.0, 2000.0, 9000.0])
@pytest.mark.parametrize("share", [0.5, 0.7, 0.85])
def test_the_minimum_identity_holds_for_any_fee_and_any_share(guarantee, share):
    units = np.linspace(0, 3000, 601)
    fixed = phase5.payoff_fixed_guarantee(units, CONTRIBUTION, FIXED, guarantee)
    split = phase5.payoff_door_split(units, CONTRIBUTION, FIXED, share)
    either = phase5.payoff_guarantee_or_split(units, CONTRIBUTION, FIXED, guarantee, share)
    assert np.allclose(either, np.minimum(fixed, split))


def test_below_the_crossover_it_behaves_like_the_flat_fee_and_above_like_the_split():
    crossover = phase5.crossover_net(GUARANTEE, SHARE)
    assert crossover == pytest.approx(GUARANTEE / SHARE)

    quiet = (crossover / 2 + FIXED) / CONTRIBUTION  # units giving Net = crossover / 2
    busy = (crossover * 2 + FIXED) / CONTRIBUTION

    assert phase5.payoff_guarantee_or_split(
        quiet, CONTRIBUTION, FIXED, GUARANTEE, SHARE
    ) == pytest.approx(phase5.payoff_fixed_guarantee(quiet, CONTRIBUTION, FIXED, GUARANTEE))
    assert phase5.payoff_guarantee_or_split(
        busy, CONTRIBUTION, FIXED, GUARANTEE, SHARE
    ) == pytest.approx(phase5.payoff_door_split(busy, CONTRIBUTION, FIXED, SHARE))


def test_at_the_crossover_all_three_structures_pay_the_same():
    units = (phase5.crossover_net(GUARANTEE, SHARE) + FIXED) / CONTRIBUTION
    fixed = phase5.payoff_fixed_guarantee(units, CONTRIBUTION, FIXED, GUARANTEE)
    split = phase5.payoff_door_split(units, CONTRIBUTION, FIXED, SHARE)
    either = phase5.payoff_guarantee_or_split(units, CONTRIBUTION, FIXED, GUARANTEE, SHARE)
    assert fixed == pytest.approx(split)
    assert either == pytest.approx(fixed)


def test_every_structure_pays_more_when_more_tickets_are_sold():
    units = np.linspace(0, 2000, 401)
    for payoff in (
        phase5.payoff_fixed_guarantee(units, CONTRIBUTION, FIXED, GUARANTEE),
        phase5.payoff_door_split(units, CONTRIBUTION, FIXED, SHARE),
        phase5.payoff_guarantee_or_split(units, CONTRIBUTION, FIXED, GUARANTEE, SHARE),
    ):
        assert np.all(np.diff(payoff) >= -1e-9)


def test_the_split_deal_caps_your_upside_and_the_guarantee_does_not():
    """Slope check: you keep all of the last ticket on a guarantee, 30% of it on a split."""
    slope_fixed = np.diff(
        phase5.payoff_fixed_guarantee(np.array([1000.0, 1001.0]), CONTRIBUTION, FIXED, GUARANTEE)
    )
    slope_split = np.diff(
        phase5.payoff_door_split(np.array([1000.0, 1001.0]), CONTRIBUTION, FIXED, SHARE)
    )
    assert slope_fixed[0] == pytest.approx(CONTRIBUTION)
    assert slope_split[0] == pytest.approx((1 - SHARE) * CONTRIBUTION)


def test_break_even_is_the_units_at_which_the_guarantee_deal_makes_exactly_zero():
    """SPEC.md 6.3's formula and its meaning have to be the same statement."""
    break_even = (FIXED + GUARANTEE) / CONTRIBUTION
    assert phase5.payoff_fixed_guarantee(
        break_even, CONTRIBUTION, FIXED, GUARANTEE
    ) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# The cost stack
# --------------------------------------------------------------------------------------


def test_panel_excludes_cancelled_events_and_zero_sale_events(panel):
    data = synthetic.generate(seed=synthetic.DEFAULT_SEED)
    cancelled = set(data.events.loc[data.events["cancelled"], "event_id"])
    assert not (set(panel["event_id"]) & cancelled)
    assert (panel["units_sold"] > 0).all()


def test_break_even_matches_the_spec_formula(panel):
    expected = panel["fixed_cost_stack"] / panel["contribution_per_ticket"]
    assert np.allclose(panel["break_even_units"], expected)


def test_contribution_per_ticket_is_face_value_not_buyer_price(panel):
    """SPEC.md 8.6: the booking fee is the platform's. The promoter banks the face value."""
    assert np.allclose(
        panel["contribution_per_ticket"], panel["promoter_revenue"] / panel["units_sold"]
    )
    assert (panel["mean_buyer_price"] >= panel["mean_face_price"]).all()


def test_comps_are_not_counted_as_demand(panel):
    """The panel's units must be the paid units, not the room count."""
    data = synthetic.generate(seed=synthetic.DEFAULT_SEED)
    paid = data.transactions[~data.transactions["is_comp"]]
    expected = paid.groupby("event_id")["quantity"].sum()
    for event_id, units in panel.set_index("event_id")["units_sold"].items():
        assert units == expected[event_id]


# --------------------------------------------------------------------------------------
# The bootstrap
# --------------------------------------------------------------------------------------


def test_the_residual_distribution_comes_from_phase_2_out_of_fold_predictions():
    data = synthetic.generate(seed=synthetic.DEFAULT_SEED)
    forecast = phase5.demand_distribution(data.events, data.transactions)
    assert {"event_id", "predicted", "log_predicted", "log_resid"} <= set(forecast.columns)
    assert forecast["event_id"].is_unique
    assert np.isfinite(forecast["log_resid"]).all()
    # Out-of-fold residuals are not centred at zero by construction, unlike in-sample ones.
    assert forecast["log_resid"].std() > 0.05


def test_the_monte_carlo_resamples_real_residuals_and_fits_nothing():
    """The method claim, pinned (review finding, 2026-08-12).

    The report says "no normal distribution is fitted to anything" and VIVA Q5.2 is an
    entire answer about why not. Before this test, replacing `rng.choice(residuals)` with
    `rng.normal(0, residuals.std())` left the whole suite green — the claim was unpinned.

    An empirical bootstrap can only ever produce `exp(log_predicted + r)` for an `r` that
    is literally in the residual array, so the set of distinct simulated unit counts is a
    subset of a set with `len(residuals)` members. A parametric draw produces a continuum
    and fails immediately.
    """
    residuals = np.array([-0.55, -0.17, 0.11, 0.62])
    units = phase5.bootstrap_units(
        expected_log_units=6.0,
        residuals=residuals,
        rng=np.random.default_rng(0),
        n_draws=5000,
        capacity=1e9,  # no clipping, so the only thing shaping the values is the resampling
    )
    reachable = set(np.round(np.exp(6.0 + residuals)))
    assert set(np.unique(units)) == reachable
    assert len(np.unique(units)) == len(residuals), "every residual is used and nothing else is"


def test_simulate_event_draws_through_that_same_bootstrap(panel, results):
    """...and the per-event simulation must be the thing that calls it.

    Otherwise the test above pins a function nobody uses. Same seed, same residuals, same
    expected units — the two paths have to produce the identical draws.
    """
    row = panel.merge(
        results["panel"][["event_id", "log_predicted"]], on="event_id", how="inner"
    ).iloc[0]
    residuals = results["oos_residuals"]
    risk = phase5.simulate_event(
        row, float(row["log_predicted"]), residuals, np.random.default_rng(3), n_draws=800
    )
    units = phase5.bootstrap_units(
        float(row["log_predicted"]),
        residuals,
        np.random.default_rng(3),
        800,
        float(row["capacity"]),
    )
    assert risk.expected_units == pytest.approx(float(np.mean(units)))


def test_the_report_measures_its_monte_carlo_error_rather_than_asserting_it(results):
    """Review finding, 2026-08-12: the old report claimed a precision it did not have.

    "4,000 draws gives a 5th percentile stable to about a pound" was false — re-running on
    a second seed moved one event by £184. The fix is that the number is now measured on
    every run, from two seeds over the same residuals, and printed. This test pins that the
    measurement exists, that it is real (a smaller draw count must move it), and that the
    old unmeasured wording is gone.
    """
    spread = results["mc_spread"]
    assert set(spread) == {"p05_fixed", "mean_profit_fixed", "p_clear"}
    assert all(v["max"] >= v["median"] >= 0.0 for v in spread.values())

    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase5.report(results, phaseio.resolve(args))
    assert "stable to about a pound" not in text
    assert "How much of this is just the random number generator?" in text
    assert f"£{spread['p05_fixed']['max']:,.0f}" in text

    # And the measurement has teeth: fewer draws must show up as a bigger wobble.
    coarse = _run_with_draws(300)["mc_spread"]
    assert coarse["mean_profit_fixed"]["max"] > spread["mean_profit_fixed"]["max"]


def test_simulated_units_never_exceed_the_room(panel, results):
    """You cannot sell a capacity you do not have, however good the draw.

    Review finding, 2026-08-12: this used to assert on `expected_units`, which is the MEAN
    of the draws — a broken clip that let one draw in a hundred through the roof would have
    passed it comfortably. The clip is a property of every draw, so it is asserted on every
    draw, on an event whose predicted demand is deliberately far above its capacity.
    """
    room = 400.0
    over_the_roof = phase5.bootstrap_units(
        expected_log_units=np.log(4 * room),  # a forecast of four times the room
        residuals=results["oos_residuals"],
        rng=np.random.default_rng(11),
        n_draws=5000,
        capacity=room,
    )
    assert over_the_roof.max() == room, "the clip must bind, or this test proves nothing"
    assert (over_the_roof <= room).all()
    assert (over_the_roof >= 0.0).all()

    # ...and through the real path, on a real event, with the same conclusion per draw.
    row = panel.merge(
        results["panel"][["event_id", "log_predicted"]], on="event_id", how="inner"
    ).iloc[0]
    draws = phase5.bootstrap_units(
        float(row["log_predicted"]),
        results["oos_residuals"],
        np.random.default_rng(3),
        2000,
        float(row["capacity"]),
    )
    assert draws.max() <= float(row["capacity"])
    for risk in results["risks"]:
        assert risk.expected_units <= risk.capacity + 1e-9


def test_p_clear_agrees_with_the_probability_the_guarantee_deal_makes_money(results):
    """P(Q > break-even) and P(profit > 0) are the same event, so they must agree."""
    for risk in results["risks"][:25]:
        assert risk.p_clear == pytest.approx(risk.profit["fixed_guarantee"]["p_positive"], abs=0.02)


def test_the_expected_profit_of_the_worse_deal_is_never_the_best_of_the_three(results):
    for risk in results["risks"]:
        assert (
            risk.profit["guarantee_or_split"]["mean"]
            <= min(risk.profit["fixed_guarantee"]["mean"], risk.profit["door_split"]["mean"]) + 1e-6
        )


def test_p_clear_falls_as_the_fee_rises(results):
    curve = results["example_curve"]
    assert (np.diff(curve["p_clear"].to_numpy()) <= 1e-9).all()
    assert (np.diff(curve["mean_profit"].to_numpy()) < 0).all()


def test_the_most_payable_fee_is_the_fee_that_makes_it_a_coin_flip(results):
    """`max_guarantee_at_target` is defined by a probability, so check the probability."""
    curve = results["example_curve"]
    row = results["summary"].set_index("event_id").loc[results["example_event"]]
    fee = float(row["max_guarantee_at_target"])
    nearest = curve.iloc[(curve["guarantee"] - fee).abs().argmin()]
    assert abs(float(nearest["p_clear"]) - phase5.P_CLEAR_TARGET) < 0.25


def test_a_price_change_moves_demand_only_through_beta(panel, results):
    """Phase 6's hook: raising prices with a negative beta must sell fewer tickets."""
    row = panel.merge(
        results["panel"][["event_id", "log_predicted"]], on="event_id", how="inner"
    ).iloc[0]
    rng = np.random.default_rng(0)
    base = phase5.simulate_event(row, float(row["log_predicted"]), results["oos_residuals"], rng)
    dearer = phase5.simulate_event(
        row,
        float(row["log_predicted"]),
        results["oos_residuals"],
        np.random.default_rng(0),
        log_price_change=np.log(1.2),
        beta=-0.8,
    )
    assert dearer.expected_units < base.expected_units
    assert dearer.contribution_per_ticket > base.contribution_per_ticket


# --------------------------------------------------------------------------------------
# The decision rule
# --------------------------------------------------------------------------------------


def _profit(fixed_mean, fixed_p05, split_mean) -> dict:
    return {
        "fixed_guarantee": {"mean": fixed_mean, "p05": fixed_p05, "p_positive": 0.5},
        "door_split": {"mean": split_mean, "p05": split_mean, "p_positive": 0.5},
        "guarantee_or_split": {
            "mean": min(fixed_mean, split_mean),
            "p05": fixed_p05,
            "p_positive": 0.5,
        },
    }


def test_the_split_is_recommended_when_it_pays_more():
    got = phase5.recommend_structure(_profit(fixed_mean=100, fixed_p05=-10, split_mean=500))
    assert got.startswith("door split")


def test_the_guarantee_is_recommended_when_it_pays_more_and_the_tail_is_survivable():
    got = phase5.recommend_structure(_profit(fixed_mean=900, fixed_p05=-200, split_mean=400))
    assert got.startswith("fixed guarantee")


def test_an_unsurvivable_tail_overrides_the_higher_expected_profit():
    """SPEC.md 6.3: risk-adjusted, not expected-value. The tail gets a veto."""
    got = phase5.recommend_structure(
        _profit(fixed_mean=900, fixed_p05=-9000, split_mean=400),
        max_survivable_loss=1500,
    )
    assert got.startswith("door split")
    assert "absorb" in got


# --------------------------------------------------------------------------------------
# Report and CLI
# --------------------------------------------------------------------------------------


def test_report_states_every_assumption_and_names_the_structures(results):
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase5.report(results, phaseio.resolve(args))
    assert "fixed guarantee" in text
    assert "door split" in text
    assert "whichever is greater" in text
    assert "Assumptions, all of them" in text
    assert "SPEC.md 8.6" in text


def test_the_cancellation_caveat_counts_the_events_and_invents_no_reason(results):
    """Review finding, 2026-08-12: the report used to state a fact about the real programme.

    It said "the three events that were pulled were pulled *because* they were selling
    badly". The count was true of one fixture, and the causal claim is not checkable at
    all: DECISIONS.md records that the cancelled events' presale histories were destroyed
    on-platform. The count must now come from the data, and the claim must say what cannot
    be known.
    """
    data = synthetic.generate(seed=synthetic.DEFAULT_SEED)
    assert results["n_cancelled"] == int(data.events["cancelled"].sum())

    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase5.report(results, phaseio.resolve(args))
    assert f"{results['n_cancelled']} events were pulled or downscaled" in text
    assert "three events that were pulled were pulled" not in text
    assert "destroyed on-platform" in text
    assert "not something this dataset can check" in text


def test_the_most_payable_fee_is_clamped_at_zero_and_says_why(results):
    """Review finding, 2026-08-12: the run printed `median payable fee @50% £-1,001`.

    A negative "largest fee you can pay" is not a number to take into a phone call. The
    quantile goes negative whenever the night is odds-against clearing before the artist
    is paid anything, and the honest reading of that is £0 with the reason attached.
    """
    row = pd.Series(
        {
            "event_id": "FAKE-EV-999",
            "capacity": 500.0,
            "contribution_per_ticket": 5.0,
            "fixed_stack_ex_artist": 40_000.0,  # a stack no plausible draw can cover
            "artist_guarantee": 1_000.0,
            "units_sold": 200.0,
        }
    )
    risk = phase5.simulate_event(
        row, np.log(200.0), results["oos_residuals"], np.random.default_rng(5), n_draws=500
    )
    assert risk.max_guarantee_at_target == 0.0
    assert risk.p_clear == 0.0

    assert (results["summary"]["max_guarantee_at_target"] >= 0).all()

    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase5.report(results, phaseio.resolve(args))
    assert "**£0 means no fee clears at this room**" in text
    assert "clamped at zero" in text


def test_the_bootstrap_seed_is_a_stated_assumption_like_every_other_number(results):
    """Review finding, 2026-08-12: the Monte Carlo RNG was a default argument, unstated.

    The assumptions table claims to list every number someone chose, and `--seed` re-rolls
    the fixture rather than the bootstrap — so a reader could not tell which seed produced
    a percentile. Both facts now appear in the report.
    """
    assert results["bootstrap_seed"] == phase5.BOOTSTRAP_SEED

    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase5.report(results, phaseio.resolve(args))
    assert f"| bootstrap RNG seed | {phase5.BOOTSTRAP_SEED} |" in text
    assert f"BOOTSTRAP_SEED = {phase5.BOOTSTRAP_SEED}" in text
    assert "re-rolls the synthetic fixture, NOT the Monte Carlo" in text


def test_report_uses_no_option_market_jargon(results):
    """DECISIONS/brief: model the structure correctly and let the reader spot it."""
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase5.report(results, phaseio.resolve(args)).lower()
    for word in [" option", "short call", "strike price", "premium", "convexity", "payoff diagram"]:
        assert word not in text, f"option jargon leaked into the report: {word!r}"


def test_cli_writes_a_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(phaseio, "SYNTHETIC_RESULTS_DIR", tmp_path)
    assert phase5.main(["--synthetic"]) == 0
    assert "median break-even" in capsys.readouterr().out
    assert (tmp_path / phase5.REPORT_FILENAME).exists()


def test_cli_real_run_is_refused_while_the_preregistration_is_a_draft(capsys):
    assert phase5.main(["--real"]) == 1
    assert "preregistration.md is DRAFT" in capsys.readouterr().err
