"""Tests for Phase 6 — the counterfactual (SPEC.md 6.4).

Two of these carry more weight than the rest:

    test_the_true_process_and_the_model_agree_on_the_uplift
        the pinned sign-and-magnitude check against the generator's own demand process,
        with a wide tolerance on purpose.

    test_a_not_identified_verdict_removes_the_headline_number
        the honesty rule. If Phase 4 says NOT-IDENTIFIED there is no uplift number, only a
        curve over assumed elasticities. That branch is rendered and asserted here so it
        cannot rot while the synthetic fixture keeps passing its identification checks.
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from pricing import phase4, phase5, phase6, phaseio, synthetic

SEEDS = [42, synthetic.DEFAULT_SEED]


@lru_cache(maxsize=4)
def _run(seed: int) -> dict:
    args = argparse.Namespace(real=False, synthetic=True, seed=seed, no_write=True)
    return phase6.run(phaseio.resolve(args))


@lru_cache(maxsize=4)
def _panel(seed: int) -> pd.DataFrame:
    data = synthetic.generate(seed=seed)
    return phase6.build_panel(data.event_tier, data.events)


def _run_data(seed: int = synthetic.DEFAULT_SEED) -> phaseio.Run:
    return phaseio.resolve(argparse.Namespace(real=False, synthetic=True, seed=seed, no_write=True))


# A bold markdown percentage: `**+5.3%**`. The token that survives a copy-paste out of the
# report and into a CV, which is why the NOT-IDENTIFIED branch is not allowed to print one
# without a qualifier on the same line.
BOLD_PERCENT = re.compile(r"\*\*[+-]?\d+(?:\.\d+)?%\*\*")
# A row of the assumed-beta curve: `| -1.5 | **+17.1%** | ...`. Every one of these is
# labelled with the elasticity it assumes, in the first cell, so a bold number there is
# already conditional by construction.
CURVE_ROW = re.compile(r"^\|\s*[+-]\d\.\d\s*\|")


@pytest.fixture(scope="module")
def results() -> dict:
    return _run(synthetic.DEFAULT_SEED)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel(synthetic.DEFAULT_SEED)


# --------------------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------------------


def test_the_do_nothing_grid_point_changes_nothing(panel):
    """At (level 1.00, spread 1.00) the counterfactual must reproduce what happened.

    The cheapest possible check that the machinery is not quietly adding something, and
    the reason 1.00 is on both grids.
    """
    delta = phase6.ladder_change(panel, 1.0, 1.0)
    assert np.allclose(delta, 0.0)
    got = phase6.counterfactual_contribution(panel, delta, beta=-0.8)
    actual = panel.groupby("event_id")["actual_contribution"].sum()
    assert np.allclose(got.sort_index(), actual.sort_index())


def test_contribution_means_what_phase_5_says_it_means(monkeypatch):
    """Review finding, 2026-08-12: Phase 6 hardcoded face value as contribution.

    Correct today, because `VARIABLE_COST_PER_TICKET` is 0.00 — and silently wrong the
    moment a real per-head cost lands, at which point Phase 5 and Phase 6 would be using
    the same word for two different numbers. Both sides of Phase 6's arithmetic now read
    Phase 5's constant, so moving it moves them.
    """
    data = synthetic.generate(seed=synthetic.DEFAULT_SEED)
    before = phase6.build_panel(data.event_tier, data.events)

    monkeypatch.setattr(phase5, "VARIABLE_COST_PER_TICKET", 1.50)
    after = phase6.build_panel(data.event_tier, data.events)

    expected = before["actual_contribution"] - 1.50 * before["units_sold"]
    assert np.allclose(after["actual_contribution"], expected)

    # ...and the counterfactual side, at the do-nothing point, moves with it identically.
    delta = phase6.ladder_change(after, 1.0, 1.0)
    got = phase6.counterfactual_contribution(after, delta, beta=-0.8)
    assert np.allclose(
        got.sort_index(), after.groupby("event_id")["actual_contribution"].sum().sort_index()
    )
    assert got.sum() < before.groupby("event_id")["actual_contribution"].sum().sum()


def test_level_moves_every_rung_by_the_same_proportion(panel):
    delta = phase6.ladder_change(panel, 1.10, 1.0)
    assert np.allclose(delta, np.log(1.10))


def test_spread_zero_collapses_the_ladder_to_one_price(panel):
    delta = phase6.ladder_change(panel, 1.0, 0.0)
    new_log_price = panel["log_p"].to_numpy() + delta
    spread_within_event = pd.Series(new_log_price).groupby(panel["event_id"]).std().fillna(0.0)
    assert spread_within_event.max() < 1e-9


def test_with_a_zero_elasticity_contribution_scales_with_the_price(panel):
    """beta = 0 means quantity never moves, so revenue is exactly proportional to price."""
    actual = panel.groupby("event_id")["actual_contribution"].sum()
    got = phase6.counterfactual_contribution(panel, phase6.ladder_change(panel, 1.2, 1.0), beta=0.0)
    assert np.allclose(got.sort_index(), actual.sort_index() * 1.2)


def test_capacity_is_a_hard_constraint(panel):
    """Cut prices far enough with elastic demand and the room, not the model, binds."""
    delta = phase6.ladder_change(panel, 0.70, 1.0)
    units = panel["units_sold"].to_numpy() * np.exp(-2.0 * delta)
    uncapped = pd.Series(units).groupby(panel["event_id"]).sum()
    capacity = panel.groupby("event_id")["capacity"].first()
    assert (uncapped > capacity).any(), "the test is toothless unless the cap actually binds"

    contribution = phase6.counterfactual_contribution(panel, delta, beta=-2.0)
    face_new = panel["face_price"].to_numpy() * np.exp(delta)
    max_price = pd.Series(face_new).groupby(panel["event_id"]).max()
    assert (contribution <= capacity * max_price + 1e-6).all()


def test_the_grid_is_every_combination_and_nothing_else(panel):
    grid = phase6.grid_search(panel, beta=-0.8)
    n_events = panel["event_id"].nunique()
    assert len(grid) == n_events * len(phase6.LEVEL_GRID) * len(phase6.SPREAD_GRID)
    assert set(grid["level"]) == set(phase6.LEVEL_GRID)
    assert set(grid["spread"]) == set(phase6.SPREAD_GRID)


def test_the_optimum_can_never_be_worse_than_doing_nothing(panel):
    """(1.00, 1.00) is on the grid, so the search cannot lose money by construction."""
    for beta in (-0.4, -0.8, -1.0, -1.6):
        best = phase6.optimise(panel, beta)
        assert best["uplift"] >= -1e-9
        assert (best["per_event"]["uplift"] >= -1e-9).all()


def test_the_uplift_is_smallest_at_unit_elasticity(panel):
    """Arithmetic, not a finding: at beta = -1 the level of prices does not move revenue."""
    curve = phase6.uplift_curve(panel, betas=(-0.6, -0.8, -1.0, -1.2, -1.5))
    at_unit = curve.loc[curve["assumed_beta"] == -1.0, "uplift"].iloc[0]
    assert at_unit == curve["uplift"].min()
    assert at_unit < 0.02


def test_inelastic_demand_wants_higher_prices_and_elastic_demand_wants_lower(panel):
    """The two halves of the curve are contradictory advice, which is why beta matters."""
    curve = phase6.uplift_curve(panel, betas=(-0.4, -1.6))
    assert curve.loc[curve["assumed_beta"] == -0.4, "modal_level"].iloc[0] > 1.0
    assert curve.loc[curve["assumed_beta"] == -1.6, "modal_level"].iloc[0] < 1.0


def test_uplift_at_a_stated_dose_needs_no_optimisation(panel):
    """The number that survives the honesty rules: one pre-committed price change."""
    got = phase6.uplift_at(panel, beta=0.0, level=1.10)
    assert got == pytest.approx(0.10)


# --------------------------------------------------------------------------------------
# The pinned ground-truth check
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_the_true_process_and_the_model_agree_on_the_uplift(seed):
    """PINNED. Push the chosen ladders through the generator's own demand process.

    The model applies one elasticity tier by tier; the generator reallocates shares with a
    softmax and moves the event total with the average log price, then rounds and caps.
    They are different arithmetic, so exact agreement would be suspicious — what is
    required is that they agree in SIGN and in rough MAGNITUDE. The tolerance is wide on
    purpose: this test is asking "does the simplification throw the answer away", and the
    answer has to be no by a margin that survives a re-seed.
    """
    results = _run(seed)
    check = results["ground_truth_check"]
    assert check is not None
    assert check["same_sign"], (
        f"seed {seed}: model {check['model_uplift']:+.2%} and true process "
        f"{check['true_uplift']:+.2%} disagree about the direction"
    )
    assert 0.25 <= check["ratio"] <= 4.0, f"seed {seed}: ratio {check['ratio']:.2f} is not close"
    assert abs(check["model_uplift"] - check["true_uplift"]) < 0.15


def test_a_uniform_price_change_is_the_one_case_where_the_two_processes_must_agree_exactly():
    """Both reduce to Q * m^beta when every rung moves by the same factor.

    Worth pinning, because it localises any future disagreement to the ladder's SHAPE —
    which is exactly where the per-tier elasticity is the strongest assumption made.
    """
    data = synthetic.generate(seed=synthetic.DEFAULT_SEED)
    panel = phase6.build_panel(data.event_tier, data.events)
    beta_true = float(data.truth["beta_true"])
    delta = phase6.ladder_change(panel, 1.10, 1.0)

    model = phase6.counterfactual_contribution(panel, delta, beta_true, max_sell_through=10.0)
    truth = phase6.true_process_contribution(panel, delta, data.truth, max_sell_through=10.0)
    # Rounding and the generator's 3-ticket floor are the only differences left.
    assert np.allclose(model.sort_index(), truth.sort_index(), rtol=0.02)


# --------------------------------------------------------------------------------------
# The honesty rule
# --------------------------------------------------------------------------------------


def test_the_synthetic_run_is_identified_so_the_point_uplift_branch_is_exercised(results):
    assert results["identified"] is True
    assert results["verdict"] == "IDENTIFIED"
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase6.report(results, phaseio.resolve(args))
    assert "projected uplift in contribution margin" in text
    assert "Projected, not realised" in text


def test_the_headline_is_the_pre_committed_dose_not_the_grid_corner(results):
    """Review finding, 2026-08-12: the headline was 100% a grid-boundary artefact.

    Every event picked the top corner of the level grid, so `## Headline: +5.3%` was a
    restatement of "the level grid stops at x1.30". The report body already said so; the
    headline did not. The number with no search in it — the uplift at the experiment's
    pre-committed +15% dose — leads instead, and the grid optimum is demoted to a bound.
    """
    best = results["best"]
    assert best["share_at_level_boundary"] > 0, "this fixture must exercise the corner branch"

    text = phase6.report(results, _run_data())
    headline = text.split("\n## ")[1]  # everything under the `## Headline:` heading

    assert f"**{results['uplift_at_experiment_dose']:+.1%} projected uplift" in headline
    assert f"**{best['uplift']:+.1%} projected uplift" not in text
    assert "Corner solution — this is the grid bound, not a result" in headline
    assert f"at least {best['uplift']:+.1%} within the" in headline
    assert f"{best['share_at_level_boundary']:.0%} of events pick a level at the edge" in headline

    # The terminal summary has to agree with the report about which number is the headline.
    summary = "\n".join(phase6.summary_lines(results))
    assert f"PROJECTED UPLIFT               {results['uplift_at_experiment_dose']:+.1%}" in summary
    assert f"grid optimum (a BOUND)         {best['uplift']:+.1%}" in summary


def _not_identified(results: dict) -> dict:
    """The same run with Phase 4's verdict forced to NOT-IDENTIFIED."""
    broken = dict(results)
    elasticity = dict(results["elasticity"])
    elasticity["verdict"] = phase4.identification_verdict(
        results["elasticity"]["fe_lead"],
        {**results["elasticity"]["diagnostics"], "residual_price_share": 0.001},
    )
    broken["elasticity"] = elasticity
    broken["identified"] = False
    broken["verdict"] = elasticity["verdict"].verdict
    return broken


def test_a_not_identified_verdict_removes_the_headline_number(results):
    """The rule that governs this module: no identified beta, no uplift number."""
    text = phase6.report(_not_identified(results), _run_data())
    assert "there is no headline uplift number" in text
    assert "PROVISIONAL" in text
    assert "experiment/DESIGN.md" in text
    assert "conditional on" in text.lower()
    assert "projected uplift in contribution margin**" not in text


def test_a_not_identified_report_never_prints_an_unqualified_bold_percentage(results):
    """Review finding, 2026-08-12. The old test only checked one exact string.

    The report refused a headline on line 8, labelled the point estimate PROVISIONAL on
    line 14 — and printed `- programme uplift: **+5.3%**` on line 48 with nothing attached.
    A section heading does not survive a copy-paste; a bold number does. So the rule is
    now structural: in the NOT-IDENTIFIED report, every bold percentage must either be a
    row of the assumed-beta curve (labelled with its assumption in the first cell) or sit
    on a line that says PROVISIONAL.
    """
    text = phase6.report(_not_identified(results), _run_data())
    offenders = [
        line
        for line in text.splitlines()
        if BOLD_PERCENT.search(line) and not CURVE_ROW.match(line) and "PROVISIONAL" not in line
    ]
    assert offenders == [], f"unqualified bold percentages in a NOT-IDENTIFIED report: {offenders}"

    # ...and the qualifier is on the numbers themselves, not only in the preamble.
    assert "PROVISIONAL — programme uplift at the grid optimum" in text
    assert "PROVISIONAL — THE HEADLINE NUMBER — uplift from the experiment's +15% dose" in text


def test_an_identified_report_still_prints_its_headline_in_bold(results):
    """The other half of the rule: the qualifier appears because the verdict says so.

    If suppressing the bold were unconditional, the honesty rule would have stopped being
    a rule and become formatting.
    """
    text = phase6.report(results, _run_data())
    assert f"grid optimum — a bound, see the headline: **{results['best']['uplift']:+.1%}**" in text
    assert "PROVISIONAL" not in text


@pytest.mark.parametrize("seed", SEEDS)
def test_the_contradictory_advice_sentence_is_read_off_the_curve(seed):
    """Review finding, 2026-08-12: two percentages were typed into the prose by hand.

    "+17% uplift at beta = -1.5 and a +12% uplift at beta = -0.6" was literal text. It was
    right on one seed by coincidence and wrong on the next, so the report contradicted its
    own table two paragraphs above. Both numbers now come off `results["curve"]`, and this
    test compares the sentence against the table on more than one seed — which is what a
    hand-typed number cannot survive.
    """
    results = _run(seed)
    curve = results["curve"]
    text = phase6.report(results, _run_data(seed))

    elastic = curve.loc[curve["assumed_beta"].idxmin()]
    inelastic = curve.loc[curve["assumed_beta"].idxmax()]
    assert (
        f"So the {elastic['uplift']:+.1%} uplift at beta = {elastic['assumed_beta']:+.1f}" in text
    )
    assert f"{inelastic['uplift']:+.1%} uplift at beta = {inelastic['assumed_beta']:+.1f}" in text
    # The advice attached to each has to be the advice the table gives, too.
    assert f"{elastic['uplift']:+.1%}" in _curve_row(text, elastic["assumed_beta"])
    assert f"{inelastic['uplift']:+.1%}" in _curve_row(text, inelastic["assumed_beta"])


def _curve_row(text: str, beta: float) -> str:
    """The rendered curve-table row for one assumed beta. Empty string if it is missing."""
    prefix = f"| {beta:+.1f} |"
    return next((line for line in text.splitlines() if line.startswith(prefix)), "")


def test_the_uplift_curve_is_reported_whatever_the_verdict(results):
    """The curve is not the fallback — it is always there, labelled with its assumption."""
    curve = results["curve"]
    assert list(curve["assumed_beta"]) == list(phase6.ASSUMED_BETAS)
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    for text in (
        phase6.report(results, phaseio.resolve(args)),
        phase6.report(_not_identified(results), phaseio.resolve(args)),
    ):
        assert "assumed beta" in text
        assert "experiment/DESIGN.md" in text


def test_the_report_admits_the_corner_solution_and_the_flat_ladder(results):
    """Both are arithmetic. A reader who spots them first assumes we did not understand."""
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase6.report(results, phaseio.resolve(args))
    assert "arithmetic, not discoveries" in text
    assert "flatten the ladder" in text
    assert "is an assumption" in text


def test_the_report_carries_the_fee_treatment_caveat(results):
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    assert "SPEC.md 8.6" in phase6.report(results, phaseio.resolve(args))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def test_cli_writes_a_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(phaseio, "SYNTHETIC_RESULTS_DIR", tmp_path)
    assert phase6.main(["--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "PROJECTED UPLIFT" in out
    assert "ground truth check" in out
    assert (tmp_path / phase6.REPORT_FILENAME).exists()


def test_cli_real_run_is_refused_while_the_preregistration_is_a_draft(capsys):
    assert phase6.main(["--real"]) == 1
    assert "preregistration.md is DRAFT" in capsys.readouterr().err
