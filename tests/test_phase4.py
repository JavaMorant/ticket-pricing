"""Tests for Phase 4 — elasticity (SPEC.md 4).

Three of these are the ACCEPTANCE tests the synthetic fixture was built to hand over
(PROGRESS.md, Phase 0.5): the Phase 4 estimator has to reproduce them on data whose answer
we chose before it is allowed anywhere near real data.

    test_naive_pooled_regression_comes_out_wrong_signed      SPEC.md 4.1
    test_event_fe_with_lead_control_recovers_beta_true       SPEC.md 4.2
    test_plain_event_fe_is_biased_by_the_lead_time_confound  SPEC.md 4.3

The rest are unit tests on the verdict function, which is the piece that decides whether a
number gets published at all — so its criteria are tested against hand-built diagnostics
rather than only against the fixture that happens to pass them.
"""

from __future__ import annotations

import argparse
import operator
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

import tests.synthetic as fixtures
from pricing import dataset, phase4, phaseio, synthetic

SEEDS = [1, 42, synthetic.DEFAULT_SEED]
BETA_TOLERANCE = 0.15  # same tolerance the fixture ships with; see tests/test_synthetic.py


@lru_cache(maxsize=8)
def _run(seed: int) -> dict:
    args = argparse.Namespace(real=False, synthetic=True, seed=seed, no_write=True)
    return phase4.run(phaseio.resolve(args))


@lru_cache(maxsize=8)
def _panel(seed: int) -> pd.DataFrame:
    data = synthetic.generate(seed=seed)
    return phase4.build_panel(data.event_tier, data.events)


@pytest.fixture(scope="module")
def results() -> dict:
    return _run(synthetic.DEFAULT_SEED)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel(synthetic.DEFAULT_SEED)


# --------------------------------------------------------------------------------------
# The three acceptance tests
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_naive_pooled_regression_comes_out_wrong_signed(seed):
    """SPEC.md 4.1. The pooled regression must say higher prices cause more sales.

    Pinned, not tolerated. If this ever starts coming out negative on the fixture, either
    the generator stopped planting the endogeneity or the estimator stopped pooling — and
    both would silently remove the finding the whole project is built on.
    """
    est = phase4.naive_pooled(_panel(seed))
    assert est.beta > 0, f"seed {seed}: pooled beta {est.beta:+.3f} was supposed to be positive"
    assert est.ci_low > 0, "the wrong sign should be significant, not a coin flip"


@pytest.mark.parametrize("seed", SEEDS)
def test_event_fe_with_lead_control_recovers_beta_true(seed):
    """SPEC.md 4.2 + 4.3. Event dummies plus a lead-time control land on the truth."""
    beta_true = synthetic.PARAMS["beta_true"]
    est = phase4.event_fixed_effects(_panel(seed), control_lead_time=True)
    assert abs(est.beta - beta_true) < BETA_TOLERANCE, (
        f"seed {seed}: recovered {est.beta:+.3f} against a true {beta_true:+.3f}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_plain_event_fe_is_biased_by_the_lead_time_confound(seed):
    """SPEC.md 4.3. Fixed effects ALONE are not enough, and the fixture proves it.

    Early Bird is cheap AND early, so without the lead-time control the estimate charges
    the timing effect to the price and comes out too elastic. This test exists so nobody
    ever concludes that event dummies are the whole answer.
    """
    panel = _panel(seed)
    fe_only = phase4.event_fixed_effects(panel, control_lead_time=False)
    fe_lead = phase4.event_fixed_effects(panel, control_lead_time=True)
    assert fe_only.beta < fe_lead.beta - 0.2, (
        f"seed {seed}: plain FE {fe_only.beta:+.3f} should be visibly more elastic than "
        f"FE + lead {fe_lead.beta:+.3f} while lead_confound > 0"
    )


def test_switching_the_confound_off_makes_plain_fe_work():
    """The knob is real: set `lead_confound` to 0 and plain fixed effects recover beta."""
    data = synthetic.generate(seed=7, params={"lead_confound": 0.0})
    panel = phase4.build_panel(data.event_tier, data.events)
    est = phase4.event_fixed_effects(panel, control_lead_time=False)
    assert abs(est.beta - synthetic.PARAMS["beta_true"]) < BETA_TOLERANCE


# --------------------------------------------------------------------------------------
# The panel
# --------------------------------------------------------------------------------------


def test_panel_drops_cancelled_comp_zero_and_single_tier_rows(panel):
    data = synthetic.generate(seed=synthetic.DEFAULT_SEED)
    cancelled = set(data.events.loc[data.events["cancelled"], "event_id"])
    assert not (set(panel["event_id"]) & cancelled), "cancelled events must not be in the panel"
    assert (panel["units_sold"] > 0).all()
    assert (panel["price"] > 0).all()
    assert panel.groupby("event_id").size().min() >= 2


def test_panel_log_columns_are_the_logs_they_claim_to_be(panel):
    assert np.allclose(panel["log_q"], np.log(panel["units_sold"]))
    assert np.allclose(panel["log_p"], np.log(panel["price"]))
    assert np.allclose(panel["lead_term"], np.log1p(panel["lead_time_open_days"].clip(lower=0)))


def test_event_panel_is_one_row_per_event():
    data = synthetic.generate(seed=synthetic.DEFAULT_SEED)
    ep = phase4.build_event_panel(data.event_tier, data.events)
    assert ep["event_id"].is_unique
    cancelled = set(data.events.loc[data.events["cancelled"], "event_id"])
    assert not (set(ep["event_id"]) & cancelled)
    assert (ep["units"] > 0).all()


def test_event_panel_log_price_is_the_unweighted_ladder_mean():
    """Not quantity-weighted, on purpose: the weights would be the outcome variable."""
    data = synthetic.generate(seed=42)
    ep = phase4.build_event_panel(data.event_tier, data.events)
    tiers = data.event_tier[(data.event_tier["units_sold"] > 0) & (data.event_tier["price"] > 0)]
    one = ep.iloc[0]["event_id"]
    expected = float(np.mean(np.log(tiers.loc[tiers["event_id"] == one, "price"])))
    assert ep.loc[ep["event_id"] == one, "log_p"].iloc[0] == pytest.approx(expected)


# --------------------------------------------------------------------------------------
# Clustered standard errors
# --------------------------------------------------------------------------------------


def test_standard_errors_are_clustered_by_event(panel):
    """Clustering must actually change the numbers, or it is not being applied."""
    import statsmodels.formula.api as smf

    plain = smf.ols("log_q ~ log_p", data=panel).fit()
    clustered = phase4.naive_pooled(panel)
    assert clustered.beta == pytest.approx(float(plain.params["log_p"]))
    assert clustered.se != pytest.approx(float(plain.bse["log_p"]), rel=1e-3)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("control_lead_time", [True, False])
def test_the_within_transform_and_the_dummies_are_the_same_regression(seed, control_lead_time):
    """Frisch-Waugh-Lovell, pinned.

    `event_fixed_effects` demeans by event; `event_fixed_effects_dummies` puts a dummy per
    event on the right-hand side. They are the same regression and must give the same
    slope. If they ever diverge, the demeaning is wrong (a groupby on the wrong key, say)
    and every number downstream of beta is wrong with it.
    """
    panel = _panel(seed)
    within = phase4.event_fixed_effects(panel, control_lead_time=control_lead_time)
    dummies = phase4.event_fixed_effects_dummies(panel, control_lead_time=control_lead_time)
    assert within.beta == pytest.approx(dummies.beta, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_within_estimator_has_far_more_clusters_than_parameters(seed):
    """The reason the within form is the one that is quoted (review finding, 2026-08-12).

    A cluster-robust covariance matrix is a sum of one outer product per cluster, so its
    rank is at most G-1. The dummy form asks for more parameters than there are clusters,
    which is not a footing an interval can stand on; the within form asks for two.
    """
    panel = _panel(seed)
    within = phase4.event_fixed_effects(panel, control_lead_time=True)
    dummies = phase4.event_fixed_effects_dummies(panel, control_lead_time=True)

    assert within.n_params == 2, "within: log_p and lead_term, no intercept, no dummies"
    assert within.n_params < within.n_events - 1, "params must sit well below G-1"
    assert dummies.n_params > dummies.n_events, (
        "the dummy form is supposed to be the one with more parameters than clusters — if "
        "that stops being true this test is no longer describing the problem it guards"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_the_interval_uses_a_t_on_g_minus_one_clusters_not_a_normal(seed):
    """Cluster-robust inference has G independent pieces, not N. The interval must say so."""
    from scipy import stats

    est = phase4.event_fixed_effects(_panel(seed), control_lead_time=True)
    critical = est.ci_width / (2 * est.se)
    assert critical == pytest.approx(stats.t.ppf(0.975, est.n_events - 1), abs=1e-6)
    assert critical > phase4.Z_ALPHA, "a t on G-1 df must be wider than the normal"


# --------------------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------------------


def test_vif_is_exactly_one_over_the_residual_price_share(panel):
    """They are the same statement in different units, and the report says so."""
    d = phase4.collinearity_diagnostics(panel)
    assert d["vif_price"] == pytest.approx(1.0 / d["residual_price_share"])


def test_residual_price_share_is_between_zero_and_one(panel):
    d = phase4.collinearity_diagnostics(panel)
    assert 0.0 < d["residual_price_share"] < 1.0
    assert d["price_share_explained_by_event"] < d["price_share_explained_by_event_and_lead"]


def test_price_and_lead_time_are_strongly_negatively_correlated_within_event(panel):
    """SPEC.md 4.3 in one number: cheap tiers are early tiers."""
    d = phase4.collinearity_diagnostics(panel)
    assert d["corr_price_lead_within_event"] < -0.5


# --------------------------------------------------------------------------------------
# The verdict function — tested against hand-built diagnostics, not only the fixture
# --------------------------------------------------------------------------------------


def _estimate(beta=-0.8, ci=(-0.95, -0.65), n_events=100) -> phase4.Estimate:
    return phase4.Estimate(
        name="test",
        formula="",
        beta=beta,
        se=0.05,
        ci_low=ci[0],
        ci_high=ci[1],
        n_rows=n_events * 3,
        n_events=n_events,
        n_params=2,
    )


def _diagnostics(residual_share=0.10, corr=-0.7) -> dict:
    return {
        "residual_price_share": residual_share,
        "vif_price": 1.0 / residual_share,
        "corr_price_lead_within_event": corr,
    }


def test_verdict_is_identified_when_every_criterion_passes():
    v = phase4.identification_verdict(_estimate(), _diagnostics())
    assert v.verdict == "IDENTIFIED"
    assert v.identified is True
    assert v.failures == []


def test_verdict_fails_when_no_price_variation_survives_the_controls():
    """The criterion that matters most: with nothing left to fit, nothing is identified."""
    v = phase4.identification_verdict(_estimate(), _diagnostics(residual_share=0.005))
    assert v.verdict == "NOT-IDENTIFIED"
    assert [c.name for c in v.failures] == ["residual price variation"]


def test_verdict_fails_when_the_interval_is_too_wide():
    v = phase4.identification_verdict(_estimate(ci=(-1.9, -0.1)), _diagnostics())
    assert v.verdict == "NOT-IDENTIFIED"
    assert "95% CI width" in [c.name for c in v.failures]


def test_verdict_fails_when_the_interval_straddles_zero():
    v = phase4.identification_verdict(_estimate(beta=-0.2, ci=(-0.5, 0.1)), _diagnostics())
    assert v.verdict == "NOT-IDENTIFIED"
    assert "CI lies below zero" in [c.name for c in v.failures]


def test_verdict_fails_on_too_few_events():
    v = phase4.identification_verdict(_estimate(n_events=12), _diagnostics())
    assert v.verdict == "NOT-IDENTIFIED"
    assert "events with 2+ priced tiers" in [c.name for c in v.failures]


def test_verdict_reports_every_failure_not_just_the_first():
    v = phase4.identification_verdict(
        _estimate(ci=(-2.0, 0.4), n_events=5), _diagnostics(residual_share=0.001)
    )
    assert len(v.failures) == 4


def test_a_high_vif_is_flagged_but_does_not_fail_the_verdict():
    """A VIF only says the SE is inflated. Whether the interval is usable is the CI test."""
    v = phase4.identification_verdict(_estimate(), _diagnostics(residual_share=0.05))
    assert v.verdict == "IDENTIFIED"
    assert any("VIF" in f for f in v.flags)


def test_near_perfect_price_timing_collinearity_is_flagged():
    v = phase4.identification_verdict(_estimate(), _diagnostics(corr=-0.97))
    assert any("price and timing" in f for f in v.flags)


def test_criteria_can_be_overridden_and_the_verdict_follows():
    """The thresholds are arguments, so a reviewer can re-run the verdict at their own."""
    estimate, diagnostics = _estimate(), _diagnostics(residual_share=0.03)
    assert phase4.identification_verdict(estimate, diagnostics).identified
    stricter = phase4.identification_verdict(
        estimate, diagnostics, criteria={"min_residual_price_share": 0.20}
    )
    assert not stricter.identified


def test_the_synthetic_fixture_is_identified(results):
    """It has to be, or Phase 6's point-uplift branch would never be exercised."""
    assert results["verdict"].verdict == "IDENTIFIED"


def test_the_sign_criterion_is_carried_as_a_different_kind_of_check():
    """Review finding, 2026-08-12: three of the four criteria are about the DESIGN.

    Residual variation, interval width and event count can all be evaluated without ever
    looking at what the estimate came out as. "CI lies below zero" cannot — it conditions
    on the sign. Labelling it in the data structure is what lets the report print it under
    its own heading instead of smuggling it in as if it were the same kind of thing.
    """
    v = phase4.identification_verdict(_estimate(), _diagnostics())
    design = [c.name for c in v.checks if c.kind == "design"]
    sign = [c.name for c in v.checks if c.kind == "sign"]
    assert design == ["residual price variation", "95% CI width", "events with 2+ priced tiers"]
    assert sign == ["CI lies below zero"]


def test_every_criterion_prints_the_comparison_it_actually_made():
    """Review finding, 2026-08-12: the table inferred the comparator from the check's NAME.

    "CI lies below zero" is `ci_high < 0`, but the table guessed `<=` and rendered
    `<= 0.000`, so a reader reproducing the verdict by hand got a different rule at the
    boundary. The comparator is carried on the Check now, and this asserts that applying
    the printed comparison to the printed numbers reproduces the printed PASS/FAIL.
    """
    comparisons = {
        ">=": operator.ge,
        "<=": operator.le,
        "<": operator.lt,
        ">": operator.gt,
    }
    v = phase4.identification_verdict(_estimate(), _diagnostics())
    for chk in v.checks:
        assert comparisons[chk.comparator](chk.value, chk.threshold) is chk.passed

    rendered = "\n".join(phase4._criteria_table(v.checks))
    assert "| CI lies below zero | -0.650 | < 0.000 | PASS |" in rendered

    # ...and the boundary case the old label got wrong: ci_high exactly 0 is a FAIL.
    boundary = phase4.identification_verdict(_estimate(beta=-0.2, ci=(-0.4, 0.0)), _diagnostics())
    assert [c.name for c in boundary.failures] == ["CI lies below zero"]


def test_a_positive_estimate_fails_only_the_sign_check_and_the_report_says_so(results):
    """The exact case the criticism is about: a well-designed study with the wrong sign.

    Plenty of residual variation, a tight interval, 100 events — every design criterion
    passes — and a positive beta. The verdict is NOT-IDENTIFIED, and the only failure is
    the check that conditions on the sign. The report must not present that as if the
    design had failed.
    """
    positive = _estimate(beta=0.30, ci=(0.18, 0.42))
    v = phase4.identification_verdict(positive, _diagnostics())
    assert v.verdict == "NOT-IDENTIFIED"
    assert [c.name for c in v.failures] == ["CI lies below zero"]
    assert all(c.passed for c in v.checks if c.kind == "design")

    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase4.report({**results, "verdict": v, "identified": False}, phaseio.resolve(args))
    assert "### Design criteria — can this data answer the question?" in text
    assert "### A sign check, stated as such" in text
    assert "conditions on the sign of the answer" in text
    assert "not the same as p-hacking the sign" in text


# --------------------------------------------------------------------------------------
# The instrument
# --------------------------------------------------------------------------------------


def test_iv_recovers_a_known_slope_where_ols_cannot():
    """The IV arithmetic, checked on three lines of hand-built data with a known answer.

    y = 1 - 0.8 p + u, price contaminated by u (so OLS is biased upward), and an
    instrument z that moves price and is independent of u. IV must land near -0.8 and OLS
    must not.
    """
    rng = np.random.default_rng(0)
    n = 4000
    u = rng.normal(size=n)
    z = rng.normal(size=n)
    p = 0.8 * z + 1.2 * u + rng.normal(scale=0.3, size=n)
    y = 1.0 - 0.8 * p + u
    df = pd.DataFrame({"log_q": y, "log_p": p, "z": z, "const": 1.0})
    result = phase4.iv_estimate(df, "log_q", "log_p", "z", "1", "unit test", proxies=())
    assert result.relevant
    assert result.beta == pytest.approx(-0.8, abs=0.1)
    assert result.ols_beta > -0.5, "OLS should be visibly biased here, or the test is toothless"
    assert result.ci_low < -0.8 < result.ci_high


def test_a_weak_instrument_is_refused_rather_than_reported():
    """Below the F floor, no number comes out. A weak IV is worse than no IV."""
    rng = np.random.default_rng(1)
    n = 300
    u = rng.normal(size=n)
    z = rng.normal(size=n)
    p = 0.01 * z + u  # z barely moves price
    y = 1.0 - 0.8 * p + u
    df = pd.DataFrame({"log_q": y, "log_p": p, "z": z})
    result = phase4.iv_estimate(df, "log_q", "log_p", "z", "1", "weak", proxies=())
    assert not result.relevant
    assert result.beta is None
    assert "TOO WEAK" in result.note


def test_an_event_with_a_missing_control_leaves_the_sample_instead_of_corrupting_it():
    """Review finding, 2026-08-12: only y, p and z were dropped on, never the controls.

    statsmodels' default is `missing="drop"`, so one null venue used to give a residual
    vector one row shorter than the other two — and `z~ . p~` then silently dotted two
    different sets of events together. It cannot happen on the fixture; it happens on the
    first real export with a venue nobody filled in.
    """
    rng = np.random.default_rng(3)
    n = 200
    z = rng.normal(size=n)
    p = 0.9 * z + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame(
        {
            "log_q": 1.0 - 0.8 * p + rng.normal(scale=0.2, size=n),
            "log_p": p,
            "z": z,
            "venue": rng.choice(["V1", "V2"], size=n),
        }
    )
    df.loc[0, "venue"] = None

    result = phase4.iv_estimate(df, "log_q", "log_p", "z", "C(venue)", "null venue", proxies=())
    assert result.n == n - 1, "the incomplete row must leave the sample, once, explicitly"
    assert result.beta == pytest.approx(-0.8, abs=0.15)

    # ...and the tripwire under it: residualising a frame that still has a hole raises
    # rather than quietly returning a shorter vector.
    with pytest.raises(Exception, match="[Nn]a[Nn]|missing"):
        phase4._residualise(df, "log_p", "C(venue)")


def test_the_first_stage_f_pays_for_the_degrees_of_freedom_the_controls_used():
    """Review finding, 2026-08-12: F was t^2 from a regression that counted n, not n - k.

    The residualised first stage is fitted on n rows, but the controls already spent
    k parameters, so the naive t^2 overstates the F — in the direction that lets a weak
    instrument through the relevance floor.
    """
    rng = np.random.default_rng(4)
    n = 120
    z = rng.normal(size=n)
    p = 0.5 * z + rng.normal(size=n)
    venue = rng.choice([f"V{i}" for i in range(20)], size=n)  # 20 dummies = 20 parameters
    df = pd.DataFrame({"log_q": 1.0 - 0.8 * p, "log_p": p, "z": z, "venue": venue})

    import statsmodels.formula.api as smf

    result = phase4.iv_estimate(df, "log_q", "log_p", "z", "C(venue)", "df test", proxies=())

    residualised = pd.DataFrame(
        {
            "p": phase4._residualise(df, "log_p", "C(venue)"),
            "z": phase4._residualise(df, "z", "C(venue)"),
        }
    )
    uncorrected = float(
        smf.ols("p ~ z - 1", data=residualised).fit(cov_type="HC1").tvalues["z"] ** 2
    )
    k_controls = 20  # 19 venue dummies + the intercept
    assert result.first_stage_f == pytest.approx(uncorrected * (n - k_controls - 1) / n)
    assert result.first_stage_f < uncorrected, "the correction can only ever lower the F"


def test_the_prespecified_cost_shifter_is_reported_either_way(results):
    """Whatever the fixture does, spec I must appear with its first-stage F attached."""
    specs = results["instruments"]
    assert specs, "the cost-shifter scaffold must run"
    assert "PRE-SPECIFIED" in specs[0].label
    assert np.isfinite(specs[0].first_stage_f)
    assert (specs[0].beta is None) == (specs[0].first_stage_f < phase4.FIRST_STAGE_F_FLOOR)


def test_the_invalid_instrument_is_strong_and_wrong(results):
    """SPEC.md 4.5's lesson, pinned: a strong first stage says nothing about validity.

    The artist guarantee moves price hard (bigger artist, bigger fee, higher ticket) and
    moves demand directly, so it fails exclusion by construction. On the fixture it has
    the strongest first stage of the three specs and the least usable estimate.
    """
    guarantee = [s for s in results["instruments"] if "guarantee" in s.instrument]
    assert guarantee, "the invalid instrument must stay in the report"
    invalid = guarantee[0]
    assert invalid.first_stage_f > phase4.FIRST_STAGE_F_FLOOR
    assert invalid.beta is not None and invalid.beta > 0, (
        "an instrument that fails exclusion should give an obviously wrong answer here"
    )


# --------------------------------------------------------------------------------------
# The report and the CLI
# --------------------------------------------------------------------------------------


def test_report_contains_all_four_estimates_and_the_verdict(results):
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase4.report(results, phaseio.resolve(args))
    assert "naive pooled OLS" in text
    assert "event FE + lead-time control" in text
    assert results["verdict"].verdict in text
    assert "experiment/DESIGN.md" in text, "the report must always point at the experiment"
    assert "SPEC.md 8.6" in text, "the fee-treatment caveat travels with every price number"


def test_report_prints_both_fixed_effects_intervals_and_says_which_one_binds(results):
    """Switching to the within transform narrowed the interval, so the report must show both.

    A reader who is told only the narrower number has no way to check that the estimator
    choice did not buy the verdict. The dummy row, its width, and an explicit statement
    about whether the choice could have changed the answer all have to be on the page.
    """
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase4.report(results, phaseio.resolve(args))
    within, dummies = results["fe_lead"], results["fe_lead_dummies"]

    assert "3b. the same, written with event dummies" in text
    assert f"{dummies.ci_low:+.3f}" in text and f"{dummies.ci_high:+.3f}" in text
    assert f"width {within.ci_width:.3f}" in text
    assert f"width {dummies.ci_width:.3f}" in text
    assert "cannot have bought the verdict" in text or "DOES move the verdict" in text
    assert "areg" in text, "name the areg/xtreg disagreement rather than hiding the choice"


def test_report_of_a_not_identified_run_states_the_spec_44_wording(results):
    """The NOT-IDENTIFIED branch is first-class, so it is rendered and tested directly."""
    broken = dict(results)
    broken["verdict"] = phase4.identification_verdict(
        results["fe_lead"], {**results["diagnostics"], "residual_price_share": 0.001}
    )
    broken["identified"] = False
    args = argparse.Namespace(
        real=False, synthetic=True, seed=synthetic.DEFAULT_SEED, no_write=True
    )
    text = phase4.report(broken, phaseio.resolve(args))
    assert "NOT-IDENTIFIED is the result, not a missing result" in text
    assert "randomisation required to recover it" in text
    assert "experiment/DESIGN.md" in text


def test_cli_writes_a_report_to_the_synthetic_results_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(phaseio, "SYNTHETIC_RESULTS_DIR", tmp_path)
    assert phase4.main(["--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "VERDICT" in out
    assert (tmp_path / phase4.REPORT_FILENAME).exists()


@pytest.mark.parametrize(
    ("gate", "expected"),
    [("DRAFT", "preregistration.md is DRAFT"), ("FROZEN", "no derived tables")],
)
def test_cli_real_run_is_refused(gate, expected, capsys, tmp_path, monkeypatch):
    """SPEC.md 5.3, both gates (see test_phase1 for the full note). Refusal on STDERR."""
    if gate == "DRAFT":
        monkeypatch.setattr(dataset, "PREREGISTRATION", fixtures.draft_preregistration(tmp_path))
    assert phase4.main(["--real"]) == 1
    captured = capsys.readouterr()
    assert "refused" in captured.err
    assert expected in captured.err
    assert "refused" not in captured.out
    assert expected not in captured.out
