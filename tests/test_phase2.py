"""Tests for Phase 2 — demand forecast.

The three that matter, in order of how easy it would be to fool yourself without them:

    test_groups_are_never_split_across_a_fold        SPEC.md 8.1
    test_lookahead_check_catches_a_planted_leak      SPEC.md 8.2
    test_model_beats_the_naive_baseline_out_of_sample the only reason to fit anything

The rest is arithmetic.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

import tests.synthetic as fixtures
from pricing import dataset, phase2, phaseio, synthetic


@lru_cache(maxsize=2)
def _data(seed: int = synthetic.DEFAULT_SEED) -> synthetic.SyntheticData:
    return synthetic.generate(seed=seed)


def _run(data: synthetic.SyntheticData, out_dir) -> phaseio.Run:
    """One phase run, pointed at a temp directory instead of results/."""
    return phaseio.Run(
        "synthetic",
        data.events,
        data.transactions,
        data.event_tier,
        data.truth,
        out_dir,
        synthetic.DEFAULT_SEED,
    )


@pytest.fixture(scope="module")
def data() -> synthetic.SyntheticData:
    return _data()


@pytest.fixture(scope="module")
def features(data) -> pd.DataFrame:
    return phase2.build_features(data.events, data.transactions)


# --------------------------------------------------------------------------------------
# Grouped splits (SPEC.md 8.1)
# --------------------------------------------------------------------------------------


def test_groups_are_never_split_across_a_fold():
    """The single easiest way to fool yourself here, as a test.

    Deliberately run on groups that repeat — five events with several rows each — because
    on Phase 2's own event-level frame every group has exactly one row and the property is
    true by accident. Phases 3 and 4 reuse this function on data where it is not.
    """
    groups = pd.Series(["e1"] * 4 + ["e2"] * 3 + ["e3"] * 5 + ["e4"] * 2 + ["e5"] * 6)
    folds = phase2.grouped_kfold(groups, k=3, seed=7)

    for train_idx, test_idx in folds:
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert not (train_groups & test_groups), "an event appeared in train AND test"
        assert set(train_idx) | set(test_idx) == set(range(len(groups)))
        assert not set(train_idx) & set(test_idx)


def test_every_row_is_tested_exactly_once():
    groups = pd.Series([f"e{i // 3}" for i in range(30)])
    tested = np.concatenate([test for _, test in phase2.grouped_kfold(groups, k=5, seed=1)])
    assert sorted(tested) == list(range(30))


def test_fold_assignment_is_deterministic():
    groups = pd.Series([f"e{i}" for i in range(20)])
    a = phase2.grouped_kfold(groups, k=4, seed=3)
    b = phase2.grouped_kfold(groups, k=4, seed=3)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b, strict=True))


def test_final_year_split_holds_out_the_last_year(features):
    train_idx, test_idx = phase2.final_year_split(features)
    train, test = features.iloc[train_idx], features.iloc[test_idx]
    assert test["year_index"].nunique() == 1
    assert test["year_index"].iloc[0] == features["year_index"].max()
    assert train["year_index"].max() < test["year_index"].iloc[0]
    assert len(train) + len(test) == len(features)


# --------------------------------------------------------------------------------------
# Look-ahead (SPEC.md 8.2)
# --------------------------------------------------------------------------------------


def test_lookahead_check_passes_on_the_fixture(data):
    message = phase2.assert_no_lookahead(data.events, data.transactions)
    assert message.startswith("PASS")


def test_no_outcome_column_is_in_the_feature_frame(features):
    assert not set(features.columns) & set(phase2.OUTCOME_COLUMNS) - {"tickets"}
    # `tickets` is the TARGET and is deliberately in the frame; it must not be a feature.
    assert "tickets" not in phase2.FEATURE_COLUMNS


def test_lookahead_check_catches_a_planted_leak(data, monkeypatch):
    """Plant a leak and prove the check finds it.

    `sell_through` is a pure outcome. Adding it to FEATURE_COLUMNS is exactly the accident
    SPEC.md 8.2 warns about, and a check that cannot detect it is decoration.
    """
    monkeypatch.setattr(phase2, "FEATURE_COLUMNS", [*phase2.FEATURE_COLUMNS, "sell_through"])
    with pytest.raises(AssertionError, match="outcomes, not pricing-time features"):
        phase2.assert_no_lookahead(data.events, data.transactions)


def test_features_are_unchanged_when_later_sales_are_removed(data):
    """The value-level look-ahead test, written out in the open.

    Truncate every event's transactions to the moment of its first paid sale — the decision
    moment — and rebuild. A feature that reads later sales moves; none may.
    """
    tx = data.transactions
    paid = tx[~tx["is_comp"]]
    decision = tx["event_id"].map(paid.groupby("event_id")["purchased_at"].min())
    truncated = tx[tx["purchased_at"] <= decision]

    assert len(truncated) < len(tx), "truncation removed nothing; the test proves nothing"
    full = phase2.build_features(data.events, tx)
    replayed = phase2.build_features(data.events, truncated)
    pd.testing.assert_frame_equal(full[phase2.FEATURE_COLUMNS], replayed[phase2.FEATURE_COLUMNS])


def test_lead_to_announce_is_positive_and_plausible(features):
    lead = features["lead_to_announce_days"]
    assert (lead > 0).all()
    assert lead.max() < 365


# --------------------------------------------------------------------------------------
# Features and sample
# --------------------------------------------------------------------------------------


def test_cancelled_events_are_excluded_from_the_forecast_sample(features, data):
    cancelled = set(data.events.loc[data.events["cancelled"], "event_id"])
    assert cancelled, "fixture should contain cancelled events"
    assert not set(features["event_id"]) & cancelled


def test_city_enters_through_venue_not_as_its_own_term(features):
    """Perfect collinearity gives numbers, not an error. Better not to ask the question."""
    assert features.groupby("venue")["city"].nunique().max() == 1
    assert "C(city)" not in phase2.formula(features)
    assert "C(venue)" in phase2.formula(features)


def test_formula_refuses_when_a_venue_moves_city(features):
    """If the nesting ever breaks, the model must fail loudly rather than bias the venues."""
    moved = features.copy()
    moved.loc[moved.index[0], "city"] = "City Z"
    with pytest.raises(ValueError, match="more than one city"):
        phase2.formula(moved)


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def test_scores_arithmetic():
    got = phase2.scores(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert got["MAE"] == pytest.approx(15.0)
    assert got["MAPE_pct"] == pytest.approx((0.10 + 0.10) / 2 * 100)


def test_baseline_falls_back_from_pair_to_brand_to_global(features):
    train = features.head(60)
    test = features.tail(10).copy()
    # A brand nobody has ever seen must still get a number, not NaN.
    test.loc[test.index[0], "brand"] = "Brand Z"
    test.loc[test.index[1], "venue"] = "V99"

    predicted = phase2.baseline_predict(train, test)
    assert np.isfinite(predicted).all()
    assert predicted[0] == pytest.approx(train["tickets"].mean())
    brand = test["brand"].iloc[1]
    assert predicted[1] == pytest.approx(train.loc[train["brand"] == brand, "tickets"].mean())


def test_predictions_are_on_the_ticket_scale_not_the_log_scale(features):
    model = phase2.fit(features)
    predicted = phase2.predict(model, features)
    assert predicted.min() > 10
    assert np.corrcoef(predicted, features["tickets"])[0, 1] > 0.9


def test_smearing_factor_is_at_least_one(features):
    """mean(exp(residuals)) >= exp(mean(residuals)) = 1 by Jensen. A cheap sanity check."""
    assert phase2.smearing_factor(phase2.fit(features)) >= 1.0


# --------------------------------------------------------------------------------------
# The thing the phase exists for
# --------------------------------------------------------------------------------------


def test_model_beats_the_naive_baseline_out_of_sample(features):
    folds, oof = phase2.cross_validate(features)
    pooled = folds.iloc[-1]
    assert pooled["split"] == "all folds pooled"
    assert pooled["model_MAE"] < pooled["baseline_MAE"]
    assert pooled["model_MAPE_pct"] < pooled["baseline_MAPE_pct"]
    assert len(oof) == len(features), "every event should be predicted exactly once"


def test_model_beats_the_baseline_on_the_final_year_holdout(features):
    train_idx, test_idx = phase2.final_year_split(features)
    row, preds = phase2.evaluate_split(features, train_idx, test_idx, "holdout")
    assert row["model_MAE"] < row["baseline_MAE"]
    assert len(preds) == len(test_idx)


@pytest.mark.parametrize("seed", [1, 42])
def test_the_model_beats_the_baseline_on_other_seeds(seed):
    """A property of the design, not of one lucky draw."""
    data = _data(seed)
    features = phase2.build_features(data.events, data.transactions)
    folds, _ = phase2.cross_validate(features)
    assert folds.iloc[-1]["model_MAE"] < folds.iloc[-1]["baseline_MAE"]


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def test_build_writes_a_report_and_three_figures(data, tmp_path):
    paths = phase2.build(_run(data, tmp_path))
    text = (tmp_path / "phase2_demand_forecast.md").read_text()
    assert len(list(tmp_path.glob("*.png"))) == 3
    assert len(paths) == 4
    assert "Final-year holdout" in text
    assert "SPEC.md §5.4" in text, "the report must say which features are gated on the prereg"
    assert "upper bound" in text
    # The list was FROZEN in 1c7196d (2026-08-12); the report may not still call it a draft.
    assert "draft" not in text.lower(), "the pre-registration has been FROZEN since 1c7196d"


@pytest.mark.parametrize(
    ("gate", "expected"),
    [("DRAFT", "preregistration.md is DRAFT"), ("FROZEN", "no derived tables")],
)
def test_real_data_is_refused(gate, expected, capsys, tmp_path, monkeypatch):
    """SPEC.md 5.3, both gates (see test_phase1 for the full note). Refusal on STDERR."""
    if gate == "DRAFT":
        monkeypatch.setattr(dataset, "PREREGISTRATION", fixtures.draft_preregistration(tmp_path))
    assert phase2.main(["--real"]) == 1
    captured = capsys.readouterr()
    assert "refused" in captured.err
    assert expected in captured.err
    assert "refused" not in captured.out
    assert expected not in captured.out
