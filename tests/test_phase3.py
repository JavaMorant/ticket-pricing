"""Tests for Phase 3 — sales curve.

The invariants worth having:

    curves are monotone and end at 1.0            it is a cumulative share; anything else
                                                  is an arithmetic bug wearing a plot
    comps and cancelled events are out            SPEC.md 8.7, 8.4
    forecasts are held out                        SPEC.md 8.1 — the curve must not have
                                                  been drawn using the event it forecasts
    error falls as the event approaches           if it did not, the curve is doing nothing
    the curve beats "assume no more sales"        the floor any method must clear
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

import tests.synthetic as fixtures
from pricing import dataset, phase1, phase3, phaseio, synthetic


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
def curves(data):
    return phase3.cumulative_sales(data.events, data.transactions)


@pytest.fixture(scope="module")
def ev(data) -> pd.DataFrame:
    return phase1.live(phase1.event_outcomes(data.events, data.transactions))


# --------------------------------------------------------------------------------------
# The curves themselves
# --------------------------------------------------------------------------------------


def test_columns_run_from_max_lead_down_to_the_door(curves):
    cum, share, _ = curves
    assert list(cum.columns) == list(range(phase3.MAX_LEAD, -1, -1))
    assert list(share.columns) == list(cum.columns)


def test_cumulative_share_is_monotone_and_ends_at_one(curves):
    """A cumulative share can only go up, and every event ends having sold 100% of itself."""
    _, share, _ = curves
    diffs = share.diff(axis=1).iloc[:, 1:]
    assert (diffs >= -1e-12).all().all(), "cumulative share went down"
    assert np.allclose(share.iloc[:, -1], 1.0)
    assert (share >= 0).all().all()


def test_final_totals_match_the_phase_1_ticket_counts(curves, ev):
    _, _, final = curves
    expected = ev.set_index("event_id")["tickets"].reindex(final.index)
    assert (final == expected).all()


def test_comps_are_excluded_from_the_curve(curves, data):
    """SPEC.md 8.7 — a guestlist spot issued three days out is not late demand."""
    _, _, final = curves
    paid_only = data.transactions[~data.transactions["is_comp"]]
    assert final.sum() == paid_only[paid_only["event_id"].isin(final.index)]["quantity"].sum()


def test_cancelled_events_are_excluded_from_the_curve(curves, data):
    """SPEC.md 8.4 — a pulled event's final total is not a final total."""
    _, share, _ = curves
    cancelled = set(data.events.loc[data.events["cancelled"], "event_id"])
    assert cancelled
    assert not set(share.index) & cancelled


def test_early_sales_are_bucketed_not_dropped(curves, data):
    """Tickets sold earlier than MAX_LEAD land in the first column, so curves still hit 1.0."""
    tx = data.transactions
    assert (tx["lead_time_days"] > phase3.MAX_LEAD).any(), "fixture has no very early sales"
    cum, _, final = curves
    assert (cum[phase3.MAX_LEAD] > 0).any()
    assert (cum.iloc[:, -1] == final).all()


# --------------------------------------------------------------------------------------
# Segments and median curves
# --------------------------------------------------------------------------------------


def test_segment_labels_contain_no_pipe(ev):
    labels = phase3.segment_labels(ev, ["brand", "sellout"])
    assert not labels.str.contains(r"\|").any()
    assert labels.str.contains("sellout").any()


def test_median_curve_is_itself_a_valid_cumulative_curve(curves, ev):
    """The pointwise median of monotone curves is monotone — free, and worth asserting."""
    _, share, _ = curves
    table, _ = phase3.median_curves(share, phase3.segment_labels(ev, ["brand"]))
    assert (table.diff(axis=1).iloc[:, 1:] >= -1e-12).all().all()
    assert np.allclose(table.iloc[:, -1], 1.0)
    assert "__pooled__" in table.index


def test_thin_segments_fall_back_to_the_pooled_curve(curves, ev):
    _, share, _ = curves
    segments = phase3.segment_labels(ev, ["brand", "sellout"])
    table, fell_back = phase3.median_curves(share, segments)

    sizes = segments.value_counts()
    thin = [name for name, n in sizes.items() if n < phase3.MIN_EVENTS_PER_SEGMENT]
    assert thin, "fixture should contain a thin sellout segment"
    assert len(fell_back) == len(thin)
    for name in thin:
        pd.testing.assert_series_equal(table.loc[name], table.loc["__pooled__"], check_names=False)


# --------------------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------------------


def test_forecast_is_never_below_what_has_already_sold(curves, ev):
    cum, share, _ = curves
    segments = phase3.segment_labels(ev, ["brand"])
    table, _ = phase3.median_curves(share, segments)
    for day in phase3.FORECAST_DAYS:
        forecast = phase3.forecast_final(cum, table, segments, day)
        assert (forecast >= cum[day] - 1e-9).all()


def test_forecast_is_sales_divided_by_the_segment_share():
    """The whole method, on three lines of made-up data."""
    cum = pd.DataFrame({7: [50.0, 90.0]}, index=["e1", "e2"])
    table = pd.DataFrame({7: [0.5]}, index=["Brand A"])
    segments = pd.Series(["Brand A", "Brand A"], index=["e1", "e2"], name="segment")
    got = phase3.forecast_final(cum, table, segments, 7)
    assert got.tolist() == [100.0, 180.0]


def test_a_tiny_share_is_floored_so_the_forecast_cannot_explode():
    cum = pd.DataFrame({7: [10.0]}, index=["e1"])
    table = pd.DataFrame({7: [0.001]}, index=["Brand A"])
    segments = pd.Series(["Brand A"], index=["e1"], name="segment")
    got = phase3.forecast_final(cum, table, segments, 7)
    assert got.iloc[0] == pytest.approx(10.0 / phase3.MIN_SHARE)


def test_an_unknown_segment_falls_back_to_the_pooled_curve():
    cum = pd.DataFrame({7: [50.0]}, index=["e1"])
    table = pd.DataFrame({7: [0.5, 0.25]}, index=["__pooled__", "Brand A"])
    segments = pd.Series(["Brand Z"], index=["e1"], name="segment")
    assert phase3.forecast_final(cum, table, segments, 7).iloc[0] == pytest.approx(100.0)


# --------------------------------------------------------------------------------------
# Held-out evaluation
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluation(curves, ev):
    cum, share, final = curves
    return phase3.evaluate(cum, share, final, phase3.segment_labels(ev, ["brand"]))


def test_every_event_is_forecast_once_per_decision_day(evaluation, curves):
    _, held = evaluation
    _, share, _ = curves
    for day in phase3.FORECAST_DAYS:
        subset = held[held["day"] == day]
        assert len(subset) == len(share)
        assert subset["event_id"].is_unique


def test_error_falls_as_the_event_approaches(evaluation):
    metrics, _ = evaluation
    mapes = metrics.set_index("day")["segment_MAPE_pct"]
    assert mapes["-7"] > mapes["-3"] > mapes["-1"]


def test_curve_beats_assuming_no_further_sales(evaluation):
    """The floor. If the curve cannot clear this, it is not doing anything."""
    metrics, _ = evaluation
    assert (metrics["segment_MAPE_pct"] < metrics["nosale_MAPE_pct"]).all()
    assert (metrics["segment_MAE"] < metrics["nosale_MAE"]).all()


def test_median_share_sold_rises_towards_the_event(evaluation):
    metrics, _ = evaluation
    assert metrics["median_share_sold"].is_monotonic_increasing
    assert metrics["median_share_sold"].between(0, 1).all()


def test_forecast_curves_are_built_without_the_event_being_forecast(curves, ev, monkeypatch):
    """SPEC.md 8.1 in its sales-curve costume.

    Replace grouped_kfold with a single 50/50 split and check that the median curve used
    for the test half is the one built from the train half — i.e. the evaluation really
    does refit the curve per fold rather than reusing a curve drawn on everything.
    """
    cum, share, final = curves
    n = len(share)
    half = n // 2
    monkeypatch.setattr(
        phase3, "grouped_kfold", lambda groups, **kw: [(np.arange(half), np.arange(half, n))]
    )
    segments = phase3.segment_labels(ev, ["brand"])
    _, held = phase3.evaluate(cum, share, final, segments)

    train_ids = share.index[:half]
    expected_curves, _ = phase3.median_curves(share.loc[train_ids], segments.reindex(train_ids))
    test_ids = share.index[half:]
    expected = phase3.forecast_final(cum.loc[test_ids], expected_curves, segments, 7)
    got = held[held["day"] == 7].set_index("event_id")["forecast"]
    assert np.allclose(got.reindex(test_ids), expected.reindex(test_ids))


@pytest.mark.parametrize("seed", [1, 42])
def test_error_falls_towards_the_event_on_other_seeds(seed):
    data = _data(seed)
    ev = phase1.live(phase1.event_outcomes(data.events, data.transactions))
    cum, share, final = phase3.cumulative_sales(data.events, data.transactions)
    metrics, _ = phase3.evaluate(cum, share, final, phase3.segment_labels(ev, ["brand"]))
    mapes = metrics.set_index("day")["segment_MAPE_pct"]
    assert mapes["-7"] > mapes["-1"]
    assert (metrics["segment_MAPE_pct"] < metrics["nosale_MAPE_pct"]).all()


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def test_build_writes_a_report_and_three_figures(data, tmp_path):
    paths = phase3.build(_run(data, tmp_path))
    text = (tmp_path / "phase3_sales_curve.md").read_text()
    assert len(list(tmp_path.glob("*.png"))) == 3
    assert len(paths) == 4
    assert "must never be used to forecast" in text, "the sellout-segmentation trap must be stated"
    assert "Bass" in text, "the report should say why the method is not Bass diffusion"


@pytest.mark.parametrize(
    ("gate", "expected"),
    [("DRAFT", "preregistration.md is DRAFT"), ("FROZEN", "no derived tables")],
)
def test_real_data_is_refused(gate, expected, capsys, tmp_path, monkeypatch):
    """SPEC.md 5.3, both gates (see test_phase1 for the full note). Refusal on STDERR."""
    if gate == "DRAFT":
        monkeypatch.setattr(dataset, "PREREGISTRATION", fixtures.draft_preregistration(tmp_path))
    assert phase3.main(["--real"]) == 1
    captured = capsys.readouterr()
    assert "refused" in captured.err
    assert expected in captured.err
    assert "refused" not in captured.out
    assert expected not in captured.out
