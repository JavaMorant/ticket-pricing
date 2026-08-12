"""Tests for Phase 1 — descriptives.

Two kinds of test here. The ordinary kind checks the arithmetic (tickets exclude comps,
sell-through is tickets over capacity, rungs are ordered by on-sale time). The interesting
kind checks a *discipline*, in three parts, because one part was doing all the work and
could not fail:

    test_report_crosses_no_calendar_feature_with_an_outcome
        no calendar word in a table row — the guard against someone adding a "tickets by
        term week" table in six months.
    test_report_body_contains_no_inference_vocabulary_at_all
        no p-value, no correction, no significance claim, no model, anywhere in the body.
        This is the claim the report actually makes, and unlike the calendar words these
        are things a careless edit really could introduce.
    test_report_discloses_the_cross_tabs_it_does_show
        the report DOES cross outcomes with brand, venue and academic year, all of which
        are §5.4 items, so it has to say so. Claim and content, pinned together.
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import pandas as pd
import pytest

from pricing import phase1, phaseio, report, synthetic


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
def ev(data) -> pd.DataFrame:
    return phase1.event_outcomes(data.events, data.transactions)


@pytest.fixture(scope="module")
def rungs(data) -> pd.DataFrame:
    return phase1.ladder(data.event_tier)


# --------------------------------------------------------------------------------------
# academic_year — one function for the whole repo, in phaseio
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2023-09-01", "2023/24"),  # September starts the academic year
        ("2023-08-31", "2022/23"),  # one day earlier is still the old one
        ("2024-06-30", "2023/24"),  # a June ball belongs to the year that started in Sept
        ("2024-01-15", "2023/24"),
        ("2099-09-01", "2099/00"),  # the century roll-over the two old copies disagreed on
    ],
)
def test_academic_year_boundaries(date, expected):
    got = phaseio.academic_year(pd.Series([pd.Timestamp(date)]))
    assert got.iloc[0] == expected


def test_there_is_exactly_one_academic_year_function(ev):
    """Review finding, 2026-08-12: `phase1` and `phaseio` each had one, and they disagreed.

    Phase 4 used `phaseio`'s, Phases 1/2/3/5 used `phase1`'s (via `event_outcomes`), and at
    the century roll-over they returned "2099/100" and "2099/00" for the same night. One
    function now; this asserts the label Phase 1's fact table carries is that function's.
    """
    assert not hasattr(phase1, "academic_year")
    expected = phaseio.academic_year(ev["event_date"])
    assert (ev["academic_year"].to_numpy() == expected.to_numpy()).all()


# --------------------------------------------------------------------------------------
# event_outcomes — the definitions every later phase inherits
# --------------------------------------------------------------------------------------


def test_one_row_per_event(ev, data):
    assert len(ev) == len(data.events)
    assert ev["event_id"].is_unique


def test_tickets_exclude_comps(ev, data):
    """SPEC.md 8.7. A comp is not demand, and this is where that is enforced."""
    tx = data.transactions
    paid_total = int(tx.loc[~tx["is_comp"], "quantity"].sum())
    comp_total = int(tx.loc[tx["is_comp"], "quantity"].sum())

    assert ev["tickets"].sum() == paid_total
    assert ev["comps"].sum() == comp_total
    assert comp_total > 0, "fixture should contain comps, or this test proves nothing"


def test_tickets_match_event_tier_units(ev, data):
    """The two derived tables have to agree about how many tickets were sold."""
    from_panel = data.event_tier.groupby("event_id")["units_sold"].sum()
    merged = ev.set_index("event_id")["tickets"].reindex(from_panel.index)
    assert (merged == from_panel).all()


def test_sell_through_and_sellout_flag(ev):
    assert (ev["sell_through"] == ev["tickets"] / ev["capacity"]).all()
    assert (ev["sellout"] == (ev["sell_through"] >= phase1.SELLOUT_THRESHOLD)).all()
    # room_fill counts comps, sell_through does not, so it can only be larger.
    assert (ev["room_fill"] >= ev["sell_through"]).all()


def test_cancelled_events_are_kept_and_flagged(ev):
    """SPEC.md 8.4 — the pulled events are the most informative rows; never drop silently."""
    assert ev["cancelled"].sum() > 0
    assert len(phase1.live(ev)) == len(ev) - int(ev["cancelled"].sum())


def test_revenue_face_is_never_above_buyer_revenue(ev):
    """buyer price = face + fee under the current (UNKNOWN) fee treatment, so face <= buyer."""
    assert (ev["revenue_face"] <= ev["revenue_buyer"] + 1e-9).all()


# --------------------------------------------------------------------------------------
# Ladder
# --------------------------------------------------------------------------------------


def test_rungs_are_consecutive_and_time_ordered(rungs):
    for _, group in rungs.groupby("event_id"):
        assert list(group["rung"]) == list(range(1, len(group) + 1))
        assert group["window_open"].is_monotonic_increasing


def test_ladder_drops_comp_only_tiers(rungs, data):
    """A Guestlist row has no price and no paid units; it carries no ladder information."""
    assert rungs["price"].notna().all()
    assert (rungs["units_sold"] > 0).all()
    assert len(rungs) < len(data.event_tier)


def test_prices_rise_up_the_ladder_on_average(rungs):
    """Not a hypothesis test — a sanity check that `rung` is oriented the right way round."""
    medians = rungs.groupby("rung")["price"].median()
    assert medians.is_monotonic_increasing


def test_ladder_shape_columns(rungs):
    shape = phase1.ladder_shape(rungs)
    assert (shape["n_rungs"] >= 1).all()
    assert (shape["price_ratio"] > 0).all()
    assert (shape["ladder_span_days"] >= 0).all()


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("phase1")
    paths = phase1.build(_run(data, out))
    return out, paths


def test_build_writes_a_report_and_five_figures(built):
    out, paths = built
    assert (out / "phase1_descriptives.md").exists()
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert len(pngs) == 5
    assert len(paths) == 6


def test_report_declares_the_prereg_gate(built):
    out, _ = built
    text = (out / "phase1_descriptives.md").read_text()
    assert "No model is fitted here" in text
    assert "preregistration.md" in text
    assert "SYNTHETIC DATA" in text


def test_the_cli_records_the_seed_it_actually_ran_on(tmp_path, monkeypatch):
    """Review finding, 2026-08-12: Phases 1-3 committed reports with no seed at all.

    `--seed` defaulted to None, the loader silently fell back to `synthetic.DEFAULT_SEED`
    and the header suppressed the line because None is falsy — so the committed report
    said only ``Source: `synthetic` `` while Phase 4's said `seed 20260811`. "Reports
    regenerate every number" (DECISIONS.md) is not checkable without it. Run through the
    CLI, because the default is the half of the defect a fixture-built report cannot see.
    """
    monkeypatch.setattr(phaseio, "SYNTHETIC_RESULTS_DIR", tmp_path)
    assert phase1.main(["--synthetic"]) == 0
    text = (tmp_path / "phase1_descriptives.md").read_text()
    assert f"seed `{synthetic.DEFAULT_SEED}`" in text
    assert (
        f"Regenerate: `uv run python -m pricing.phase1 --synthetic --seed {synthetic.DEFAULT_SEED}`"
        in text
    )


def test_report_crosses_no_calendar_feature_with_an_outcome(built):
    """The discipline, as a test — the CALENDAR half of it.

    Phase 1 may count events by academic year and by day of week (composition). It may NOT
    cross a calendar feature with an outcome — that needs a frozen preregistration.md, an
    FDR correction and an out-of-sample confirmation (SPEC.md 5.3-5.5). If a future edit
    adds a "tickets in freshers' week" table, this fails and the person adding it has to
    think.

    The words are matched only in table rows, i.e. where a result would arrive. The
    preamble names them all while explaining why they are absent.
    """
    out, _ = built
    text = (out / "phase1_descriptives.md").read_text().lower()
    # Table rows all start with a pipe; prose does not. A hypothesis would arrive as a table.
    table_rows = "\n".join(line for line in text.splitlines() if line.startswith("|"))
    for pattern in ["fresher", "exam", "reading week", "loan", "revision", "ball season"]:
        assert pattern not in table_rows, f"Phase 1 must not report a {pattern!r} result"


# Vocabulary that only appears when something has been TESTED rather than described. The
# preamble is where the report says these are absent, so the body is what gets scanned.
INFERENCE_WORDS = [
    "p-value",
    "p =",
    "p <",
    "p<",
    "significan",
    "fdr",
    "benjamini",
    "confidence interval",
    "null hypothesis",
    "correlat",
    "regression",
    "coefficient",
]


def test_report_body_contains_no_inference_vocabulary_at_all(built):
    """The other half of the discipline, and the half the old test could not fail.

    Review finding, 2026-08-12: the previous guard grepped table rows for calendar words
    that can never appear, because Phase 1 builds no calendar features — so it pinned
    nothing. Meanwhile the report DOES cross outcomes with brand, venue and academic year,
    all of which are on the §5.4 list, so the claim it was guarding was too strong anyway.

    The accurate claim is that nothing is *tested*: no p-value, no correction, no
    significance claim, no model. That is what this asserts, over the body of the report —
    everything after the preamble in which those words are used to say they are absent.
    """
    out, _ = built
    text = (out / "phase1_descriptives.md").read_text()
    assert "\n---\n" in text, "the preamble must stay separated from the body by a rule"
    body = text.split("\n---\n", 1)[1].lower()
    for word in INFERENCE_WORDS:
        assert word not in body, f"Phase 1 reported something that reads as a test: {word!r}"


def test_report_discloses_the_cross_tabs_it_does_show(built):
    """The claim and the content have to stay in step.

    The report shows median tickets by venue, median sell-through by brand and sellout rate
    by academic year — all §5.4 items. Saying "no §5.4 pattern appears here" would be
    false, so the preamble says which ones do appear and why SPEC.md §8.9 requires them.
    If someone tightens the wording back to the overreaching version, this fails.
    """
    out, _ = built
    text = (out / "phase1_descriptives.md").read_text()
    preamble = text.split("\n---\n", 1)[0]
    assert "no hypothesis is" in preamble and "tested" in preamble
    assert "cross-tabs by academic year, brand and venue" in preamble
    assert "§8.9" in preamble

    # ...and those cross-tabs really are in the report, so the disclosure is not theatre.
    body = text.split("\n---\n", 1)[1]
    assert "median_sell_through" in body and "median_tickets" in body
    assert "sellout_rate" in body


def test_synthetic_output_goes_to_the_committable_directory():
    """Privacy: synthetic results are committable, real results are not (DECISIONS.md)."""
    args = argparse.Namespace(real=False, synthetic=True, seed=synthetic.DEFAULT_SEED)
    assert phaseio.resolve(args).out_dir == phaseio.SYNTHETIC_RESULTS_DIR
    assert phaseio.SYNTHETIC_RESULTS_DIR.is_relative_to(phaseio.REAL_RESULTS_DIR)
    assert phaseio.SYNTHETIC_RESULTS_DIR != phaseio.REAL_RESULTS_DIR


def test_real_data_is_refused_while_the_preregistration_is_draft(capsys):
    """SPEC.md 5.3. The whole real path is wired and closed; --real must exit non-zero.

    The reason goes to STDERR, in every phase, so that `2>/dev/null` cannot hide half the
    refusals (review finding, 2026-08-12).
    """
    assert phase1.main(["--real"]) == 1
    captured = capsys.readouterr()
    assert "refused" in captured.err
    assert "preregistration.md" in captured.err
    assert "refused" not in captured.out


# --------------------------------------------------------------------------------------
# report.py plumbing
# --------------------------------------------------------------------------------------


def test_md_table_escapes_pipes_and_formats_types():
    frame = pd.DataFrame([{"a": "x | y", "b": 1234, "c": 1.5, "d": True, "e": None, "f": pd.NaT}])
    rendered = report.md_table(frame)
    row = rendered.splitlines()[2]
    assert row.count("|") == len(frame.columns) + 1, "a pipe inside a cell would add columns"
    assert "1,234" in row
    assert "1.50" in row
    assert "yes" in row
