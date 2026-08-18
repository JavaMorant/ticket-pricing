"""Tests for the synthetic ground-truth generator.

Four of these are not ordinary unit tests. They are the ACCEPTANCE TESTS for Phase 4:

    test_naive_pooled_regression_has_the_wrong_sign   SPEC.md 4.1
    test_within_event_regression_recovers_beta_true   SPEC.md 4.2
    test_lead_time_confound_biases_plain_event_fe     SPEC.md 4.3
    test_switching_the_confound_off_makes_plain_fe_work

When the Phase 4 estimator exists it has to reproduce all four on this fixture before it
is pointed at real data. An estimator that cannot tell a wrong-signed regression from a
right one on data whose answer we chose has no business producing a CV bullet.

Runtime: generate() takes about a second, so it is cached per seed.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

import tests.synthetic as fixtures
from pricing import dataset, normalize, phaseio, synthetic, tables

# Three seeds, so the acceptance properties are shown to be properties of the DESIGN and
# not a lucky draw. Kept to three because each one costs ~1s to generate.
SEEDS = [1, 42, synthetic.DEFAULT_SEED]

# Tolerance on beta recovery. The identifying variation is `price_jitter_sd` — the slice
# of price movement that is not lead time — so the estimate has real sampling error at
# ~400 panel rows. Measured spread across eight seeds was under 0.08; 0.15 leaves room
# without letting a genuinely broken estimator through.
BETA_TOLERANCE = 0.15


@lru_cache(maxsize=8)
def _generate(seed: int, params_items: tuple = ()) -> synthetic.SyntheticData:
    return synthetic.generate(seed=seed, params=dict(params_items) or None)


@lru_cache(maxsize=8)
def _panel(seed: int, params_items: tuple = ()) -> pd.DataFrame:
    data = _generate(seed, params_items)
    return synthetic.analysis_panel(data.event_tier, data.events)


@pytest.fixture(scope="module")
def data() -> synthetic.SyntheticData:
    return _generate(synthetic.DEFAULT_SEED)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel(synthetic.DEFAULT_SEED)


# --------------------------------------------------------------------------------------
# 1. Schema equality with tables.py — the fixture must be the same shape as real output
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_tables() -> dict[str, pd.DataFrame]:
    """The three tables built by tables.py from the hand-written Phase 0 fixture.

    This is the reference schema. It is built by the same functions the real ingest
    calls, so if the generator's output matches it column for column and dtype for dtype,
    the generator is producing exactly what the live pipeline will produce.
    """
    events = tables.build_events(fixtures.fake_costs())
    transactions = tables.build_transactions(fixtures.fake_transactions(), events)
    return {
        "events": events,
        "transactions": transactions,
        "event_tier": tables.build_event_tier(transactions),
    }


@pytest.mark.parametrize("table", ["events", "transactions", "event_tier"])
def test_columns_match_tables_py_output_exactly(data, reference_tables, table):
    assert list(getattr(data, table).columns) == list(reference_tables[table].columns)


@pytest.mark.parametrize("table", ["events", "transactions", "event_tier"])
def test_dtypes_match_tables_py_output_exactly(data, reference_tables, table):
    generated, reference = getattr(data, table), reference_tables[table]
    mismatched = {
        col: (str(generated[col].dtype), str(reference[col].dtype))
        for col in reference.columns
        if str(generated[col].dtype) != str(reference[col].dtype)
    }
    assert mismatched == {}


@pytest.mark.parametrize("table", ["events", "transactions", "event_tier"])
def test_every_column_is_on_the_pii_allowlist(data, table):
    """write_table would refuse anything else, so a new column must be a deliberate act."""
    normalize.assert_no_pii(getattr(data, table), where=table)


def test_no_buyer_pii_columns_anywhere(data):
    for table in (data.events, data.transactions, data.event_tier):
        assert normalize.pii_columns(table) == []


def test_everything_is_obviously_fake(data):
    """Nothing committable may look like a real event, brand, venue or order."""
    assert data.events["event_id"].str.startswith("FAKE-EV-").all()
    assert data.events["event_name"].str.startswith("FAKE ").all()
    assert data.transactions["order_id"].str.startswith("FAKE-ORD-").all()
    assert set(data.events["brand"]) == {"Brand A", "Brand B"}
    assert set(data.events["venue"]) <= {"V1", "V2", "V3", "V4"}
    assert set(data.events["city"]) <= {"City A", "City B"}


# --------------------------------------------------------------------------------------
# 2. Determinism
# --------------------------------------------------------------------------------------


def test_same_seed_gives_identical_tables():
    a = synthetic.generate(seed=7)
    b = synthetic.generate(seed=7)
    for table in ("events", "transactions", "event_tier"):
        pd.testing.assert_frame_equal(getattr(a, table), getattr(b, table))


def test_same_seed_gives_identical_ground_truth():
    a, b = synthetic.generate(seed=7).truth, synthetic.generate(seed=7).truth
    pd.testing.assert_frame_equal(a["event_truth"], b["event_truth"])
    pd.testing.assert_frame_equal(a["tier_truth"], b["tier_truth"])
    assert a["params"] == b["params"]
    assert a["beta_true"] == b["beta_true"]


def test_different_seeds_give_different_data():
    a = synthetic.generate(seed=7)
    b = synthetic.generate(seed=8)
    assert not a.event_tier["price"].equals(b.event_tier["price"])
    assert not a.truth["event_truth"]["demand_shock"].equals(b.truth["event_truth"]["demand_shock"])


# --------------------------------------------------------------------------------------
# 3. ACCEPTANCE TEST — the naive pooled regression comes out wrong-signed (SPEC.md 4.1)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_naive_pooled_regression_has_the_wrong_sign(seed):
    """log Q ~ log P, pooled: higher prices "cause" more sales.

    This is the finding SPEC.md 4.1 says to expect and to write down rather than fix.
    It happens because the operator's price loads on the demand shock (lambda > 0) and
    the pooled regression omits that shock, so price carries its own omitted variable.
    """
    beta = synthetic.naive_pooled_beta(_panel(seed))
    assert beta > 0, f"seed {seed}: pooled beta {beta:+.3f} is not wrong-signed"


@pytest.mark.parametrize("seed", SEEDS)
def test_pooled_and_within_event_estimates_disagree_about_the_sign(seed):
    """The whole Phase 4 story in one assertion: same data, one term apart, opposite sign."""
    pooled = synthetic.naive_pooled_beta(_panel(seed))
    within = synthetic.within_event_beta(_panel(seed))
    assert pooled > 0 > within


# --------------------------------------------------------------------------------------
# 4. ACCEPTANCE TEST — within-event variation recovers BETA_TRUE (SPEC.md 4.2)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_within_event_regression_recovers_beta_true(seed):
    """Event fixed effects + the lead-time control land on the elasticity we planted."""
    beta_true = synthetic.PARAMS["beta_true"]
    beta = synthetic.within_event_beta(_panel(seed), control_lead_time=True)
    assert abs(beta - beta_true) < BETA_TOLERANCE, (
        f"seed {seed}: recovered {beta:+.3f}, expected {beta_true:+.3f} +/- {BETA_TOLERANCE}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_lead_time_confound_biases_plain_event_fe(seed):
    """SPEC.md 4.3: Early Bird is cheap AND early, so FE alone overstates elasticity.

    With the confound on, the plain event-FE estimate is materially MORE negative than
    the truth — it is charging the timing effect to the price. This test exists so nobody
    can claim event fixed effects alone are enough.
    """
    beta_true = synthetic.PARAMS["beta_true"]
    plain = synthetic.within_event_beta(_panel(seed), control_lead_time=False)
    assert plain < beta_true - 2 * BETA_TOLERANCE, (
        f"seed {seed}: plain FE {plain:+.3f} is not visibly biased away from "
        f"{beta_true:+.3f}; the confound is not doing its job"
    )


def test_switching_the_confound_off_makes_plain_fe_work():
    """The confound is a tunable knob, not a fact of life. Set it to 0 and FE is enough."""
    off = (("lead_confound", 0.0),)
    beta = synthetic.within_event_beta(_panel(SEEDS[0], off), control_lead_time=False)
    assert abs(beta - synthetic.PARAMS["beta_true"]) < BETA_TOLERANCE


# --------------------------------------------------------------------------------------
# 5. The ground-truth dict
# --------------------------------------------------------------------------------------


def test_truth_dict_shape(data):
    truth = data.truth
    assert set(truth) == {
        "seed",
        "beta_true",
        "params",
        "event_truth",
        "tier_truth",
        "instruments",
        "expected",
    }
    assert truth["seed"] == synthetic.DEFAULT_SEED
    assert truth["beta_true"] == -0.80
    assert truth["params"]["beta_true"] == truth["beta_true"]
    assert set(truth["instruments"]) == {"valid", "invalid"}


def test_event_truth_is_one_row_per_event_and_carries_the_hidden_variables(data):
    event_truth = data.truth["event_truth"]
    assert len(event_truth) == len(data.events)
    assert event_truth["event_key"].is_unique
    for column in ("demand_shock", "log_hire_index", "log_base_price", "artist_tier"):
        assert column in event_truth.columns
        # The whole point: these never reach a derived table.
        assert column not in data.events.columns


def test_tier_truth_is_one_row_per_ladder_rung(data):
    tier_truth = data.truth["tier_truth"]
    ladder_rows = data.event_tier["tier_ordinal"].notna().sum()  # comp tiers have no ordinal
    assert len(tier_truth) == ladder_rows
    assert (tier_truth["units"] >= 3).all()


def test_units_sold_matches_the_planted_truth(data):
    """The panel's units_sold is exactly the quantity the generator drew — no leakage."""
    planted = data.truth["tier_truth"].rename(columns={"event_key": "event_id"})
    merged = data.event_tier.merge(planted[["event_id", "tier_name", "units"]], how="inner")
    assert len(merged) == len(planted)
    assert (merged["units_sold"] == merged["units"]).all()


# --------------------------------------------------------------------------------------
# 6. The programme looks like the real one (SPEC.md 3, DECISIONS.md data section)
# --------------------------------------------------------------------------------------


def test_programme_shape(data):
    events = data.events
    assert len(events) == 120
    assert events["event_date"].dt.year.nunique() >= 5
    span_days = (events["event_date"].max() - events["event_date"].min()).days
    assert 4 * 365 < span_days < 6 * 365
    assert set(events["capacity"]) <= {150, 400, 800, 1500}
    assert events["capacity"].min() >= 150
    assert events["capacity"].max() <= 1500


def test_five_academic_years_each_with_events(data):
    years = data.truth["event_truth"]["academic_year"]
    assert years.nunique() == 5
    assert years.value_counts().min() >= 15


def test_tier_ladders_are_two_to_five_rungs(data):
    ladder = data.event_tier[data.event_tier["tier_ordinal"].notna()]
    sizes = ladder.groupby("event_id").size()
    assert sizes.min() >= 2
    assert sizes.max() <= 5


def test_ladder_runs_from_early_bird_to_door(data):
    """Ordinal 0 is an opening release; 99 is the door (normalize.DOOR_ORDINAL)."""
    ordinals = data.event_tier["tier_ordinal"].dropna()
    assert (ordinals == 0).sum() > 0
    assert (ordinals == normalize.DOOR_ORDINAL).sum() > 0
    opening = data.event_tier.loc[data.event_tier["tier_ordinal"] == 0, "tier_name"]
    assert set(opening) <= {"Early Bird", "First Release"}


def test_prices_are_in_a_believable_student_night_range(data):
    price = data.event_tier["price"].dropna()
    assert price.min() >= 4.0
    assert price.max() <= 25.0


def test_price_rises_up_the_ladder_within_an_event(data):
    """Not guaranteed rung by rung (there is pricing jitter), but true on average."""
    ladder = data.event_tier[data.event_tier["tier_ordinal"].notna()]
    first = ladder[ladder["tier_ordinal"] == 0]["price"].mean()
    door = ladder[ladder["tier_ordinal"] == normalize.DOOR_ORDINAL]["price"].mean()
    assert door > first


def test_door_tickets_are_sold_on_the_night(data):
    door = data.event_tier[data.event_tier["tier_ordinal"] == normalize.DOOR_ORDINAL]
    assert (door["lead_time_open_days"] == 0).all()
    assert (door["window_open"].dt.hour >= 20).all()
    # Never after midnight: tables._lead_time_days floors to the DATE, so a 00:30 sale
    # would be recorded as the day after the event.
    assert (door["window_close"].dt.date == door["event_date"].dt.date).all()


def test_comps_are_flagged_and_excluded_from_demand(data):
    comps = data.transactions[data.transactions["is_comp"]]
    assert len(comps) > 0
    assert (comps["price_paid"] == 0).all()
    assert set(comps["tier_name"]) <= set(synthetic.COMP_NAMES)
    # A comp-only tier row exists in the panel but sold nothing (SPEC.md 8.7).
    comp_tiers = data.event_tier[data.event_tier["units_comp"] > 0]
    assert (comp_tiers["units_sold"] == 0).all()
    assert data.event_tier["units_comp"].sum() == len(comps)


def test_some_events_were_cancelled(data):
    assert data.events["cancelled"].sum() == synthetic.PARAMS["n_cancelled"]
    cancelled = set(data.events.loc[data.events["cancelled"], "event_id"])
    # They are in the data with the thin presale that got them pulled (SPEC.md 8.4).
    sold = data.event_tier.groupby("event_id")["units_sold"].sum()
    assert sold[list(cancelled)].max() < sold.median()


def test_more_than_one_platform_is_present(data):
    assert set(data.transactions["platform"]) == {"fixr", "tbc", "youni"}
    # FEE_TREATMENT is UNKNOWN for every platform until a real order confirmation settles
    # it, and that provisional-ness must travel on the rows (SPEC.md 8.6).
    assert set(data.transactions["fee_treatment"]) == {"UNKNOWN"}


def test_sales_never_exceed_capacity(data):
    sold = data.event_tier.groupby("event_id")[["units_sold"]].sum()
    capacity = data.events.set_index("event_id")["capacity"]
    assert (sold["units_sold"] <= capacity.loc[sold.index]).all()


# --------------------------------------------------------------------------------------
# 7. The purchase-time process is S-shaped (SPEC.md 6.2)
# --------------------------------------------------------------------------------------


def test_cumulative_purchase_curve_is_s_shaped(data):
    """An S-shaped cumulative means a unimodal density: middle-heavy, thin at both ends.

    Measured as the share of a tier's tickets bought in each third of its window. A flat
    (uniform) arrival process would put a third in each. An S-curve puts the most in the
    middle third and the least in the last.
    """
    tx = data.transactions.merge(
        data.event_tier[["event_id", "tier_name", "window_open", "window_close"]],
        on=["event_id", "tier_name"],
        how="inner",
    )
    tx = tx[~tx["is_comp"]]
    span = (tx["window_close"] - tx["window_open"]).dt.total_seconds()
    tx = tx[span > 0]
    frac = (tx["purchased_at"] - tx["window_open"]).dt.total_seconds() / span[span > 0]

    thirds = [
        float((frac < 1 / 3).mean()),
        float(((frac >= 1 / 3) & (frac < 2 / 3)).mean()),
        float((frac >= 2 / 3).mean()),
    ]
    assert thirds[1] == max(thirds), f"not unimodal: {thirds}"
    assert thirds[1] > 0.40, f"middle third {thirds[1]:.2f} is barely above uniform"
    assert float(frac.std()) < 0.2887, "arrivals are as spread out as a uniform process"


def test_every_tier_opens_with_a_sale_at_its_on_sale_moment(data):
    """window_open is the FIRST SALE, and the panel's lead time is measured from it.

    Pinning the first ticket to the on-sale instant is what makes lead_time_open_days the
    scheduled lead rather than a noisy proxy for it — noise there would leak straight into
    the lead-time control that identifies beta.
    """
    planted = data.truth["tier_truth"].rename(columns={"event_key": "event_id"})
    merged = data.event_tier.merge(
        planted[["event_id", "tier_name", "lead_open_days"]], how="inner"
    )
    assert (merged["lead_time_open_days"] == merged["lead_open_days"]).all()


# --------------------------------------------------------------------------------------
# 8. The planted traps: growth trend, cost shifter, failed instrument
# --------------------------------------------------------------------------------------


def test_brand_growth_is_planted_at_different_rates_for_the_two_brands(data):
    """SPEC.md 8.9: the brands got bigger, at different speeds, over the five years."""
    truth = data.truth["event_truth"]
    slopes = {
        brand: np.polyfit(sub["year_index"], sub["brand_growth"], 1)[0]
        for brand, sub in truth.groupby("brand")
    }
    assert slopes["Brand A"] > 0
    assert slopes["Brand B"] > 0
    assert slopes["Brand A"] != slopes["Brand B"]


def test_brand_growth_trend_is_visible_in_the_realised_sales(data):
    """The trap, in the data rather than in the parameters.

    Note what this test needs in order to see the trend, because it is the SPEC.md 8.9
    lesson in miniature: sell-through, not raw sales (capacity varies), and controls for
    artist tier, calendar position, venue size AND the venue hire index. Hire rates drift
    up year on year, which pushes prices up and sell-through down — a real force pulling
    the other way. Regress raw sales on the year alone and the growth trend can vanish
    into the venue mix. Anything you then read off the calendar is inheriting whichever
    of those two won.
    """
    truth = data.truth["event_truth"]
    sold = data.event_tier.groupby("event_id")["units_sold"].sum().rename("sold")
    joined = truth.merge(sold, left_on="event_key", right_index=True)
    joined = joined[~joined["cancelled"]].copy()
    joined["log_sell_through"] = np.log(joined["sold"] / joined["capacity"])
    joined["log_capacity"] = np.log(joined["capacity"])

    formula = (
        "log_sell_through ~ year_index + artist_tier + calendar_shock "
        "+ log_capacity + log_hire_index"
    )
    for brand, sub in joined.groupby("brand"):
        slope = smf.ols(formula, data=sub).fit().params["year_index"]
        assert slope > 0, f"{brand}: sell-through trend {slope:+.4f} is not growing"


def test_cost_shifter_moves_price_but_not_demand(data):
    """SPEC.md 4.5: the venue hire index is a candidate instrument.

    Relevance: it correlates with the event's price level. Exclusion: it is drawn
    independently of the demand shock, so its correlation with demand is noise.
    """
    truth = data.truth["event_truth"]
    relevance = truth["log_hire_index"].corr(truth["log_base_price"])
    exclusion = truth["log_hire_index"].corr(truth["demand_shock"])
    assert relevance > 0.20, f"instrument is weak: corr with price {relevance:.2f}"
    assert abs(exclusion) < 0.20, f"instrument leaks into demand: corr {exclusion:.2f}"


def test_artist_guarantee_fails_the_exclusion_restriction(data):
    """The trap instrument. A bigger guarantee means a bigger artist means more demand."""
    truth = data.truth["event_truth"]
    assert np.log(truth["artist_guarantee"]).corr(truth["demand_shock"]) > 0.30


# --------------------------------------------------------------------------------------
# 9. The analysis panel's filters
# --------------------------------------------------------------------------------------


def test_analysis_panel_drops_cancelled_comp_and_single_tier_rows(data, panel):
    cancelled = set(data.events.loc[data.events["cancelled"], "event_id"])
    assert not set(panel["event_id"]) & cancelled
    assert (panel["units_sold"] > 0).all()
    assert (panel["price"] > 0).all()
    assert panel["log_q"].notna().all()
    assert panel.groupby("event_id").size().min() >= 2
    assert len(panel) < len(data.event_tier)


def test_analysis_panel_keeps_cancelled_events_when_no_events_table_is_passed(data):
    """The cancelled filter needs the events table — event_tier does not carry the flag."""
    everything = synthetic.analysis_panel(data.event_tier)
    with_filter = synthetic.analysis_panel(data.event_tier, data.events)
    assert len(everything) > len(with_filter)


# --------------------------------------------------------------------------------------
# 10. Writing fixtures
# --------------------------------------------------------------------------------------


def test_write_fixtures_round_trips(tmp_path, data):
    paths = synthetic.write_fixtures(data, out_dir=tmp_path)
    assert {p.name for p in paths} == {
        "events.parquet",
        "transactions.parquet",
        "event_tier.parquet",
        "ground_truth.json",
    }
    reloaded = pd.read_parquet(tmp_path / "event_tier.parquet")
    pd.testing.assert_frame_equal(reloaded, data.event_tier)


def test_write_fixtures_json_holds_the_scalar_truth(tmp_path, data):
    synthetic.write_fixtures(data, out_dir=tmp_path)
    truth = json.loads((tmp_path / "ground_truth.json").read_text())
    assert truth["beta_true"] == -0.80
    assert truth["params"]["lead_confound"] == synthetic.PARAMS["lead_confound"]
    assert "event_truth" not in truth  # DataFrames are regenerable, not written


# --------------------------------------------------------------------------------------
# 11. The real-data guard (dataset.py)
# --------------------------------------------------------------------------------------


def test_real_data_is_refused_while_the_preregistration_is_a_draft(tmp_path, monkeypatch):
    """The DRAFT arm of the guard, on an INJECTED draft — never the repo's own file.

    Until 2026-08-18 this test sniffed the ambient `preregistration.md`, so it stopped
    testing the guard the moment commit `1c7196d` froze that file. The guard was fine; the
    test was measuring the repo, not the code.
    """
    monkeypatch.setattr(dataset, "PREREGISTRATION", fixtures.draft_preregistration(tmp_path))
    assert dataset.preregistration_status() == "DRAFT"
    with pytest.raises(RuntimeError, match="OFF LIMITS"):
        dataset.load_real()


def test_the_refusal_says_why_and_what_to_do(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        dataset.require_frozen_preregistration(fixtures.draft_preregistration(tmp_path))
    message = str(excinfo.value)
    assert "SPEC.md 5.3" in message
    assert "preregistration.md" in message
    assert "--synthetic" in message


def test_a_missing_preregistration_is_also_refused(tmp_path):
    with pytest.raises(RuntimeError, match="MISSING"):
        dataset.require_frozen_preregistration(tmp_path / "nope.md")


def test_a_frozen_preregistration_opens_the_gate(tmp_path):
    frozen = tmp_path / "preregistration.md"
    frozen.write_text("# list\n\n**Status: FROZEN — 2026-08-20.**\n")
    assert dataset.preregistration_status(frozen) == "FROZEN"
    dataset.require_frozen_preregistration(frozen)  # does not raise


def test_load_real_stops_at_the_guard_before_touching_the_filesystem(tmp_path, monkeypatch):
    """The guard runs first, so a missing data/derived/ is never even the complaint.

    `derived_dir=tmp_path` exists and holds no tables, so a loader that checked the
    filesystem first would raise FileNotFoundError. The RuntimeError proves the ordering.
    """
    monkeypatch.setattr(dataset, "PREREGISTRATION", fixtures.draft_preregistration(tmp_path))
    with pytest.raises(RuntimeError, match="OFF LIMITS"):
        dataset.load_real(derived_dir=tmp_path)


def test_the_committed_preregistration_is_frozen():
    """The positive integrity claim, checked rather than assumed.

    Every other guard test injects its own file, which means nothing would notice if the
    repo's real `preregistration.md` were un-frozen or its Status line reworded. This is
    the one test that reads the committed file on purpose.
    """
    assert dataset.preregistration_status() == "FROZEN"


def _results_outside_synthetic() -> set[Path]:
    """Everything under results/ that is not part of the committable synthetic tree."""
    results = phaseio.REAL_RESULTS_DIR
    if not results.exists():
        return set()
    synthetic_dir = phaseio.SYNTHETIC_RESULTS_DIR
    return {
        path
        for path in results.rglob("*")
        if path != synthetic_dir and synthetic_dir not in path.parents
    }


def test_real_is_still_refused_when_frozen_but_no_data_has_arrived(capsys):
    """The SECOND gate — the one actually holding the door shut today.

    Since `1c7196d` the prereg is FROZEN, so gate one is open and this is the only thing
    left between `--real` and an analysis. It had zero coverage until 2026-08-18: the whole
    guard suite was resting on a gate that had already been opened. No monkeypatching here
    on purpose — this asserts a fact about the repo as it stands.
    """
    assert dataset.preregistration_status() == "FROZEN"
    before = _results_outside_synthetic()
    assert synthetic.main(["--real"]) == 1
    assert "no derived tables" in capsys.readouterr().err
    assert _results_outside_synthetic() == before, "a refused --real run wrote to results/"


def test_no_real_data_has_entered_the_repo():
    """Turns "nothing has run on real data" into a checked fact, not a PROGRESS.md claim.

    `data/raw/` is where exports would land and `data/derived/` is what `ingest` would
    build from them. Both are empty, so there is nothing for a phase to have read even if
    a guard had failed. If an export ever does arrive this test fails first, and whoever
    put it there has to come and read the pre-registration rules.
    """
    raw = dataset.REPO_ROOT / "data" / "raw"
    for folder in ("fixr", "costs", "other"):
        contents = sorted(p.name for p in (raw / folder).iterdir())
        assert contents == [], f"data/raw/{folder} is not empty: {contents}"
    assert not dataset.DERIVED_DIR.exists(), f"{dataset.DERIVED_DIR} exists — ingest has run"


def test_resolve_is_the_only_synthetic_loader_and_returns_the_tables_and_the_truth():
    """There is one loader per source (review finding, 2026-08-12: there used to be two).

    `phaseio.resolve` is it for the synthetic side — it calls the generator directly
    because it needs the ground truth as well as the three tables.
    """
    run = phaseio.resolve(argparse.Namespace(real=False, synthetic=True, seed=7))
    assert run.source == "synthetic"
    assert run.seed == 7
    assert len(run.events) == 120
    assert len(run.transactions) > 1000
    assert len(run.event_tier) > 200
    assert run.truth["beta_true"] == -0.80
    assert not hasattr(dataset, "load")


def test_features_list_is_usable_and_deduplicated():
    assert len(dataset.FEATURES) == len(set(dataset.FEATURES))
    assert all(isinstance(f, str) and f for f in dataset.FEATURES)
    # SPEC.md 5.6: the inseparable bundle is one entry, not four.
    assert "start_of_semester" in dataset.FEATURES
    assert "freshers_week" not in dataset.FEATURES
    # SPEC.md 8.9: the growth control has to be on the list or every calendar effect lies.
    assert "academic_year" in dataset.FEATURES


def test_features_needing_external_data_are_listed_separately():
    assert "weather" in dataset.FEATURES_NEEDING_EXTERNAL_DATA
    assert not set(dataset.FEATURES) & set(dataset.FEATURES_NEEDING_EXTERNAL_DATA)


# Each slug in `dataset.FEATURES`, and the words of the bullet in the FROZEN
# preregistration.md it was written from. The file is prose and the constant is code; this
# is the join between them. Several slugs share a bullet, which is the point: the frozen
# list says "Venue / city / capacity" on one line and SPEC.md §5.6 groups freshers' week,
# semester start, the loan instalment and weather into one undecomposed effect.
FEATURE_BULLETS = {
    "start_of_semester": "Freshers' week",
    "term_phase": "Semester start / mid / end",
    "exam_period": "Revision and exam periods",
    "reading_week": "Reading week",
    "end_of_term": "End-of-term release",
    "vacation_vs_term": "Vacation vs term time",
    "ball_season": "Ball season",
    "day_of_week": "Day of week",
    "lead_time_days": "Lead time / days-to-event",
    "days_since_last_event": "Days since our last event",
    "events_in_trailing_14d": "Events in trailing 14 days",
    "artist_billing_tier": "Artist billing tier",
    "venue": "Venue / city / capacity",
    "city": "Venue / city / capacity",
    "capacity": "Venue / city / capacity",
    "brand": "Brand (A vs B)",
    "academic_year": "Academic-year cohort effects",
}


def test_every_feature_slug_traces_to_a_bullet_in_the_frozen_preregistration():
    """The reconciliation done at the freeze, pinned so it cannot quietly rot.

    `dataset.FEATURES` used to carry a "THIS LIST IS PROVISIONAL — copy the frozen one in"
    comment. The list is frozen now, the two agree on substance, and this is what keeps
    them agreeing: a slug with no bullet behind it would be a pattern nobody pre-registered.
    """
    assert dataset.preregistration_status() == "FROZEN"
    bullets = [
        line.strip()
        for line in dataset.PREREGISTRATION.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- [")
    ]
    assert set(FEATURE_BULLETS) == set(dataset.FEATURES), "a slug was added or removed unmapped"
    for slug, words in FEATURE_BULLETS.items():
        assert any(words in bullet for bullet in bullets), (
            f"{slug}: nothing in the frozen preregistration.md says {words!r}"
        )


# --------------------------------------------------------------------------------------
# 12. CLI
# --------------------------------------------------------------------------------------


def test_cli_synthetic_run_prints_all_three_betas(capsys):
    assert synthetic.main(["--synthetic", "--seed", "42"]) == 0
    out = capsys.readouterr().out
    assert "BETA_TRUE" in out
    assert "naive pooled" in out
    assert "event FE + lead" in out


def test_cli_real_run_is_refused(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "PREREGISTRATION", fixtures.draft_preregistration(tmp_path))
    assert synthetic.main(["--real"]) == 1
    assert "OFF LIMITS" in capsys.readouterr().err
