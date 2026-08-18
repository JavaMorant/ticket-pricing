# PROGRESS — ticket-pricing

**Last updated:** 2026-08-18 (prereg-guard tests repaired — suite green at 412; see the
2026-08-18 entry at the end. Previously 2026-08-12: adversarial review of Phases 1–6 applied
in full — findings 1–8, then 9–24 in a second pass. 24 of 24 fixed)
**Phase:** 1–6 DONE on synthetic data + REVIEWED. Phase 0/0.5 done, **no data ingested**
**Branch:** `main`, public at github.com/JavaMorant/ticket-pricing

---

## Status in one line

The ingest framework runs end to end on synthetic data and refuses to run on real data
until four column maps are filled in; a seeded synthetic programme with a KNOWN elasticity
now exists so Phases 1–6 can be built and tested before the exports land. Phases 1–6 have
been through an adversarial review — all 24 findings fixed, every testable one with a
regression test that was verified to fail against the defect it guards.

---

## Phase status

| phase | status | runs | report | reviewed |
|---|---|---|---|---|
| 0 — ingest / normalise / tables / validate | DONE | `pricing.ingest` (real path gated) | — | earlier |
| 0.5 — synthetic ground truth | DONE | `-m pricing.synthetic --synthetic` | — | earlier |
| 1 — descriptives | DONE | `-m pricing.phase1 --synthetic` | `phase1_descriptives.md` + 5 figures | **2026-08-12, 4 findings fixed** (1, 11, 13, 14) |
| 2 — demand forecast | DONE | `-m pricing.phase2 --synthetic` | `phase2_demand_forecast.md` + 3 figures | 2026-08-12, plumbing findings only (12, 13, 14) |
| 3 — sales curve | DONE | `-m pricing.phase3 --synthetic` | `phase3_sales_curve.md` + 3 figures | 2026-08-12, plumbing findings only (12, 13, 14) |
| 4 — elasticity | DONE | `-m pricing.phase4 --synthetic` | `phase4_elasticity.md` | **2026-08-12, 7 findings fixed** (5, 6, 15, 18, 19, 22, 23) |
| 5 — break-even / P(clear) | DONE | `-m pricing.phase5 --synthetic` | `phase5_risk.md` | **2026-08-12, 6 findings fixed** (1, 3, 7, 16, 20, 21) |
| 6 — counterfactual | DONE | `-m pricing.phase6 --synthetic` | `phase6_counterfactual.md` | **2026-08-12, 5 findings fixed** (2, 4, 10, 17, and 12's plumbing) |
| 7 — Streamlit + post-event auto-report | NOT STARTED | — | — | — |

All six refuse `--real` (exit 1, reason printed to stderr) at **two gates in order**: the
pre-registration gate (`preregistration.md` must not be DRAFT or MISSING, SPEC.md §5.3) and
then the data gate (the three derived tables must exist). Since commit `1c7196d` the list is
FROZEN, so gate 1 is open and **gate 2 is what closes the real path today**. `data/raw/` is
empty, `ingest` has never run, `data/derived/` does not exist, and nothing outside the guard
reads them. Both gates are tested in all six phases (see 2026-08-18 below).

---

## Adversarial review of Phases 1–6 (2026-08-12) — 8 findings, all fixed

A second pass ran every phase on `--synthetic`, refused every `--real`, and executed
deliberate sabotages in scratch copies to see which claims were actually pinned by tests.
**No BLOCKERs. Eight MAJOR findings, all fixed, each with a regression test that was
verified to fail against the defect it guards.**

| # | where | the defect | the fix |
|---|---|---|---|
| 1 | `phase5` | report ASSERTED "4,000 draws → 5th percentile stable to about a pound". False: the worst event moved **£184** between bootstrap seeds | `N_DRAWS` 4,000 → 40,000, and `run()` now **measures** the two-seed spread and prints it (worst event £31, median £0). No precision is asserted anywhere |
| 2 | `phase6` | "+17% at beta −1.5 and +12% at beta −0.6" was **literal text** in the prose — correct on one seed, contradicting its own table on the next | `_contradiction_sentence()` reads both rows off `results["curve"]`; test compares the sentence to the table on two seeds |
| 3 | `phase5` | "The **three** events that were pulled were pulled *because* they were selling badly" — hardcoded count plus a causal claim the data cannot support | count comes from `events["cancelled"].sum()`; the claim now says the presale histories were destroyed on-platform and the bias is real but unquantifiable (DECISIONS.md, survivorship) |
| 4 | `phase6` | NOT-IDENTIFIED report refused a headline on line 8, then printed `programme uplift: **+5.3%**` unqualified on line 48. Old test only matched one exact string | bold comes off and `PROVISIONAL —` goes on the same line as every programme percentage; test now **scans** for any bold percentage outside the assumed-beta curve on a line without PROVISIONAL |
| 5 | `phase4` | clustered VCE had **119 parameters from 117 clusters** (singular meat matrix, rank ≤ 116) and the CI used the normal 1.960 | FE fitted by **within transform** (2 params, 117 clusters) with `use_t=True` → t(G−1) = 1.981. Dummy form kept, its slope pinned equal to 1e−9, **its wider interval printed alongside**, and the report computes whether the choice could have changed the verdict (it could not) |
| 6 | `phase4` | "CI lies below zero" was presented as one of four identical criteria — i.e. you can only ever be "identified" when the sign came out as predicted | `Check.kind` splits the verdict into **design criteria** and **a sign check, stated as such**, with the argument for keeping it and why it is not p-hacking, mirrored in EXPLAINERS + a new viva Q |
| 7 | `phase5` | **nothing pinned the bootstrap.** Swapping `rng.choice(residuals)` for `rng.normal(0, std)` left all 368 tests green while the report claimed "no normal distribution is fitted to anything" | draws extracted into one `bootstrap_units()`; test asserts the distinct simulated ticket counts are exactly the set the residual array can produce, plus a test that `simulate_event` goes through it |
| 8 | `phase1` | claimed it "never crosses an outcome with a calendar feature" — but prints median tickets by venue, sell-through by brand, sellout rate by year, all §5.4 items. Its guard test grepped for words that can never appear | claim restated precisely (nothing is **tested**: no p-value, no FDR, no significance claim) with §8.9 as the reason the cross-tabs are required; guard split into three tests, one of which scans the report body for inference vocabulary |

**Verified good by the review, no action needed:** the FE is genuinely within-event (not
accidentally pooled); SEs are clustered and clustering bites (plain 0.043 → HC1 0.051 →
clustered 0.063); grouped splits are disjoint by group; the Monte Carlo resamples at event
level; `assert_no_lookahead` is behavioural rather than nominal; Phase 3's fold curves are
built from train events only; the privacy plumbing is correct (`results/*` ignored,
`results/synthetic/` un-ignored, `*.parquet` still ignored, no phase reads `data/`); the
DRAFT guard refuses all six; the wrong-sign and beta-recovery pins bind under sabotage.

**Six sabotages re-run after the fixes, all now caught:** parametric draw instead of
bootstrap → `test_the_monte_carlo_resamples_real_residuals_and_fits_nothing`; bold
percentage in a NOT-IDENTIFIED report →
`test_a_not_identified_report_never_prints_an_unqualified_bold_percentage`; hand-typed
prose percentages → `test_the_contradictory_advice_sentence_is_read_off_the_curve[42]`;
`use_t` removed → `test_the_interval_uses_a_t_on_g_minus_one_clusters_not_a_normal`;
a significance claim added to the Phase 1 body →
`test_report_body_contains_no_inference_vocabulary_at_all`; demeaning on the wrong key →
20 failures including the beta-recovery acceptance tests.

`uv run pytest -q` → **393 passed**, exit 0 (was 368; +25). `ruff check` / `ruff format
--check` clean on `src` and `tests`. All six reports regenerated from the fixed code.

*(That pass was handed a findings list truncated mid-finding-8. The tail arrived and was
applied in the second pass below, which closes the review.)*

---

## Same review, second pass (2026-08-12) — findings 9–24, all fixed

The rest of the same adversarial review: one MAJOR pair (9, 10) and the whole MINOR
section. **16 findings, 16 fixed, none skipped.** Every fix that could carry a test has
one, and each new test was re-run against the re-introduced defect to prove it fails.

| # | where | the defect | the fix |
|---|---|---|---|
| 9 | `docs/VIVA.md` | only five of SPEC §10's **nine** questions were answered, and the survivorship answer DECISIONS.md marks non-deferrable was missing entirely. §10 answers are a first-class deliverable, so this was a gap in the product | new closing section with the other four (overfit / data-mining the calendar / freshers-vs-loan / more time) plus the survivorship question, in the same one-breath register. The §5.6 answer says plainly that freshers ≈ semester start ≈ loan instalment ≈ weather arrive together for five years, that ~120 events cannot separate them, and names the two datasets that could (a year where the instalment date moved; cross-institution term dates) |
| 10 | `phase6` | `## Headline: +5.3%` was **100% a grid-boundary artefact** — every event picked the top corner of the level grid. The report body said so; the headline did not | the headline is now the **+2.6% at the experiment's pre-committed +15% dose** — the one uplift with no search in it. The grid optimum is demoted to a sensitivity block labelled *"Corner solution — this is the grid bound, not a result"* and read as "at least +5.3% within the x0.70–x1.30 range considered". Terminal summary and `docs/` follow |
| 11 | `phase1` / `phaseio` | **two** `academic_year` functions, disagreeing at the century roll-over ("2099/100" vs "2099/00"); Phase 4 used one, Phases 1/2/3/5 the other | `phase1.academic_year` deleted, one function in `phaseio`, and the surviving one's roll-over bug fixed (`(start + 1) % 100`). Test on the September boundary **and** on 2099 |
| 12 | `report.py` / `phaseio.py` | **two shared-plumbing modules doing the same four jobs** — CLI, source loader, output-dir rule, report writer — with `write_report`'s arguments transposed between them. The largest simplicity cost in the diff | merged into `phaseio`: one CLI, one `resolve()`, one `write_report(text, filename, out_dir)` taking a string or a list, one report banner for all six phases. `report.py` keeps only what a report *reads like* (markdown tables, figure style) and phases 4–6 never import it. `dataset.load` deleted with it — it had no callers left, and one loader per source is the point |
| 13 | `report.py` | Phases 1–3 recorded **no seed**: `--seed` defaulted to None and the header suppressed a falsy seed, so the committed reports said only ``Source: `synthetic` `` | `--seed` defaults to `DEFAULT_SEED` everywhere and the banner always prints it, plus `Regenerate:` with the exact command. Pinned through the CLI, since the default is the half a fixture-built report cannot see |
| 14 | `report.py` | the DRAFT refusal went to **stdout** for Phases 1–3 and stderr for Phases 4–6, so `2>/dev/null` hid half of them | stderr everywhere; the three Phase 1–3 tests now assert it is on stderr and **not** on stdout |
| 15 | `phase4` | the verdict table inferred its comparator from the check's NAME, rendering the strict `ci_high < 0` rule as `<= 0.000` — a reader reproducing the verdict by hand got a different rule at the boundary | `Check.comparator` carries the comparison that was actually made. Test applies the printed comparator to the printed numbers and requires it to reproduce the printed PASS/FAIL |
| 16 | `phase5` | `max_guarantee_at_target` went negative and printed as a negotiating number: `median payable fee @50% £-1,001` | clamped at 0, with the meaning stated: **£0 = no fee clears at this room**, not even a free booking. Table cells and the programme median say it in words |
| 17 | `phase6` | contribution hardcoded as `face_price`, bypassing `phase5.VARIABLE_COST_PER_TICKET` — right today (it is 0.00) and silently wrong the day a per-head cost lands | both sides of the arithmetic read Phase 5's constant. Test moves the constant to £1.50 and requires Phase 6's numbers to move with it |
| 18 | `phase4` | `iv_estimate` dropped NaNs on `y`, `p`, `z` only. One null venue on a real export and statsmodels' `missing="drop"` returns a shorter residual vector — after which `z~ · p~` dots two different sets of events together, silently | rows are dropped once, on the controls too (`_control_columns`), and `_residualise` passes `missing="raise"` as the tripwire. `exclusion_balance_p` and `instrument_trend_share` drop their own. Test uses a panel with a null venue row |
| 19 | `phase4` | the first-stage F ignored the df the residualisation consumed, so it was overstated — in the direction that lets a weak instrument through the floor | scaled by `(n − k_controls − 1) / n`. F changes: spec I **3.0 → 2.8**, II 16.3 → 15.6, III 104.6 → 96.5. No relevance verdict flips; `docs/` updated to match |
| 20 | `tests/test_phase5.py` | `test_simulated_units_never_exceed_the_room` asserted on the **mean** of the draws — a broken clip letting 1% of draws through the roof would have passed | asserts every draw, on an event whose forecast is four times its capacity, and requires the clip to bind first ("or this test proves nothing") |
| 21 | `phase5` | the bootstrap RNG seed was a hardcoded default argument, absent from the "Assumptions, all of them" table that claims to list every chosen number | `BOOTSTRAP_SEED` named with the others, carried in `results`, printed as an assumptions row, and the Reproduce block says what `--seed` does *not* re-roll |
| 22 | `phase4` | `out["sell_through"]` computed in `build_event_panel` and never used | deleted |
| 23 | `phase4` | `from pricing.dataset import FEATURES  # noqa: F401` — an import existing only to make a future reader re-check the module | deleted; the comment above it was doing all the work and stays |
| 24 | `docs/VIVA.md` | a committable doc in a public repo carried an internal process note naming the owner | reworded to "no number from this repo is quoted in an interview until these vivas are passed"; the personal gate lives here now (see *Owed before the CV bullet is written*) |

**Sabotage evidence.** Each of the eleven testable fixes was re-broken in place and its
guard test run alone: all eleven failed, and the source was restored between each. The five
non-testable ones are documentation (9, 24), deletions (22, 23) and the module merge (12,
pinned indirectly by `not hasattr(dataset, "load")` and `not hasattr(phase1,
"academic_year")`).

`uv run pytest` → **402 passed**, exit 0 (was 393; +9). `ruff check` / `ruff format --check`
clean on `src` and `tests`. All six reports regenerated; all six still refuse `--real` with
the DRAFT reason, on stderr, exit 1.

**Synthetic headline numbers that moved:** Phase 6's headline is now **+2.6%** (the +15%
dose) with the +5.3% grid optimum demoted to a stated bound; Phase 5's median payable fee is
**£0 with the reason attached** instead of −£1,001; Phase 4's instrument F statistics fall
by the df correction (2.8 / 15.6 / 96.5). Beta, the verdict, the CIs, the MAEs and the
ground-truth check are unchanged.

---

## Phases 4–6 — elasticity, risk, counterfactual (2026-08-12) — THE STATISTICAL CORE

**Status: DONE on synthetic data.** Three scripts, three markdown reports in
`results/synthetic/`, all refusing `--real` while `preregistration.md` says DRAFT.

```
uv run python -m pricing.phase4 --synthetic     # phase4_elasticity.md
uv run python -m pricing.phase5 --synthetic     # phase5_risk.md
uv run python -m pricing.phase6 --synthetic     # phase6_counterfactual.md
```

### Headline numbers (seed 20260811, 117 events / 402 panel rows) — post-review

```
BETA_TRUE                                       -0.800
naive pooled OLS      log Q ~ log P             +0.698   <-- WRONG SIGN (SPEC 4.1)
event FE only         within transform          -1.260   <-- lead-time confound (SPEC 4.3)
event FE + lead       within, + lead_term       -0.815   CI [-0.920, -0.711]   <-- SPEC 4.2
  the same with 117 event dummies               -0.815   CI [-0.939, -0.691]   (printed alongside)
IV I  hire rate, venue + year FE                not run  F = 2.8  (excludable, too weak)
IV II hire rate, venue FE only                  +0.366   F = 15.6 (strong, NOT excludable)
IV III artist guarantee                         +1.008   F = 96.5 (invalid by construction)

within: 2 params / 117 clusters, t(116) = 1.981   dummies: 119 params / 117 clusters
residual price variation after FE + lead         0.065   (VIF 15.4, flagged not failed)
VERDICT                                          IDENTIFIED
                                                 3 design criteria + 1 sign check, all pass

Phase 5: median break-even 586 tickets (113% of the room), median P(clear) 0%
         40,000 draws; measured MC error: worst event's p05 moves £31 between seeds
         example fee curve FAKE-EV-020 (most headroom), 3 cancelled events excluded
         mean E[profit] fixed -£2,709 / split -£302 / guarantee-or-split -£2,712
Phase 6: HEADLINE projected uplift +2.6% (the pre-committed +15% dose, no search)
         grid optimum +5.3% -- a BOUND: 100% of events sit at the grid edge
         uplift curve: beta -0.4 -> +18.9% (raise prices), beta -1.0 -> +0.0%,
                       beta -2.0 -> +34.5% (cut prices)  <-- contradictory advice
         ground-truth check: model +5.3% vs true process +8.7%, ratio 0.61, signs agree
```

The FE interval changed at the review (finding 5): the headline is now the within-transform
interval [-0.920, -0.711], and the dummy interval [-0.939, -0.691] is printed next to it so
the estimator choice is visible. Both are inside the 1.00 CI-width criterion and both lie
below zero, so the verdict does not turn on it — the report computes and states that.

`uv run pytest -q` → **393 passed**, exit 0 at that pass; **402** after findings 9–24.
`ruff check` / `ruff format --check` clean.

### Three findings a reviewer should look at

1. **The cost-shifter instrument fails honestly, in both directions.** Uncontrolled, the
   venue hire rate has a strong first stage (F = 15.6) and is not excludable — hire rates and
   brand demand both trend upward (SPEC 8.9), so the instrument is partly a time trend, and
   the estimate comes out wrong-signed. Control the year and it is excludable but weak
   (F = 2.8), below the stated relevance floor of 10, so **no 2SLS estimate is reported**. The
   artist guarantee — the invalid instrument — has the strongest first stage in the table.
   That pairing is the SPEC 4.5 lesson, and it is now a pinned test.
2. **Phase 6's optimum is a corner solution, and since finding 10 it is no longer the
   headline.** With `|beta| < 1`, constant-elasticity revenue rises with price without
   limit, so the optimum sits on the grid's upper edge for 100% of events and the ±30% bound
   is an *assumption*, not a result. The headline is now the uplift at the experiment's
   pre-committed +15% dose (+2.6%, no search, no corner) and the grid number is printed
   below it as "at least +5.3% within the range considered". Same for "the model wants to
   flatten the ladder" — arithmetic from one beta per rung, not advice.
3. **The synthetic P&L is not a P&L.** The generator's cost stack was never calibrated
   against its ticket prices (median break-even is 1.1x median tickets sold), so P(clear)
   is ~0 across the fixture. Phase 5's report prints that as a box before any number, and
   reports the *shapes* — structure ordering, tail behaviour, the fee curve — as the things
   worth reading until the real cost stack lands.

### Decisions taken here that a reviewer should check

- **Phase 5 fits no demand model.** It takes Phase 2's grouped-K-fold out-of-fold
  predictions and bootstraps their residuals; units and revenue come from
  `phase1.event_outcomes`. Two demand models in one repo means two answers to "how many
  will come".
- **The identification verdict is a function of four numeric criteria** in
  `phase4.IDENTIFICATION_CRITERIA`, written before the estimate was looked at, and passed
  as an argument so a reviewer can re-run at their own thresholds. VIF is a *flag*, not a
  criterion: what binds is the CI width, because a VIF says only how inflated the standard
  error is, not whether the interval is usable.
- **2SLS is hand-rolled as a ratio** (just-identified IV via Frisch-Waugh-Lovell), not two
  stacked OLS calls — the second-stage residuals from the naive version are computed
  against fitted prices and come out too small.
- **`phaseio.py` is the one plumbing module for all six phases** (CLI, source resolution,
  where reports may be written, the writer, the banner, the academic-year label). It was two
  modules until finding 12; `report.py` now holds only markdown tables and the figure house
  style, which phases 4-6 do not use. A real run cannot write real margins into the
  committable `results/synthetic/` tree because only `resolve()` picks the directory.
- **Phase 6 re-runs Phase 4** rather than caching its verdict, because the honesty rule
  ("no identified beta, no headline number") has to read a verdict that came from this
  data, not from a file someone edited.

### Owed before the CV bullet is written

- **Fable gate 2 (Phase 4 identification review)** is still owed — DECISIONS.md makes it the
  gate before the elasticity bullet exists. The 2026-08-12 adversarial review covered
  Phases 1–6 and found the clusters-vs-parameters problem and the sign-criterion framing
  (findings 5 and 6 above), both now fixed; it does **not** substitute for the named gate,
  which is a review of the identification argument itself.
- **The vivas.** `docs/VIVA.md` covers **all nine** of SPEC §10's questions with one-breath
  answers (findings 9 and 24 closed the four that were missing, plus survivorship). Per the
  2026-08-11 amendment: **no interview quotes any number from this repo until Awande passes
  these vivas out loud.** That gate lives here rather than in the public `docs/VIVA.md`,
  which states the rule impersonally.

---

## Phases 1–3 — descriptives, demand forecast, sales curve (2026-08-12)

**Status: DONE on synthetic data.** All three run as scripts, write a markdown report plus
figures to `results/synthetic/`, and refuse `--real` while `preregistration.md` says DRAFT.
No real data has been read; `data/raw/` and `data/derived/` are still empty.

```
uv run python -m pricing.phase1 --synthetic     # descriptives
uv run python -m pricing.phase2 --synthetic     # demand forecast
uv run python -m pricing.phase3 --synthetic     # sales curve
```

**Definition of done, per phase:**

| phase | observable outcome | result (synthetic, seed 20260811) |
|---|---|---|
| 1 | counts/distributions/ladder shape reported, **nothing tested** — no p-value, no FDR, no significance claim, no calendar feature crossed with an outcome | 117 events ran + 3 pulled, 32,999 paid tickets, median sell-through 51%, sellout rate 3.4%, ladders of 2–5 rungs |
| 2 | held-out MAE beats a naive brand-venue mean, look-ahead check passes | CV MAE **42.7** (14.5% MAPE) vs baseline 68.0 — **37% better**; final-year holdout MAE **53.0** (16.6%) vs 80.1 — **34% better** |
| 3 | forecast from partial presale beats "assume no more sales", error falls towards the event | day −7 MAPE **15.7%** vs 29.0% floor; day −3 12.9%; day −1 11.2% |

**Two findings worth carrying forward, both negative and both reported:**

- Brand segmentation of the sales curve buys **nothing** (−0.11 MAPE points). Use the
  pooled curve; the segmentation code stays for real data.
- At day −7 the presale forecast is **worse** than the Phase 2 pricing-time model
  (15.7% vs 14.5%) and only overtakes it at day −3. The two use different information and
  should be combined — Phase 5's job, not Phase 3's.

**Discipline enforced by tests, not by intent:**

- `test_report_crosses_no_calendar_feature_with_an_outcome` — fails if a §5.4 calendar word
  ever appears in a *results table* of the Phase 1 report. Prose explaining the absence is
  fine. (Renamed and joined by two siblings at the 2026-08-12 review, finding 8: on its own
  it could never fail, and the claim it guarded was too strong — Phase 1 does cross
  outcomes with brand, venue and year, and now says so.)
- `test_report_body_contains_no_inference_vocabulary_at_all` — no `p-value`, `significan…`,
  `FDR`, `Benjamini`, `confidence interval`, `correlat…`, `regression` or `coefficient`
  anywhere in the body of the Phase 1 report. This is the claim the report actually makes.
- `test_report_discloses_the_cross_tabs_it_does_show` — the preamble must name the
  year/brand/venue cross-tabs and cite §8.9. Claim and content, pinned together.
- `assert_no_lookahead` — rebuilds the whole Phase 2 feature frame from transactions
  truncated at each event's first sale and requires it to be identical. A planted leak
  (`sell_through` as a feature) is tested to fail it.
- `grouped_kfold` — hand-rolled, ten lines, tested on groups that actually repeat.
- Phase 3's median curves are refit per fold, so no event is forecast by a curve it drew.

**Files:** `src/pricing/report.py`, `phase1.py`, `phase2.py`, `phase3.py`;
`tests/test_phase1.py`, `test_phase2.py`, `test_phase3.py`; `docs/EXPLAINERS.md` +
`docs/VIVA.md` gained a section and 5–6 questions per phase; `matplotlib` added to
`pyproject.toml`. Nothing committed.

**~~Known duplication to reconcile~~ — done (finding 12, 2026-08-12).** `report.py` and
`phaseio.py` were written in parallel and overlapped on all four plumbing jobs. `phaseio`
absorbed the lot; `report.py` keeps markdown tables and the figure style, and phases 1–3
`build(run)` from the same `Run` object phases 4–6 use.

---

## Phase 0.5 — synthetic ground truth (2026-08-12)

**Definition of done, all three observable:**

1. `generate(seed)` returns the three derived tables in exactly `tables.py`'s schemas —
   verified column-for-column and dtype-for-dtype against `tables.build_*` output.
2. The naive pooled regression on the fixture comes out **wrong-signed** (SPEC §4.1).
3. The within-event regression recovers **BETA_TRUE = −0.80** inside tolerance (SPEC §4.2).

**Evidence** (`uv run python -m pricing.synthetic --synthetic`, seed 20260811):

```
  events            120   2021-09-23 .. 2026-06-20   (3 cancelled)
  transactions   33,804   585 comps
  event_tier        480   402 usable panel rows
  buyer price   £4.40 .. £24.75
  capacity      150 .. 1500   venues ['V1','V2','V3','V4']   brands ['Brand A','Brand B']

  BETA_TRUE                                     -0.800
  naive pooled      log Q ~ log P               +0.698   <-- WRONG SIGN (SPEC 4.1)
  event FE only     + C(event_id)               -1.260   <-- lead-time confound (SPEC 4.3)
  event FE + lead   + lead_term + C(event_id)   -0.815   <-- recovers BETA_TRUE (SPEC 4.2)
```

`uv run pytest -q` → **212 passed** (was 149), exit 0. `ruff check` / `ruff format --check`
clean.

### Files added

```
src/pricing/synthetic.py       the generator + the two acceptance regressions + CLI
src/pricing/dataset.py         load("synthetic"|"real"), the DRAFT-preregistration guard,
                               and the FEATURES list (provisional — final contents come
                               from the frozen preregistration.md)
tests/test_synthetic.py        63 tests
docs/EXPLAINERS.md             Phase 0.5 section: method, why, what can go wrong
docs/VIVA.md                   8 viva questions with answers
results/synthetic/README.md    what is in results/synthetic and why the parquet is not
                               committed
```

### The four acceptance tests Phase 4 inherits

`tests/test_synthetic.py` asserts these on three seeds each. The Phase 4 estimator has to
reproduce all four on this fixture before it is pointed at real data:

| test | asserts |
|---|---|
| `test_naive_pooled_regression_has_the_wrong_sign` | pooled β > 0 |
| `test_within_event_regression_recovers_beta_true` | \|β̂ − (−0.80)\| < 0.15 with the lead-time control |
| `test_lead_time_confound_biases_plain_event_fe` | plain event FE is visibly over-elastic while `lead_confound > 0` |
| `test_switching_the_confound_off_makes_plain_fe_work` | set `lead_confound = 0` and plain FE recovers β |

### What is planted, and where

Demand shock per event (artist billing tier, brand growth, academic-calendar position,
noise); the operator's price loads on that shock at `lambda = 0.55` — that, and nothing
else, is what makes the pooled regression wrong-signed. On top: the §4.3 lead-time
confound (tunable, `lead_confound = 0.08`), the §8.9 brand growth trend (0.16 and 0.10 log
points per year, centred on the middle year), a **valid** cost-shifter instrument (venue
hire index, moves price only) and an **invalid** one (`artist_guarantee`, moves price and
demand), endogenous marketing spend, 3 cancelled events (§8.4), comps excluded from
`units_sold` (§8.7), and `fee_treatment = "UNKNOWN"` carried on every row (§8.6).

### Guard on real data

`pricing.dataset.load_real()` reads the Status line of `preregistration.md` and **refuses
while it says DRAFT**, printing SPEC §5.3, the file path, and the two-word edit that opens
it. `uv run python -m pricing.synthetic --real` exits 1 with that message. Nothing in the
repo reads `data/raw/` or `data/derived/` outside `ingest.py` and that guard.

### Decisions taken here that a reviewer should check

- **`results/` is now gitignored, `results/synthetic/` is not** (`results/*` + a negation,
  because a plain `results/` would stop git descending). Real-data results stay local
  until the pre-publication gate.
- **The synthetic parquet fixtures are still ignored** by the repo-wide `*.parquet` rule,
  deliberately: they regenerate from a seed in one command, tests call `generate()`
  directly, and un-ignoring them would cost the "nothing shaped like an export reaches
  git" guarantee. Rationale written up in `results/synthetic/README.md`; one negation line
  reverses it if a later phase needs them committed.
- **`dataset.py` is a second file, not part of `synthetic.py`.** The freeze guard, the real
  loader and the FEATURES list are shared plumbing every phase imports; a phase module
  importing them from `synthetic` would read wrong. The generator itself is one file, as
  specified.
- **The brand-growth test needs controls to see the trend** (sell-through, artist tier,
  calendar, capacity, hire index). That is not test-fudging — hire rates drift up, prices
  follow, sell-through falls, and the raw trend can genuinely vanish into the venue mix.
  It is SPEC §8.9's lesson happening inside its own test, and the test says so.

---

## What exists and is verified

| | Evidence |
|---|---|
| `uv sync` resolves, py3.12 | exit 0; `uv.lock` committed to the working tree |
| `uv run pytest -q` | **149 passed in 0.82s**, exit 0 |
| `uv run ruff check src tests` | All checks passed (`ruff format --check`: 12 files formatted) |
| `uv run python -m pricing.ingest --sniff` | exit 0 on the empty `data/raw/`, prints the expected-layout message |
| `uv run python -m pricing.ingest` | exit 1 with `no ticket exports found under data/raw` — fails loudly, as designed |
| Full chain (maps filled at runtime, fake exports) | wrote `events` 2 rows / `transactions` 17 / `event_tier` 8 to parquet, printed the validation report, and **stripped the buyer-PII columns that were deliberately mapped through**. Re-run after findings 14–23 with a paid "Tier 1 + Free Drink" tier and one mid-window price change: the paid tier kept its 4 sales instead of being stripped as a comp, the report flagged 1 tier whose price moved, and `fee_treatment=UNKNOWN` reached both parquet files |

### Files

```
pyproject.toml                 py3.12, pandas / statsmodels / openpyxl / pyarrow; pytest + ruff
uv.lock
src/pricing/__init__.py        module map
src/pricing/adapters.py        4 adapter stubs, canonical schemas, FEE_TREATMENT, --sniff
src/pricing/normalize.py       UK date + money parsing, tier -> ordinal, comp flag, PII
src/pricing/tables.py          build_events / build_transactions / build_event_tier / write_table
src/pricing/validate.py        validation report (dict) + format_report (text)
src/pricing/ingest.py          entry point, cross-file de-duplication
tests/synthetic.py             fake fixtures ("FAKE Buyer 001", Brand A/B, V1/V2)
tests/test_normalize.py        96 tests
tests/test_adapters.py         12 tests
tests/test_tables.py           23 tests
tests/test_validate.py         18 tests
```

`.gitignore` extended: `.ruff_cache/`, `.venv/`, `*.xls`, `reports/`, plus a note that the
repo-wide `*.csv` / `*.xls[x]` / `*.parquet` globs are why fixtures are built in Python
rather than committed as files. `data/` was already covered, so `data/raw/` and
`data/derived/` are both ignored — verified with `git check-ignore` on `reports/sniff.txt`,
`reports/validation_report.md`, `notes/export.xls`, `data/derived/validation_report.txt`.

### Privacy guard (DECISIONS.md — non-negotiable)

Four layers, all tested:

1. `strip_pii()` runs inside every adapter, so PII dies before anything else sees the frame.
2. `build_events` / `build_transactions` strip again.
3. `write_table()` calls `assert_no_pii()` and **raises rather than writing** the parquet.
4. That guard is **fail-closed**: it refuses any column not on `normalize.ALLOWED_COLUMNS`
   (the canonical + derived field names), rather than hunting for names that look like
   PII. A pattern list loses to "Lead Booker" or "Instagram Handle"; an allowlist does not.
   The pattern list still exists, but only to *drop* raw columns and to warn in `--sniff`.

`--sniff` prints column names, dtypes, null rates and the date span of genuinely date-like
columns. No cell values, no `head()`, no uniques, no filenames (files are labelled
`fixr/file 1 of 2`). Number columns — including money written as text, "£1,500" — never get
a date range, because `pd.to_datetime` reads a number as epoch nanoseconds and would print
the real min and max. Suspected-PII columns never get one either.

**Neither the sniff output nor the validation report is safe to paste into a chat.** They
carry real tier labels, real column names and real event counts. Local use only; the report
is written to `data/derived/validation_report.txt`, inside the gitignore.

---

## Review outcome (2026-08-11, second pass)

An independent review reproduced every finding by running the code. **13 findings were
legible in the handover (2 privacy BLOCKERs, 2 correctness BLOCKERs, 7 MAJOR, 2 MINOR);
all 13 are fixed, each with a regression test that fails on the old behaviour.** Test count
went 78 → 119.

| # | Sev | What was wrong | Fix |
|---|---|---|---|
| 1 | BLOCKER | `--sniff` leaked money and date-of-birth values: `pd.to_datetime` reads numbers as epoch nanoseconds, so price and guarantee columns printed their real min/max, and PII columns got a range too | `_date_range` skips anything that parses as a number (text money included), skips suspected-PII columns, and emits `.date()` not timestamps |
| 2 | BLOCKER | PII pattern list missed the names real exports use (`Booking Name`, `Ticket Holder`, `Lead Booker`, `Instagram Handle`…), so the report's "no buyer PII" line and the sniff warning both under-reported | `assert_no_pii` is now a fail-closed allowlist; the pattern list was broadened as well but is no longer the last line of defence |
| 3 | MAJOR | Sniff printed real filenames (`BrandB_master_finance.xlsx`); both artifacts claimed to be "safe to paste into a chat" | Files labelled `folder/file N of M`; the safe-to-share claims are replaced with "local use only" in `adapters.py`, `validate.py` and here |
| 4 | MAJOR | The report was never written anywhere despite the docstring, so it would be redirected by hand; `.gitignore` missed `*.xls` and `reports/` | `run_ingest` writes `data/derived/validation_report.txt`; `.gitignore` covers `*.xls` and `reports/` |
| 5 | MINOR | `data/derived` was CWD-relative — running from elsewhere wrote real tables outside the guard | `REPO_ROOT = Path(__file__).resolve().parents[2]`; absolute paths printed |
| 6 | MINOR | `except Exception` put arbitrary pandas error text (which can quote file content) into the "safe" sniff output | Reports `type(exc).__name__` only |
| 7 | BLOCKER | `event_tier` grouped on `(event_id, tier_ordinal)`, which is not unique — five tiers at five prices collapsed to two rows, deleting the within-event price variation SPEC §4.2 identifies β off | Groups on `(event_id, tier_name)`, carries `tier_ordinal` as the group min; `validate` reports any shared ordinal |
| 8 | BLOCKER | No `dayfirst` anywhere: UK `03/04/2024` was read as 4 March | One `parse_uk_datetime()` used by all three call sites, plus an ambiguity share (day ≤ 12) in the report |
| 9 | MAJOR | Mixed GMT/BST offsets killed the run; uniform tz-aware data broke the `lead_time_days` subtraction | Same helper converts to Europe/London and drops the offset — local wall-clock is what a day-of-week feature means |
| 10 | MAJOR | `cancelled` of `"N"`/`"Y"` marked **every** event cancelled (`astype(bool)` on a non-empty string) | Explicit truthy set; count printed in the report |
| 11 | MAJOR | `"£10.00"` prices coerced to NaN, and NaN was treated as `<= 0`, turning every ticket into a comp and `units_sold` into 0 | `parse_number()` strips `£` and separators; an unparsed price is never free and is a loud report failure |
| 12 | MAJOR | A duplicate event row in the finance workbook fanned out the transaction join and doubled `units_sold` silently | `build_events` refuses to build a non-unique `event_id`; count in the report |
| 13 | MAJOR | Every file was concatenated with no dedup, so one overlapping re-pull double-counted revenue | `_drop_repulled_rows` drops rows an earlier file already had, but keeps genuine repeats *inside* one file (two tickets on one order look identical) |

**Nothing was skipped.** No finding was judged not-worth-fixing.

Two things the fixes changed that are worth knowing:

- **pandas is 3.0.5 here, not 2.2.** `dayfirst=True` on its own *mangles ISO dates*
  (`2026-03-01` → 3 January) because it infers one format from the first cell. The helper
  therefore parses ISO first and falls back to `dayfirst` + `format="mixed"` for the rest.
- **`parse_number` tests dtype with `is_numeric_dtype`, not `== object`**, because pandas 3
  gives string columns a `str` dtype and the `object` check silently did nothing.

---

## Review outcome (2026-08-11, findings 14–23)

The rest of the same review, re-sent after the truncation. **Findings 14–17, 19 and 21–23
are fixed, each with a regression test that fails on the old behaviour.** Test count went
119 → **149**. Finding 18 was checked and is genuinely moot; finding 20 is a process
action for the owner, not code.

| # | Sev | What was wrong | Fix |
|---|---|---|---|
| 14 | MAJOR | The comp regex was unanchored, so any tier whose NAME contained `free`/`guest`/`artist` was stripped from demand. Reproduced `comp=True` for `Tier 1 + Free Drink`, `Free Entry Before 11`, `Girls Free B4 11`, `Artist Package`, `Plus Guest` — SPEC 8.7 says strip comps, and an undercount of paid sales is invisible downstream | `is_comp_name` matches the WHOLE cleaned label against `normalize.COMP_LABELS`; the price == 0 test still catches an unlisted comp spelling. The report now prints comp counts **grouped by tier_name**, so a paid tier being stripped is visible in one line |
| 15 | MINOR | `event_date` is midnight, so a door sale at 23:30 on the night gave `lead_time_days = -1` (reproduced) | One `_lead_time_days()` helper floors BOTH sides to the date before subtracting; used by `transactions.lead_time_days` and `event_tier.lead_time_open_days`. Bought on the day = 0. Rule documented in the docstring |
| 16 | MINOR | `_apply_map` never checked that map VALUES are unique, so two raw columns mapped to `price_paid` would silently keep whichever renamed last | Raises before renaming, naming the duplicated canonical field |
| 17 | MAJOR | Under `FEE_TREATMENT="UNKNOWN"`, `buyer_price` was written to parquet as `price_paid + booking_fee` with no marker — the guess was visible only in a report nothing downstream reads (SPEC 8.6) | A `fee_treatment` column is carried on **transactions and event_tier** (and on the allowlist, so it survives the write). Chosen over writing NaN so the pipeline still runs end to end |
| 18 | — | *Skipped as moot — verified.* No merge is keyed on the `<NA>`-prone `tier_ordinal` any more; the panel merges on `(event_id, tier_name)` and on `event_id`. Re-ran with a null `tier_name` on both the paid and the comp side: 8 rows, no fan-out, `units_sold + units_comp` still equals the transaction count | (no change) |
| 19 | MINOR | `event_tier.price` was the modal realised price — an outcome, not the posted price. It used the whole window to describe its own opening (SPEC 8.2) and averaged away mid-window corrections, which are exactly the gold-dust variation SPEC 4.3 wants | `price` / `face_price` are now the FIRST observed price (rows sorted by `purchased_at` first, so "first" means first sold, not first in the file), plus a new `n_distinct_prices` column and a report section counting tiers whose price moved mid-window |
| 20 | — | *Skipped:* owner process action, not code | (no change) |
| 21 | MINOR | `_TIER_RULES`' second tuple element meant three different things (literal ordinal, `-1` = captured digits, `-2` = digits − 1) and parked Final/Last Release on a 98 sentinel that two spellings shared | A plain `FIXED_TIER_ORDINALS` dict plus two explicit `re.match` blocks (`tier N`, `release N`). The 98 sentinel is gone: a "final release" label states that a tier is last, not which position it is, so it comes back **unmapped** and `window_open` orders it. The report says so instead of asking for a rule |
| 22 | MINOR | The PII mega-regex was searched twice per column (spaced form and underscored form) because one pattern was written with underscores while the `\b` ones only fire on the spaced form | Checked first whether it was dead after the allowlist fix — it is **not**, and a comment now says why: the allowlist only guards derived tables, this list drops arbitrary RAW columns and feeds the `--sniff` warning. Column names are normalised to spaced words and searched **once**; redundant prefixes dropped from the name pattern |
| 23 | MINOR | Copies of the modal-price lambda in `build_event_tier` | Deleted with `_mode` itself — finding 19 replaced it with a plain `"first"` aggregation, so there is no helper left to keep in sync |

---

## Blocked on real data — the exact TODO points

All four are `TODO(real-data)` comments; `grep -rn "TODO(real-data)" src/` finds them.

| # | Where | What is needed |
|---|---|---|
| 1 | `adapters.py` `FIXR_COLUMN_MAP` | `{raw column: canonical field}` for the Fixr per-ticket export (~150 events). Required canonical fields: `order_id, event_key, tier_name, price_paid, purchased_at`. Optional but wanted: `booking_fee, quantity, promo_code`. |
| 2 | `adapters.py` `TBC_COLUMN_MAP` | Same, for TBC.xyz. Also decide whether TBC and Youni need one map or two. |
| 3 | `adapters.py` `YOUNI_COLUMN_MAP` | Same, for Youni. |
| 4 | `adapters.py` `COSTS_COLUMN_MAP` + `COSTS_SHEET_NAME` | The two brand master finance workbooks, 5 years. Required: `event_key, event_date, brand, capacity`. Watch for merged header rows, one sheet per year, and totals rows that are not events. |
| 5 | `adapters.py` `FEE_TREATMENT` | **SPEC §8.6.** All three platforms are `"UNKNOWN"`. Resolve by comparing one real export row against one real order confirmation per platform, then set `"face_plus_fee"` or `"inclusive"`. |
| 6 | `adapters.adapter_for` | `data/raw/other/` files are routed by filename. If the real filenames do not say `tbc` or `youni`, rename them on arrival — do not add guessing logic. |

**Workflow when the exports land:** drop files in → `--sniff` → paste the column names into
the maps → run ingest → read the validation report (UNMAPPED tier labels, the comp counts
by tier name, the tiers whose price moved) → add rules to `FIXED_TIER_ORDINALS` or labels to
`COMP_LABELS` where the report asks for them → re-run.

### Face-vs-fee is the one that poisons everything

While `FEE_TREATMENT` is `UNKNOWN`, `buyer_price` is computed provisionally as
`price_paid + booking_fee`, the validation report prints `<-- RESOLVE THIS`, **and every
row of `transactions` and `event_tier` carries a `fee_treatment` column reading
`UNKNOWN`** — so the guess travels with the number into any downstream join, not just into
a report nobody re-reads. Every price-side result (elasticity, counterfactual, uplift) is
provisional until this is settled. It is cheap to resolve and expensive to get wrong.

---

## Known limits of what was built tonight

- **The tier ordinal is derived from the name, not from observed timing.** `Tier N` uses N;
  `Release N` uses N−1; Door is parked at 99 so it always sorts last. A label that does not
  state its position — "Final Release", "Last Release" — is deliberately left **unmapped**
  rather than guessed at, and its place comes from `window_open`. When real data lands,
  cross-check the name-derived ordinal against each tier's first-sale timestamp within the
  event — if a "Tier 2" opened before a "Tier 1", the name is lying and the timing wins.
- **`event_key` join is a string match** between the platform's event name and the finance
  workbook's. SPEC §3.1 warns these will be inconsistent. `validate()` reports both sides
  of the join failure (`join_health`), but no fuzzy matching is implemented and none should
  be added without looking at the real names first.
- **`price` in `event_tier` is the FIRST buyer price observed** in the tier — the price it
  opened at — not the modal realised price. `n_distinct_prices > 1` means the price moved
  while the tier was on sale; the report counts those tiers. Read that count before Phase 4:
  a mid-window correction is price variation at a fixed lead time (SPEC §4.3), which is the
  one thing that separates the price effect from the timing effect — but a refund or a
  mapping error looks identical, so each one needs eyes on it.
- **A comp is recognised by its WHOLE name** (`normalize.COMP_LABELS`) or by a price of 0,
  never by a word inside a longer name — a bare-word search deleted paid tiers like
  "Tier 1 + Free Drink" from demand. The cost is that a comp spelling nobody listed is
  missed by name; it is still caught by its price, and the report's comp-by-tier-name list
  is there to be read when the real labels land.
- **The panel is keyed on the raw tier NAME**, so two spellings of one tier inside one
  event ("Early Bird" and "EARLY BIRD") would be two rows. That is the safe direction —
  the alternative merged different prices — but check the `UNMAPPED LABELS` list and the
  `event_tiers_sharing_an_ordinal` count in the report when real labels land.
- **De-duplication assumes a re-pull repeats a file, not a row.** Rows an earlier export
  already contained are dropped; identical rows *within* one export are kept, because a
  per-ticket export repeats `order_id` when one order holds several tickets. If a platform
  turns out to emit a per-ticket id, map it and this gets simpler.
- **Cancelled events** are deferred (DECISIONS.md 2026-07-17). `events` carries a
  `cancelled` column and nothing reads it yet. The README survivorship paragraph is still
  owed by the pre-publication gate.
- **No statistics have been written.** No regression, no fixed effects, no Monte Carlo.

---

## Next session

1. **Freeze `preregistration.md`.** Still DRAFT; per SPEC §5.3 it must be frozen *before
   first data contact*, and `dataset.load_real()` will refuse until it is. Then copy the
   frozen list into `dataset.FEATURES` and delete the "provisional" comment.
2. Get the exports into `data/raw/`. Everything real is blocked on this.
3. `--sniff`, fill the four maps, resolve face-vs-fee, run ingest, read the report.
4. Fill SPEC §7's `[N]` events and `[M]`k transactions from the report's row counts.
5. Phase 1 descriptives — buildable **now**, against `--synthetic`, without waiting for
   any of the above.

**Gate check:** nothing here touches live money or the statistical core, so no Fable gate
is due. The next Fable gate is #2 (Phase 4 identification review), a long way off.

---

## 2026-08-17 — PARKED (note: sections above predate the 2026-08-12 code-complete session)

- Phases 0–6 + the prereg freeze are DONE (commits `7415c8c`, `1c7196d`): 402 tests green as of
  2026-08-12, machinery-level gate-2 review of phase 4 PASSED (Fable, in-session). The
  "Next session" list above is superseded except item 2 (exports).
- **PARKED indefinitely (Awande, 2026-08-17): platform exports are not coming for now.**
  `data/raw/{fixr,costs,other}` remain empty; nothing has run on real data.
- **First experimental on-sale did NOT go out w/e 15/16 Aug** (slipped). The frozen design stays
  valid; DESIGN.md §4 SOP governs whenever an autumn on-sale actually happens.
- CV: the Resume_2027 ticket-pricing entry is being rewritten to machinery-only truth (no panel
  counts, no elasticity/MC results until real data exists). Master ledger:
  `~/dev/quant/NEXT-SESSION.md`.

### Addendum 2026-08-18 — the suite is RED at HEAD
A verification sweep ran `uv run pytest` at `ea89ffa`: **392 passed / 10 FAILED**. All ten are
the "real data is refused while the preregistration is a draft" guard tests (test_phase1–6 +
test_synthetic). ~~Hypothesis, unconfirmed:~~ **RESOLVED 2026-08-18 — see the entry below.**
The hypothesis was right in outcome and wrong in mechanism. The freeze commit did not "flip
prereg out of DRAFT": before `1c7196d` the file was **untracked**, so the "402 green" run was
green against an untracked DRAFT working copy, and `1c7196d` created the tracked file already
FROZEN. **The guard was never dead** — it was the tests that were, because they read the
repo's ambient `preregistration.md` instead of injecting a fixture, so they asserted a fact
about the repo rather than a property of the guard. The real path stayed closed throughout at
the SECOND gate (frozen + no derived tables → `FileNotFoundError`, exit 1), which at that
point had zero test coverage. Fixed and green below; the test-count embargo is lifted.

---

## 2026-08-18 — guard tests repaired, both gates now covered (412 green)

**Root cause.** The ten failing guard tests sniffed the repo's ambient `preregistration.md`
rather than injecting their own, so they were measuring the repo's state, not the guard's
behaviour; commit `1c7196d` created that file already FROZEN (it was untracked before, which
is why the historical "402 green" run passed against a DRAFT working copy). The guard
implementation was correct the whole time — DRAFT refuses, MISSING refuses, FROZEN opens —
and the real-data path was never actually open: since the freeze it has been held shut by the
second gate, `data/derived/` not existing, which had **zero** coverage until now.

**What changed** (all uncommitted, for review):

1. `dataset.preregistration_status` / `require_frozen_preregistration` now take
   `path: Path | None = None` and resolve `PREREGISTRATION` **in the body**. The old default
   argument was evaluated at import, so monkeypatching the module constant did nothing to
   no-argument callers — including `load_real`, which takes no prereg parameter. Same
   statuses, same messages, no call-site changes.
2. The four `tests/test_synthetic.py` guard tests now **inject** a DRAFT fixture
   (`tests.synthetic.draft_preregistration`) instead of sniffing. No assertion was inverted
   and no guard test was deleted.
3. Three new integrity tests: the committed prereg **is** FROZEN; `--real` is **still**
   refused when frozen because no data has arrived (and writes nothing under `results/`
   outside `results/synthetic/`); and `data/raw/{fixr,costs,other}` are empty with
   `data/derived/` absent — "nothing has run on real data" as a checked fact rather than a
   claim in this file.
4. The six phase guard tests are renamed (the old names asserted a draft that no longer
   exists) and **parametrized over both gates**: DRAFT on an injected draft, FROZEN on the
   real file. Both arms assert exit 1, the expected reason on **stderr**, and not on stdout
   (the stderr-not-stdout discipline the 2026-08-12 review pinned).
5. Phases 1 and 2 no longer print "the prereg says DRAFT" into their generated reports —
   false since the freeze. The post-freeze truth: the list is frozen (SPEC §5.3, `1c7196d`)
   and the calendar patterns are untested because no real data exists (exports parked). Both
   tests now assert the report does **not** call the prereg a draft, and
   `results/synthetic/phase1_descriptives.md` / `phase2_demand_forecast.md` were regenerated
   (text-only diff; no figure changed).
6. Freeze-scheduled housekeeping done: the "THIS LIST IS PROVISIONAL" block above
   `dataset.FEATURES` and the phase 4 "when FROZEN, re-read this module" TODO are gone,
   `FEATURES` was reconciled against the frozen list (they agree; the three blank "Awande's
   additions" bullets contribute nothing), and a new test pins every slug to a bullet in the
   frozen file so the reconciliation cannot rot.

**Result: `uv run pytest` → 412 passed, 0 failed** (was 392 passed / 10 failed; +6 from
parametrizing the six phase tests, +4 new tests). The safety property is strictly stronger
than before: the guard is now tested at both gates instead of one, and the absence of real
data is asserted rather than asserted-about.

**⚠️ DELIBERATE TRIPWIRE — read before "fixing" a future red suite.** Eight assertions now
encode "no real data exists": the six phase FROZEN arms,
`test_real_is_still_refused_when_frozen_but_no_data_has_arrived`, and
`test_no_real_data_has_entered_the_repo`. The moment exports land and `pricing.ingest` runs,
they ALL fail — **by design**. First data contact is meant to be a suite-breaking event that
forces a human to consciously rewrite these tests for the with-data world (and to re-read the
pre-registration rules while doing it). Deleting or loosening them to get green is precisely
the failure mode this entry just repaired. Also note: any external claim quoting "412 tests"
should carry the caveat that ~8 of them pin the absence of data and will change when it
arrives. (A stray `.DS_Store` in `data/raw/*` also trips the strict empty-dir assertion —
that strictness is intentional; the assertion message names the offending file.)
