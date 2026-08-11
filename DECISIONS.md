# Decision Spec — ticket-pricing

**Grilled:** 2026-07-13 (log: `~/Documents/00-inbox/grill-ticket-pricing-2026-07-13.md`)
**Input to:** Plan session (Opus). Do not build from this file without a plan.
**Companion:** `SPEC.md` — the research design itself. This file records only what the grilling added or locked; where SPEC.md already decides something (build order, identification strategy, traps, DoD checklist), it stands unamended.

---

## Goal & success

- **No hard application deadline.** Interviews Oct–Dec 2026; the repo must exist and be *defensible* before the first interview. Defensibility, not polish, is the bar.
- **Deep understanding is a first-class deliverable**, equal to the repo. Success includes Awande answering every §10 interview question cold.
- Definition of done = SPEC.md §7 checklist, unamended.

## Scope & non-goals

- **In scope now:** scripted ingest — the two brand master Google Sheets (Brand A, Brand B) + platform exports drop into `data/raw/` (gitignored); a script validates and rebuilds `events` / `transactions` / `event_tier`. Brand is a column; per-brand finance views are derived, never separate pipelines.
- **Phase 7 (one weekend, after Phases 0–6):** Streamlit tool **plus** the post-event auto-report (P&L, sales curve vs forecast, retro P(clear)) — the "upload a sheet after every event, get a report" wish lands here, not earlier.
- **Explicit non-goals:** an expense-tracking application, upload portal, or finance UI. The Google Sheets remain the system of record for money; the repo reads them and never replaces them.

## Data

- **~165 events total: ~150 Fixr + ~15 TBC.xyz/Youni.** All platforms give full per-ticket transaction detail → include everything in every phase; add a `platform` column; verify face-value-vs-fee treatment **per platform** (§8.6).
- **Cost side:** one master Google Sheet per brand already exists — Phase 0 cost work is consolidation + validation, not archaeology.
- **Survivorship:** 1–5 cancelled/downscaled events over five years; presale histories **destroyed on-platform**. **Deferred (2026-07-17):** v1 dataset proceeds without them; ingest gains a `cancelled`-events config hook later. Non-deferrable remainder: the README survivorship paragraph (dataset excludes cancelled events; models are conditioned on events that ran) — due by the pre-publication Fable gate — and the §10 interview answer.

## Learning protocol (binding)

- **Awande hand-writes the statistical core:** the naïve regression, event fixed effects, IV/2SLS, and the Monte Carlo. Claude (Opus) tutors and writes plumbing only (ingest, normalisation, plotting, tests).
- **Phase vivas:** every phase ends with an interview-style grilling built from SPEC.md §10; failing means revise, not advance. Vivas log to the quant-prep streak.

## Risk & privacy

- **Public GitHub repo, pseudonymised.** Raw data gitignored forever; buyer PII stripped at ingest (non-negotiable). Artists → billing-tier labels, venues → numbered, £ figures banded/scaled in anything rendered; findings reported as percentages and distributions. Real absolutes stay local and are quoted verbally in interviews.
- Privacy guard (gitignore + no-raw-data check + PII strip) in place **from commit one**, before any data enters the repo.

## Sequencing

1. **This week, before any data is pulled:** Awande writes `preregistration.md` from memory — the §5.4 pattern list, edited and frozen. Pre-registration precedes first data contact, per §5.3. (Cancelled-events reconstruction deferred — see Survivorship.)
2. **Phase 0 immediately after** (full steam; the August diet does not block this project).
3. **Hard deadline ~10 Aug 2026:** autumn randomised tier-pricing experiment (§9) designed and frozen **before first on-sale (~mid-Aug)**. ~8+ autumn events across both brands; event-level randomisation, blocked on artist tier × term-week. Arm price gaps, event selection, and max acceptable revenue downside per arm are decided in that dedicated design session — they are open items, not defaults.
4. Phases 1–6 in SPEC.md order; Phase 7 last.

## Ownership & upkeep

- Post-event ritual: update the brand Sheet + pull the export → run ingest → tables rebuild. The auto-report (Phase 7) makes this self-rewarding so the dataset never rots.

## Carried assumptions

- Stack: Python via `uv`; one notebook per phase + `src/` modules; `pandas`, `statsmodels`/`linearmodels`; `pytest` on ingest; no DB unless the data demands it.
- Model policy: Opus runs build sessions and routine vivas; Fable only at four gates — (1) experiment design audit before ~10 Aug (live money), (2) Phase 4 identification review before the CV bullet is written, (3) pre-publication whole-repo + privacy review, (4) one mock defense in Oct before interviews. Plus ad hoc: any real booking decision that rides on model output.
- Public remote created at repo birth; privacy guard precedes data.

## Amendments

- **2026-08-11 (Awande, batched decision):** Learning protocol amended for the observational
  build. Opus agents write the full implementation including the statistical core, under a hard
  simplicity constraint (OLS + event FE via statsmodels, plain bootstrap Monte Carlo, grouped
  splits; no gradient boosters; 2SLS only if a cost shifter passes exclusion cleanly). In
  exchange, every phase ships a plain-English EXPLAINER.md + viva question sheet, and **no
  interview quotes any number until Awande passes the vivas** (gate 4 mock defense stands).
  Phase-4 identification review (Fable gate 2) runs in-session before the CV bullet is filled.
  Reports-as-scripts replace the one-notebook-per-phase assumption, matching the stat-arb/fraud
  reproducibility pattern (`uv run python -m ...` regenerates every number).

## Open items (deferred, with owners)

| Item | Decided in | By |
|---|---|---|
| ~~Experiment arms, price gaps, event list, downside cap~~ **DONE 2026-08-10** — design frozen in `experiment/DESIGN.md` (±15% arms, resident nights only, headliners excluded, permuted-pair assignment, no-abort risk position). Fable design-audit gate (1 of 4) satisfied in the 2026-08-09/10 grilling session; log: filed locally (gitignored) | Experiment design session | ~10 Aug hard stop |
| Per-platform fee treatment (face vs face+fee) | Phase 0, from the exports | during ingest |
| Which cost shifters pass the exclusion restriction | Phase 4 | during modelling |
| Youni/TBC schema quirks | Phase 0 first pull | during ingest |
| Cancelled-events list + `cancelled` config hook in ingest | deferred by user 2026-07-17 | README disclosure by pre-publication gate; reconstruction whenever |
