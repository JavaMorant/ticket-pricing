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
- **Survivorship:** 1–5 cancelled/downscaled events over five years, and their presale histories are **destroyed on-platform**. Memory-only reconstruction (name, date, planned capacity, why pulled, rough presales at pull), rows flagged `cancelled`, excluded from curve fits, one honest README paragraph on the residual bias.

## Learning protocol (binding)

- **Awande hand-writes the statistical core:** the naïve regression, event fixed effects, IV/2SLS, and the Monte Carlo. Claude (Opus) tutors and writes plumbing only (ingest, normalisation, plotting, tests).
- **Phase vivas:** every phase ends with an interview-style grilling built from SPEC.md §10; failing means revise, not advance. Vivas log to the quant-prep streak.

## Risk & privacy

- **Public GitHub repo, pseudonymised.** Raw data gitignored forever; buyer PII stripped at ingest (non-negotiable). Artists → billing-tier labels, venues → numbered, £ figures banded/scaled in anything rendered; findings reported as percentages and distributions. Real absolutes stay local and are quoted verbally in interviews.
- Privacy guard (gitignore + no-raw-data check + PII strip) in place **from commit one**, before any data enters the repo.

## Sequencing

1. **This week, before any data is pulled:** Awande writes `preregistration.md` from memory — the §5.4 pattern list, edited and frozen — and the cancelled-events memory list. Pre-registration precedes first data contact, per §5.3.
2. **Phase 0 immediately after** (full steam; the August diet does not block this project).
3. **Hard deadline ~10 Aug 2026:** autumn randomised tier-pricing experiment (§9) designed and frozen **before first on-sale (~mid-Aug)**. ~8+ autumn events across both brands; event-level randomisation, blocked on artist tier × term-week. Arm price gaps, event selection, and max acceptable revenue downside per arm are decided in that dedicated design session — they are open items, not defaults.
4. Phases 1–6 in SPEC.md order; Phase 7 last.

## Ownership & upkeep

- Post-event ritual: update the brand Sheet + pull the export → run ingest → tables rebuild. The auto-report (Phase 7) makes this self-rewarding so the dataset never rots.

## Carried assumptions

- Stack: Python via `uv`; one notebook per phase + `src/` modules; `pandas`, `statsmodels`/`linearmodels`; `pytest` on ingest; no DB unless the data demands it.
- Model policy: Opus runs build sessions; Fable only at review gates (final whole-branch review, and the experiment design consult if wanted).
- Public remote created at repo birth; privacy guard precedes data.

## Open items (deferred, with owners)

| Item | Decided in | By |
|---|---|---|
| Experiment arms, price gaps, event list, downside cap | Experiment design session | ~10 Aug hard stop |
| Per-platform fee treatment (face vs face+fee) | Phase 0, from the exports | during ingest |
| Which cost shifters pass the exclusion restriction | Phase 4 | during modelling |
| Youni/TBC schema quirks | Phase 0 first pull | during ingest |
