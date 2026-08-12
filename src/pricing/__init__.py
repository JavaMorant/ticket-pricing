"""Ticket pricing & demand model — Phase 0 ingest plumbing.

Design rule (DECISIONS.md, 2026-08-11 amendment): simplicity is a requirement.
Plain functions over plain pandas DataFrames. No classes, no config framework,
no clever abstractions. Every line must be explainable out loud.

Module map:
    adapters.py   read raw platform exports, rename their columns to canonical ones
    normalize.py  tier names -> ordinal, comp/guestlist flags, buyer-PII strip
    tables.py     build the three derived tables (SPEC.md 3.3)
    validate.py   validation report over the built tables
    ingest.py     entry point: uv run python -m pricing.ingest [--sniff]
    dataset.py    the guard that keeps real data out while preregistration.md says DRAFT,
                  plus the pre-registered FEATURES list
    synthetic.py  the seeded fake programme whose true elasticity we chose, so every
                  later phase can be tested rather than believed
                  uv run python -m pricing.synthetic --synthetic [--write]

    phaseio.py    the plumbing ALL SIX phases share: the CLI, the two data sources and the
                  refusal on the real one, where output is allowed to be written, the
                  report writer, the report banner, the academic-year label. No statistics.
    report.py     how a report READS, for phases 1-3 only: markdown tables and the figure
                  house style. Phases 4-6 draw nothing and never import it.
    phase1.py     descriptives (SPEC.md 2). Also owns event_outcomes(), the per-event fact
                  table phases 2-3 import, so "tickets sold" means one thing everywhere.
                  uv run python -m pricing.phase1 --synthetic
    phase2.py     demand forecast (SPEC.md 6.1): OLS on log tickets, grouped K-fold by
                  event + final-year holdout, look-ahead check.
                  uv run python -m pricing.phase2 --synthetic
    phase3.py     sales curve (SPEC.md 6.2): empirical cumulative-share curves, forecast
                  from partial presale at day -7 / -3 / -1.
                  uv run python -m pricing.phase3 --synthetic

    phase4.py     elasticity (SPEC.md 4) — THE project. Naive pooled OLS, event fixed
                  effects, the lead-time confound check with an IDENTIFIED /
                  NOT-IDENTIFIED verdict from numeric criteria, and a cost-shifter IV.
                  uv run python -m pricing.phase4 --synthetic
    phase5.py     risk (SPEC.md 6.3): contribution, break-even, P(clear) by bootstrap over
                  Phase 2's out-of-fold residuals, and the three deal structures.
                  uv run python -m pricing.phase5 --synthetic
    phase6.py     counterfactual (SPEC.md 6.4): a visible grid over ladder level and
                  spread. Reports an uplift CURVE, not a number, when Phase 4 says the
                  elasticity is not identified.
                  uv run python -m pricing.phase6 --synthetic
"""

__version__ = "0.1.0"
