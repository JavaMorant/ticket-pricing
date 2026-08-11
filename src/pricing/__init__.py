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
"""

__version__ = "0.1.0"
