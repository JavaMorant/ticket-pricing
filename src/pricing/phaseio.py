"""Shared plumbing for the phase runners: data in, report out.

Every phase module (`phase1` .. `phase6`) is a script you can run:

    uv run python -m pricing.phase4 --synthetic
    uv run python -m pricing.phase4 --real        <- refused while the prereg is DRAFT

All six need the same five things, so those five live here rather than six times:

  1. the same CLI flags,
  2. the same two data sources, with the same refusal on the real one,
  3. the same rule about WHERE a report is allowed to be written — synthetic output to
     `results/synthetic/` (committable, nothing in it is real), real output to `results/`
     (gitignored until the pre-publication gate in DECISIONS.md),
  4. one report writer with one signature, and one report banner, so "where does this
     report get written, and what does it say it is?" has exactly one answer,
  5. the academic-year label, because Phase 1's descriptives, Phase 4's instrument
     controls and Phase 5's demand features all need it and they must be the same
     function or the reports quote different years for the same event.

Nothing here does any statistics and nothing here draws: markdown tables and figures are
in `report.py`, which is about how a report READS. A reader chasing a number should find
it in the phase module, not in a helper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from pricing import dataset

REPO_ROOT = dataset.REPO_ROOT
REAL_RESULTS_DIR = REPO_ROOT / "results"
SYNTHETIC_RESULTS_DIR = REAL_RESULTS_DIR / "synthetic"

# Kept in step with synthetic.DEFAULT_SEED without importing the generator at module
# import time (the real path must not need the generator to exist).
DEFAULT_SEED = 20260811


class Run(NamedTuple):
    """One phase run: its data, where its output goes, and which source it came from."""

    source: str  # "synthetic" or "real"
    events: pd.DataFrame
    transactions: pd.DataFrame
    event_tier: pd.DataFrame
    truth: dict | None  # ground truth — synthetic only; None on real data
    out_dir: Path
    seed: int | None


def phase_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """The flags every phase runner takes. Identical on purpose.

    `--seed` defaults to a real number rather than None so that every report can print the
    seed it ran on: "regenerates every number" (DECISIONS.md) is not true of a report that
    does not say which fixture it was looking at.
    """
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--synthetic", action="store_true", help="run on the seeded fixture")
    parser.add_argument(
        "--real", action="store_true", help="run on data/derived/ (refused while prereg is DRAFT)"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="synthetic seed")
    return parser


def resolve(args: argparse.Namespace) -> Run:
    """Load the data for this run, or raise the refusal the real path is supposed to raise.

    `--real` goes through `dataset.load_real`, which reads the Status line of
    preregistration.md and refuses while it says DRAFT. The refusal is not caught here:
    each phase's `main()` catches it, prints it to stderr and exits 1, so the reason
    reaches the terminal instead of a traceback.

    Neither branch touches `data/raw/`. Only `ingest.py` does.
    """
    if args.real:
        events, transactions, event_tier = dataset.load_real()
        return Run("real", events, transactions, event_tier, None, REAL_RESULTS_DIR, None)

    from pricing import synthetic

    seed = getattr(args, "seed", DEFAULT_SEED)
    data = synthetic.generate(seed=seed)
    return Run(
        "synthetic",
        data.events,
        data.transactions,
        data.event_tier,
        data.truth,
        SYNTHETIC_RESULTS_DIR,
        seed,
    )


def run_phase(build, prog: str, description: str, argv: list[str] | None = None) -> int:
    """The whole `main()` for Phases 1-3: parse, load (or refuse), build, print paths.

    `build(run)` does the phase's actual work and returns the paths it wrote. Phases 4-6
    have their own `main()` because they print a terminal summary and take `--no-write`.
    """
    args = phase_parser(prog, description).parse_args(argv)
    try:
        run = resolve(args)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    paths = build(run)
    print()
    for path in paths:
        print(f"wrote {path}")
    return 0


def write_report(text: str | list[str], filename: str, out_dir: Path) -> Path:
    """Write a phase report as markdown. Returns the absolute path.

    Takes either the whole report as one string (Phases 4-6) or as a list of lines
    (Phases 1-3, which assemble tables into a list). One signature, one call site style.

    The output directory is decided by `resolve()`, never by the caller's shell: a real
    run must not be able to drop real prices and real margins into the committable
    `results/synthetic/` tree by being started from the wrong folder.
    """
    if isinstance(text, list):
        text = "\n".join(text) + "\n"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(text, encoding="utf-8")
    return path.resolve()


def academic_year(dates: pd.Series) -> pd.Series:
    """Academic-year label for each date: "2023/24" for anything from Sept 2023 to Aug 2024.

    September is the cut because that is when the student population turns over, and
    because the academic year — not the calendar year — is the spine of a student-events
    business: a September event and the following June event are one cohort, one marketing
    cycle, one brand-maturity level. Splitting them at 31 December would cut every season
    in half.

    It is also the SPEC.md 8.9 trend control — both brands grew year on year, so anything
    correlated with time inherits that growth, and the year label is what takes it back
    out. There is one of these functions for the whole repo, so two reports can never
    quote different years for the same event.
    """
    dates = pd.to_datetime(dates)
    start = dates.dt.year.where(dates.dt.month >= 9, dates.dt.year - 1)
    # The second half is the NEXT year's last two digits, so the modulo comes after the
    # +1: 2023 -> "24", and 2099 -> "00" rather than "100".
    return (
        start.astype("Int64").astype(str)
        + "/"
        + ((start + 1) % 100).astype("Int64").astype(str).str.zfill(2)
    )


def regenerate_command(module: str, source: str, seed: int | None) -> str:
    """The exact command that reproduces a report, built from the run that wrote it.

    Every report prints this in its own header, so "reports are scripts" (DECISIONS.md) is
    checkable rather than claimed: copy the line, run it, get the file back.
    """
    return f"uv run python -m pricing.{module} --{source}" + (
        f" --seed {seed}" if seed is not None else ""
    )


def provenance_header(title: str, source: str, seed: int | None, regenerate: str) -> str:
    """The banner every phase report opens with: what ran, on what, and how to redo it.

    One banner for all six phases. The seed is always printed on a synthetic run — a
    report that regenerates every number has to say which fixture produced them.
    """
    if source == "synthetic":
        stamp = (
            "**SYNTHETIC DATA — nothing here is real (Brand A/B, venues V1-V4, FAKE ids).**"
            f"\n\nSource: `synthetic`, seed `{seed}`"
        )
    else:
        stamp = (
            "**REAL DATA — stays local until the pre-publication review gate (DECISIONS.md).**"
            "\n\nSource: `data/derived/`. Every price-side number here is provisional while "
            "`fee_treatment` is UNKNOWN (SPEC.md 8.6)."
        )
    return f"# {title}\n\n{stamp}\nRegenerate: `{regenerate}`\n"
