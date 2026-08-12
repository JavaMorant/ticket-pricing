"""The guard that keeps real data out for now — and the pre-registered feature list.

Two sources, and this file owns the closed one:

    synthetic   the generator in synthetic.py, seeded and deterministic — loaded by
                `phaseio.resolve`, which needs its ground truth as well as its tables
    real        `load_real()`: the parquet tables in data/derived/ — REFUSED while
                preregistration.md is still marked DRAFT

Why the guard exists (SPEC.md 5.3, DECISIONS.md sequencing): the pre-registered pattern
list must be frozen BEFORE first data contact. If a phase module could read data/derived/
today, the honest ordering would be broken the first time someone ran it out of
curiosity — and once you have seen the data you cannot un-see it when you write the list.
So the real path is fully wired and fully closed, and it prints the reason rather than
failing silently or half-working.

The guard is deliberately dumb: it reads one line of a markdown file. It is a tripwire,
not a security control. Removing it is a two-word edit to preregistration.md, which is
exactly the point — the freeze is a human act that this file only records.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVED_DIR = REPO_ROOT / "data" / "derived"
PREREGISTRATION = REPO_ROOT / "preregistration.md"

TABLE_NAMES = ("events", "transactions", "event_tier")


# --------------------------------------------------------------------------------------
# Pre-registered features (SPEC.md 5.4)
# --------------------------------------------------------------------------------------
#
# THIS LIST IS PROVISIONAL. Its final contents come from the FROZEN preregistration.md,
# not from here — this constant exists so the phase modules have something to import
# today, and so that the moment the file is frozen there is exactly one place to copy it
# into. When that happens, replace this list with the frozen one and delete this comment.
#
# The names are feature *slugs*, one per bullet in preregistration.md. Several of them
# collapse into one tested effect on purpose: freshers' week, semester start, the student
# loan instalment and good weather all arrive together every year and CANNOT be separated
# with ~120 events (SPEC.md 5.6). They are grouped under `start_of_semester` and reported
# as one effect, undecomposed.
FEATURES: list[str] = [
    # academic calendar
    "start_of_semester",  # grouped: freshers + semester start + loan drop + weather
    "term_phase",  # early / mid / late within a term
    "exam_period",
    "reading_week",
    "end_of_term",
    "vacation_vs_term",
    "ball_season",
    # timing
    "day_of_week",
    "lead_time_days",
    "days_since_last_event",
    "events_in_trailing_14d",
    # event characteristics
    "artist_billing_tier",
    "venue",
    "city",
    "capacity",
    "brand",
    # environment / trend
    "academic_year",  # cohort AND brand-growth control (SPEC.md 8.9 — control it first)
]

# Named here so no phase forgets: these two are on the SPEC 5.4 list but cannot be built
# from the ticketing data alone. They need a hand-assembled file that does not exist yet.
FEATURES_NEEDING_EXTERNAL_DATA: list[str] = [
    "competing_promoter_nights",  # from Awande's memory; nobody else can reconstruct them
    "weather",  # outdoor-sensitive events only
]


# --------------------------------------------------------------------------------------
# The freeze guard
# --------------------------------------------------------------------------------------


def preregistration_status(path: Path = PREREGISTRATION) -> str:
    """Return "DRAFT", "FROZEN" or "MISSING", read off the file's Status line.

    Looks for the first line containing "status" (case-insensitive) and asks whether the
    word DRAFT appears in it. Anything else is treated as frozen; a missing file is
    treated as MISSING, which is also refused.
    """
    if not Path(path).exists():
        return "MISSING"
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "status" in line.lower():
            return "DRAFT" if "draft" in line.lower() else "FROZEN"
    return "DRAFT"  # no status line at all -> assume not frozen, refuse


def require_frozen_preregistration(path: Path = PREREGISTRATION) -> None:
    """Raise unless preregistration.md is frozen. The only gate on the real-data path."""
    status = preregistration_status(path)
    if status == "FROZEN":
        return
    raise RuntimeError(
        f"real data is OFF LIMITS: preregistration.md is {status}.\n"
        f"  file: {path}\n"
        f"  why : SPEC.md 5.3 — the pattern list is written from operating memory BEFORE\n"
        f"        first data contact. Testing patterns you picked after seeing the data is\n"
        f"        a circle, not a finding (SPEC.md 5.2).\n"
        f"  fix : edit the Status line in preregistration.md to say FROZEN once the list\n"
        f"        is final, then re-run.\n"
        f"  now : run with --synthetic instead; every phase works end to end on the\n"
        f"        seeded fixture in src/pricing/synthetic.py."
    )


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_real(derived_dir: Path = DERIVED_DIR) -> tuple[pd.DataFrame, ...]:
    """The three derived tables from data/derived/. Refuses while the prereg is DRAFT."""
    require_frozen_preregistration()
    derived_dir = Path(derived_dir)
    missing = [n for n in TABLE_NAMES if not (derived_dir / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"no derived tables for {missing} in {derived_dir}. "
            f"Run: uv run python -m pricing.ingest"
        )
    return tuple(pd.read_parquet(derived_dir / f"{n}.parquet") for n in TABLE_NAMES)


# The synthetic side of "where does a phase get its data" lives in `phaseio.resolve`,
# which calls `synthetic.generate` directly because it also needs the ground truth. There
# is deliberately no second loader here: one function per source, one caller.
