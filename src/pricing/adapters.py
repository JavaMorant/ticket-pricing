"""Adapters: read raw platform exports and rename their columns to canonical ones.

ONE adapter function per platform. When the real exports land, the only thing that
changes is the COLUMN_MAP dict directly above each adapter. Nothing else in the repo
knows what a Fixr column is called.

Until a map is filled in, its adapter raises NotImplementedError with the exact list of
canonical fields still missing. Run `--sniff` first: it prints the real column names
(and nothing else) so the map can be filled in from the printout.

Raw file layout (files have not landed yet, 2026-08-11):
    data/raw/fixr/*.csv|xlsx    per-ticket exports, ~150 events
    data/raw/other/*            TBC.xyz + Youni, ~15 events
    data/raw/costs/*.xlsx       two brand master finance workbooks, 5 years
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from pricing.normalize import parse_number, parse_uk_datetime, pii_columns, strip_pii

# --------------------------------------------------------------------------------------
# Canonical schemas — the vocabulary the rest of the repo speaks
# --------------------------------------------------------------------------------------

# One row per ticket. `platform` is added by the adapter, not mapped from the file.
TRANSACTION_FIELDS = [
    "order_id",
    "event_key",  # whatever the platform calls the event; joined to events by name later
    "tier_name",
    "price_paid",
    "booking_fee",
    "purchased_at",
    "quantity",
    "promo_code",
]
# Fields an adapter must produce or the transactions table cannot be built.
TRANSACTION_REQUIRED = ["order_id", "event_key", "tier_name", "price_paid", "purchased_at"]

# One row per event, from the brand finance workbooks (SPEC.md 3.2).
COST_FIELDS = [
    "event_key",
    "event_name",
    "event_date",
    "venue",
    "city",
    "brand",
    "capacity",
    "artist_guarantee",
    "deal_structure",
    "venue_cost",
    "security_cost",
    "production_cost",
    "staffing_cost",
    "marketing_spend",
    "cancelled",
]
COST_REQUIRED = ["event_key", "event_date", "brand", "capacity"]

# --------------------------------------------------------------------------------------
# Face value vs booking fee, per platform (SPEC.md 8.6)
# --------------------------------------------------------------------------------------
#
# Elasticity acts on the price the BUYER SEES. If the platform's price column is face
# value and the fee is charged on top, buyer_price = price_paid + booking_fee. If the
# price column is already inclusive, adding the fee double-counts it.
#
# We do not know which, per platform, until a real export is compared against a real
# order confirmation. Until then every platform stays "UNKNOWN", validate.py shouts
# about it, and downstream numbers are provisional. Do not guess this one.
#
# Allowed values: "UNKNOWN" | "face_plus_fee" | "inclusive"
FEE_TREATMENT: dict[str, str] = {
    "fixr": "UNKNOWN",  # TODO(real-data): confirm against one Fixr order confirmation.
    "tbc": "UNKNOWN",  # TODO(real-data): confirm against one TBC.xyz order confirmation.
    "youni": "UNKNOWN",  # TODO(real-data): confirm against one Youni order confirmation.
}

# --------------------------------------------------------------------------------------
# COLUMN MAPS — the only thing that changes when the real exports arrive
# --------------------------------------------------------------------------------------
# Each map is {raw column name in the export: canonical field name}.
# Fill from a `--sniff` run. Leave a canonical field out if the export genuinely lacks it.

# TODO(real-data): fill from `uv run python -m pricing.ingest --sniff` once
# data/raw/fixr/*.csv|xlsx exists. ~150 events, per-ticket rows.
FIXR_COLUMN_MAP: dict[str, str] = {}

# TODO(real-data): fill from a sniff of the TBC.xyz export in data/raw/other/. ~15 events
# shared with Youni; check whether the two platforms need separate maps or one.
TBC_COLUMN_MAP: dict[str, str] = {}

# TODO(real-data): fill from a sniff of the Youni export in data/raw/other/.
YOUNI_COLUMN_MAP: dict[str, str] = {}

# TODO(real-data): fill from a sniff of the two brand master workbooks in data/raw/costs/.
# Watch for: merged header rows, one sheet per year, and totals rows that are not events.
COSTS_COLUMN_MAP: dict[str, str] = {}

# TODO(real-data): the finance workbooks are multi-sheet. Name the sheet to read, or
# leave None to read the first sheet.
COSTS_SHEET_NAME: str | None = None


# --------------------------------------------------------------------------------------
# Reading raw files
# --------------------------------------------------------------------------------------

RAW_SUFFIXES = {".csv", ".xlsx", ".xls"}


def read_raw_file(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Read one raw export into a DataFrame, no interpretation. CSV or Excel."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet_name or 0)


def list_raw_files(directory: Path) -> list[Path]:
    """Every readable export under `directory`, sorted. Ignores dotfiles and lock files."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in RAW_SUFFIXES and not p.name.startswith((".", "~$"))
    )


def _apply_map(
    raw: pd.DataFrame,
    column_map: dict[str, str],
    required: list[str],
    platform: str,
) -> pd.DataFrame:
    """Rename raw columns to canonical ones and check the required fields arrived."""
    if not column_map:
        raise NotImplementedError(
            f"{platform}: COLUMN_MAP is empty — the real export has not been mapped yet.\n"
            f"  1. run: uv run python -m pricing.ingest --sniff\n"
            f"  2. fill {platform.upper()}_COLUMN_MAP in src/pricing/adapters.py\n"
            f"  required canonical fields: {required}"
        )

    # Two raw columns mapped to one canonical field: pandas renames both, then the column
    # selection below keeps whichever came last and the other silently disappears. That is
    # how "Price" and "Total Paid" both mapped to price_paid would quietly pick one.
    doubled = sorted(field for field, n in Counter(column_map.values()).items() if n > 1)
    if doubled:
        raise ValueError(
            f"{platform}: COLUMN_MAP sends more than one raw column to {doubled}. "
            f"Renaming would keep only one of them, without saying which. "
            f"Map exactly one raw column per canonical field in "
            f"{platform.upper()}_COLUMN_MAP."
        )

    missing_raw = [col for col in column_map if col not in raw.columns]
    if missing_raw:
        raise ValueError(
            f"{platform}: columns named in the map are absent from the file: {missing_raw}. "
            f"Re-run --sniff; the export format has changed."
        )

    out = raw.rename(columns=column_map)
    out = out[[c for c in out.columns if c in set(column_map.values())]]
    out = strip_pii(out)  # belt-and-braces: a mapped-through PII column dies here too

    missing = [f for f in required if f not in out.columns]
    if missing:
        raise NotImplementedError(
            f"{platform}: COLUMN_MAP does not yet produce required fields {missing}. "
            f"Add them to {platform.upper()}_COLUMN_MAP in src/pricing/adapters.py."
        )
    return out


# --------------------------------------------------------------------------------------
# One adapter per platform
# --------------------------------------------------------------------------------------


def adapt_fixr(raw: pd.DataFrame) -> pd.DataFrame:
    """Fixr per-ticket export -> canonical transactions columns."""
    out = _apply_map(raw, FIXR_COLUMN_MAP, TRANSACTION_REQUIRED, "fixr")
    out["platform"] = "fixr"
    return out


def adapt_tbc(raw: pd.DataFrame) -> pd.DataFrame:
    """TBC.xyz per-ticket export -> canonical transactions columns."""
    out = _apply_map(raw, TBC_COLUMN_MAP, TRANSACTION_REQUIRED, "tbc")
    out["platform"] = "tbc"
    return out


def adapt_youni(raw: pd.DataFrame) -> pd.DataFrame:
    """Youni per-ticket export -> canonical transactions columns."""
    out = _apply_map(raw, YOUNI_COLUMN_MAP, TRANSACTION_REQUIRED, "youni")
    out["platform"] = "youni"
    return out


def adapt_costs(raw: pd.DataFrame) -> pd.DataFrame:
    """Brand master finance workbook -> canonical per-event cost columns."""
    return _apply_map(raw, COSTS_COLUMN_MAP, COST_REQUIRED, "costs")


# Which adapter handles a file, decided by the folder it landed in. `other/` holds both
# TBC.xyz and Youni, so it is disambiguated by filename.
def adapter_for(path: Path) -> tuple[str, object]:
    """Return (platform, adapter function) for a raw file path."""
    parts = {p.lower() for p in Path(path).parts}
    name = Path(path).name.lower()
    if "fixr" in parts or "fixr" in name:
        return "fixr", adapt_fixr
    if "costs" in parts:
        return "costs", adapt_costs
    if "youni" in name:
        return "youni", adapt_youni
    if "tbc" in name:
        return "tbc", adapt_tbc
    # TODO(real-data): if the `other/` filenames do not say which platform they are,
    # rename them on arrival rather than adding guessing logic here.
    return "unknown", None


# --------------------------------------------------------------------------------------
# --sniff : anonymised schema summary
# --------------------------------------------------------------------------------------
#
# Prints the SHAPE of whatever landed in data/raw/ and nothing else: column names,
# dtypes, null rates, and the date span of genuinely date-like columns. No cell values,
# no head(), no uniques, and no file paths — the filenames carry real brand and event
# names (DECISIONS.md: artists become billing tiers, venues become numbers, in anything
# that leaves the machine).
#
# Even so, treat the output as LOCAL USE ONLY: column names alone can name a brand or an
# artist. Read it, fill the maps, do not paste it anywhere public.


def _date_range(series: pd.Series) -> tuple[str, str] | None:
    """Min/max DATE of a column that really parses as dates, else None.

    Number columns are skipped outright — including money written as text, "£1,500" and
    "1,200". pd.to_datetime reads a number as epoch nanoseconds, so an artist-guarantee or
    price column would "parse" happily and then print its real minimum and maximum in what
    is supposed to be a values-free summary. Dates only, to day precision.
    """
    if parse_number(series).notna().sum() >= max(1, int(0.5 * len(series))):
        return None
    parsed = parse_uk_datetime(series)
    if parsed.notna().sum() < max(1, int(0.5 * len(series))):
        return None
    return str(parsed.min().date()), str(parsed.max().date())


def sniff_file(path: Path) -> dict:
    """Anonymised schema summary of one raw file. Never returns cell values."""
    path = Path(path)
    folder = path.parent.name
    try:
        raw = read_raw_file(path)
    except Exception as exc:  # noqa: BLE001 - a locked or corrupt export must not stop the sweep
        # The exception TYPE only: pandas parse errors quote the offending file content.
        return {"folder": folder, "error": type(exc).__name__}

    platform, _ = adapter_for(path)
    suspected = set(pii_columns(raw))
    columns = []
    for col in raw.columns:
        entry = {
            "name": str(col),
            "dtype": str(raw[col].dtype),
            "null_rate": round(float(raw[col].isna().mean()), 4),
        }
        # No range for a column we already believe is buyer PII — a date-of-birth span is
        # exactly the thing this summary must not print.
        if col not in suspected:
            span = _date_range(raw[col])
            if span:
                entry["date_range"] = span
        columns.append(entry)

    return {
        "folder": folder,
        "platform": platform,
        "rows": len(raw),
        "columns": columns,
        "suspected_pii_columns": sorted(str(c) for c in suspected),
    }


def sniff_raw(raw_dir: Path) -> list[dict]:
    """Schema summary for every raw export under `raw_dir`, labelled folder/file N of M."""
    paths = list_raw_files(raw_dir)
    totals = Counter(p.parent.name for p in paths)
    seen: Counter = Counter()

    summaries = []
    for path in paths:
        folder = path.parent.name
        seen[folder] += 1
        summary = sniff_file(path)
        summary["label"] = f"{folder}/file {seen[folder]} of {totals[folder]}"
        summaries.append(summary)
    return summaries


def format_sniff(summaries: list[dict]) -> str:
    """Render sniff output as text: column names, dtypes, null rates, date spans."""
    if not summaries:
        return (
            "No raw exports found.\n"
            "Expected: data/raw/fixr/*.csv|xlsx, data/raw/other/*, data/raw/costs/*.xlsx\n"
            "(Real exports have not landed yet as of 2026-08-11.)"
        )

    lines = [
        "SCHEMA SNIFF — column names and dtypes only, no row values and no filenames.",
        "Column names may still name a brand or an artist: local use only.",
        "",
    ]
    for s in summaries:
        lines.append(f"--- {s['label']}")
        if "error" in s:
            lines.append(f"    UNREADABLE: {s['error']}")
            lines.append("")
            continue
        lines.append(f"    platform: {s['platform']}   rows: {s['rows']}")
        if s["suspected_pii_columns"]:
            lines.append(f"    PII columns (dropped at ingest): {s['suspected_pii_columns']}")
        for col in s["columns"]:
            span = (
                f"  range={col['date_range'][0]} .. {col['date_range'][1]}"
                if "date_range" in col
                else ""
            )
            lines.append(
                f"    {col['name']!r:<40} {col['dtype']:<16} null={col['null_rate']:.1%}{span}"
            )
        lines.append("")

    lines.append("Next: paste the column names into the COLUMN_MAPs in src/pricing/adapters.py.")
    return "\n".join(lines)
