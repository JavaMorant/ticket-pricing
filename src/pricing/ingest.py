"""Entry point.

    uv run python -m pricing.ingest --sniff    print an anonymised schema summary of
                                              whatever is in data/raw/ (no row values)
    uv run python -m pricing.ingest            read data/raw/, build the three derived
                                              tables into data/derived/, print the
                                              validation report and write it to
                                              data/derived/validation_report.txt

The full run needs the COLUMN_MAPs in adapters.py to be filled in. Until the real
exports land they are empty and the run stops with instructions, on purpose.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pricing import adapters, tables, validate

# Anchored to the repo, not the shell's working directory (see tables.REPO_ROOT).
RAW_DIR = tables.REPO_ROOT / "data" / "raw"
DERIVED_DIR = tables.DERIVED_DIR

# What makes a transaction row the same ticket as another: same order, same tier, same
# moment, same price.
TICKET_KEY = ["order_id", "tier_name", "purchased_at", "price_paid"]


def _drop_repulled_rows(tx: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows a second export repeated, keeping genuine repeats inside one export.

    ~150 events arrive as many files and one overlapping re-pull would double-count
    revenue. But two identical rows inside ONE file are two real tickets on one order —
    per-ticket exports repeat the order_id when an order holds several tickets — so a
    plain drop_duplicates would delete real sales.

    So: number each row's occurrence within its own file, then de-duplicate on
    (ticket key + occurrence number). File B's first copy collides with file A's first
    copy and is dropped; file A's own second copy is occurrence 2 and survives.
    """
    key = [c for c in TICKET_KEY if c in tx.columns]
    tx = tx.copy()
    tx["occurrence"] = tx.groupby([*key, "source_file"], dropna=False).cumcount()
    deduped = tx.drop_duplicates(subset=[*key, "occurrence"])
    return deduped.drop(columns=["occurrence", "source_file"]), len(tx) - len(deduped)


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every raw export, run it through its adapter, and concatenate.

    Returns (transactions, costs) in canonical columns.
    """
    tx_parts: list[pd.DataFrame] = []
    cost_parts: list[pd.DataFrame] = []
    skipped: list[str] = []

    for path in adapters.list_raw_files(raw_dir):
        platform, adapter = adapters.adapter_for(path)
        if adapter is None:
            skipped.append(path.name)
            continue
        sheet = adapters.COSTS_SHEET_NAME if platform == "costs" else None
        frame = adapter(adapters.read_raw_file(path, sheet_name=sheet))
        if platform == "costs":
            cost_parts.append(frame)
        else:
            # Kept only until the de-duplication below; never written to a derived table.
            tx_parts.append(frame.assign(source_file=path.name))

    if skipped:
        print(f"skipped (no adapter — rename the file to say tbc or youni): {skipped}")

    if not tx_parts:
        raise FileNotFoundError(f"no ticket exports found under {raw_dir}")
    if not cost_parts:
        raise FileNotFoundError(f"no cost workbooks found under {raw_dir / 'costs'}")

    transactions, dropped = _drop_repulled_rows(pd.concat(tx_parts, ignore_index=True))
    if dropped:
        print(f"dropped {dropped:,} transaction rows already present in an earlier export")

    return transactions.reset_index(drop=True), pd.concat(cost_parts, ignore_index=True)


def run_ingest(raw_dir: Path = RAW_DIR, out_dir: Path = DERIVED_DIR) -> str:
    """Raw exports -> three parquet tables in out_dir. Returns the validation report."""
    tx_raw, costs_raw = load_raw(raw_dir)

    events = tables.build_events(costs_raw)
    transactions = tables.build_transactions(tx_raw, events)
    event_tier = tables.build_event_tier(transactions)

    for name, frame in [
        ("events", events),
        ("transactions", transactions),
        ("event_tier", event_tier),
    ]:
        path = tables.write_table(frame, name, out_dir=out_dir)
        print(f"wrote {path}  ({len(frame):,} rows)")

    report = validate.format_report(validate.validate(events, transactions, event_tier))
    # Written next to the tables, inside the gitignored data/ tree: the report quotes real
    # tier labels and real event counts, so it must not sit anywhere git can see it.
    report_path = Path(out_dir).resolve() / "validation_report.txt"
    report_path.write_text(report)
    print(f"wrote {report_path}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pricing.ingest", description=__doc__)
    parser.add_argument(
        "--sniff",
        action="store_true",
        help="print an anonymised schema summary of data/raw/ and exit (no row values)",
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    args = parser.parse_args(argv)

    if args.sniff:
        print(adapters.format_sniff(adapters.sniff_raw(args.raw_dir)))
        return 0

    try:
        print(run_ingest(args.raw_dir, args.out_dir))
    except (NotImplementedError, FileNotFoundError, ValueError) as exc:
        print(f"ingest stopped: {exc}", file=sys.stderr)
        print("\nrun `uv run python -m pricing.ingest --sniff` first.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
