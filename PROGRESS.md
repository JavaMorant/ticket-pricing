# PROGRESS — ticket-pricing

**Last updated:** 2026-08-11 (Phase 0 scaffolding — review findings 1–23 applied)
**Phase:** 0 (data assembly) — plumbing built, **no data has been ingested**
**Branch:** working tree, uncommitted, left for review

---

## Status in one line

The ingest framework runs end to end on synthetic data and refuses to run on real data
until four column maps are filled in. Nothing downstream of Phase 0 has been started.

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

1. Get the exports into `data/raw/`. Everything else is blocked on this.
2. `--sniff`, fill the four maps, resolve face-vs-fee, run ingest, read the report.
3. Fill SPEC §7's `[N]` events and `[M]`k transactions from the report's row counts.
4. Freeze `preregistration.md` — it is still marked DRAFT and per SPEC §5.3 it must be
   frozen **before first data contact**, i.e. before step 1 above.
5. Then Phase 1 descriptives.

**Gate check:** nothing here touches live money or the statistical core, so no Fable gate
is due. The next Fable gate is #2 (Phase 4 identification review), a long way off.
