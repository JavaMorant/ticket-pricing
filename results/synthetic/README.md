# results/synthetic/

Output of `src/pricing/synthetic.py`. **Nothing in here is real** — Brand A / Brand B,
venues V1..V4, cities City A / City B, ids prefixed `FAKE`. Committable on purpose.

By contrast `results/` itself is gitignored: real-data results carry real prices, real
margins and real event counts, and they get committed only after the pre-publication
review gate (DECISIONS.md, Fable gate 3).

## Regenerating

```
uv run python -m pricing.synthetic --synthetic --write
```

writes `fixtures/events.parquet`, `fixtures/transactions.parquet`,
`fixtures/event_tier.parquet` and `fixtures/ground_truth.json`.

## Phase reports

Each phase is a script, and every number in its report comes back from re-running it —
there is no state between runs and nothing is edited by hand.

```
uv run python -m pricing.phase1 --synthetic     # phase1_descriptives.md   + 5 figures
uv run python -m pricing.phase2 --synthetic     # phase2_demand_forecast.md + 3 figures
uv run python -m pricing.phase3 --synthetic     # phase3_sales_curve.md     + 3 figures
uv run python -m pricing.phase4 --synthetic     # phase4_elasticity.md
uv run python -m pricing.phase5 --synthetic     # phase5_risk.md
uv run python -m pricing.phase6 --synthetic     # phase6_counterfactual.md
```

Every one of them refuses `--real` and prints why. There are two gates in order: the
pre-registration gate (SPEC.md §5.3 — `preregistration.md` must not be DRAFT or MISSING),
and then the data gate (the three derived tables must exist). Since commit `1c7196d` the
list is FROZEN, so it is the second gate that refuses today: `data/raw/` is empty, `ingest`
has never run, and `data/derived/` does not exist. The reports and PNGs here are
committable for the same reason the fixture is: they describe a programme that does not
exist.

## Why the parquet files are not committed

The repo-wide `*.csv` / `*.xls[x]` / `*.parquet` rules in `.gitignore` are the privacy
guard: nothing shaped like a data export can reach git by accident, whatever directory it
lands in. The synthetic fixtures are regenerable from a seed in one command, so they gain
nothing from being committed and would cost that guarantee. Tests call
`synthetic.generate(seed)` directly rather than reading these files.

If a later phase genuinely needs them committed — a report that has to render without
running Python, say — add one explicit negation (`!results/synthetic/fixtures/*.parquet`)
and say why here. Do not weaken the repo-wide rule.

## What the fixture is for

It is the dataset whose demand elasticity we chose (`BETA_TRUE = -0.80`), so every phase
can be tested rather than believed. See `docs/EXPLAINERS.md` for the data-generating
process and `docs/VIVA.md` for the defence.
