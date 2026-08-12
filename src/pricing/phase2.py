"""Phase 2 — demand forecast (SPEC.md 6.1). Predict final tickets sold, at pricing time.

    uv run python -m pricing.phase2 --synthetic

The question: **given only what I knew when I set the prices, how many tickets will this
event sell?** No causal claim is made or needed here (SPEC.md 6.1) — this model does not
say marketing spend *causes* sales, only that events with big marketing budgets sell more,
which is true and useful for forecasting because I decide the budget before the on-sale.

Method: plain OLS on log(tickets), fitted with statsmodels, validated two ways —

  * **grouped 5-fold cross-validation, grouped by event** (SPEC.md 8.1). Every row here is
    one event, so grouping is trivially satisfied; the split function is still written and
    tested as a grouped split because Phases 3 and 4 run on tier-level rows where it is
    not trivial at all, and because a split that is only accidentally correct stops being
    correct the first time someone reuses it.
  * **final-year holdout** (SPEC.md 5.5): fit on the first four academic years, predict the
    fifth. Harder than cross-validation and closer to how the model would actually be used.

Both are scored against a naive baseline: the mean tickets of the same brand at the same
venue, computed on the training rows only. A model that cannot beat "how did this room do
last time" is not earning its complexity.

WHAT IS NOT IN THE FEATURE SET, AND WHY
---------------------------------------
No academic-calendar feature: no freshers' week, no term phase, no exam period, no student
loan instalment date. They are on the SPEC.md 5.4 pre-registered list, `preregistration.md`
still says DRAFT, and per SPEC.md 5.3 they may not be tested until it is frozen. The
practical consequence is worth stating in the report rather than hiding: the calendar is a
large, real driver of demand, so **the error reported here is an upper bound**. Adding the
frozen calendar features later should lower it, and if it does not, that is itself a
finding.

Day of week IS included. It appears on the 5.4 list, but here it is a predictor in a
forecasting model, not a hypothesis under test — no p-value on it is quoted, no claim is
made that Thursday causes anything. The pre-registration discipline governs *pattern
claims*, and this phase makes none.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pricing import phaseio, report
from pricing.phase1 import event_outcomes, live

N_FOLDS = 5
CV_SEED = 20260812  # fixed so the fold assignment is reproducible, not cherry-picked


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
#
# Every entry is (column, what it is, why it is known before a single ticket is sold).
# The third column is the point of the table: SPEC.md 8.2 is not a rule about column names,
# it is a rule about *when you knew things*.
FEATURE_NOTES: list[tuple[str, str, str]] = [
    ("log_capacity", "log licensed capacity", "the room is booked before the on-sale"),
    (
        "log_guarantee",
        "log artist guarantee — the artist billing-tier proxy",
        (
            "the contract is signed before the on-sale; the fee IS the billing tier, priced by "
            "an agent who knows the draw"
        ),
    ),
    (
        "log_marketing",
        "log(1 + marketing spend)",
        (
            "budgeted at planning. CAVEAT: the figure in the workbook is REALISED spend, and "
            "realised spend reacts to weak presale. Swap in the budgeted figure when the "
            "workbook carries it"
        ),
    ),
    (
        "lead_to_announce_days",
        "days from first on-sale to the event",
        "it is the schedule — the operator picks it",
    ),
    ("brand", "Brand A / Brand B", "known"),
    ("venue", "venue", "booked"),
    (
        "city",
        "city",
        (
            "booked with the venue. Enters through the venue dummies rather than a term of its "
            "own — every venue sits in one city, so separate city dummies would be redundant"
        ),
    ),
    ("dow", "day of week of the event", "known"),
    (
        "year_index",
        "academic year 0..4, plus its interaction with brand",
        (
            "the brand-maturity trend (SPEC.md 8.9). Controls for growth so that nothing "
            "correlated with time gets to masquerade as a pattern"
        ),
    ),
]

FEATURE_COLUMNS = [name for name, _, _ in FEATURE_NOTES]

# Columns that are only knowable AFTER the event went on sale. None of these may ever
# appear in the design matrix; `assert_no_lookahead` enforces it by name and, more
# importantly, by value.
OUTCOME_COLUMNS = [
    "tickets",
    "comps",
    "comp_rate",
    "revenue_buyer",
    "revenue_face",
    "revenue_per_head",
    "n_orders",
    "sell_through",
    "room_fill",
    "sellout",
    "units_sold",
    "window_close",
]


def build_features(events: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per event: the target, the features, and nothing else.

    `lead_to_announce_days` is the only feature computed from `purchased_at`, and it is
    computed as the event's FIRST paid sale — the moment the ladder opened, i.e. the
    decision moment itself. Nothing after that moment enters it, which is why the
    truncation check in `assert_no_lookahead` passes. In real data this should be replaced
    by the scheduled on-sale date from the platform if it is exported; the first sale is a
    proxy for it, and an excellent one, because there is always an on-sale burst.

    Cancelled events are dropped (SPEC.md 8.4). Their ticket counts are truncated at the
    moment they were pulled, so training on them teaches the model that a cancelled event
    sells 60 tickets, which is not a fact about demand. The cost is that every number in
    this phase is conditioned on the event having gone ahead, and the report says so.
    """
    paid = transactions[~transactions["is_comp"]]
    first_onsale = paid.groupby("event_id")["purchased_at"].min().rename("first_onsale_at")

    df = live(event_outcomes(events, transactions)).merge(
        first_onsale, left_on="event_id", right_index=True, how="left"
    )
    df["lead_to_announce_days"] = (
        df["event_date"].dt.normalize() - df["first_onsale_at"].dt.normalize()
    ).dt.days

    df["log_tickets"] = np.log(df["tickets"])
    df["log_capacity"] = np.log(df["capacity"])
    df["log_guarantee"] = np.log(df["artist_guarantee"])
    df["log_marketing"] = np.log1p(df["marketing_spend"])

    # Academic years mapped to 0..4 in date order, so the trend term is a plain integer
    # slope. A categorical year would fit five free levels and could not extrapolate to the
    # held-out fifth year at all.
    years = sorted(df["academic_year"].unique())
    df["year_index"] = df["academic_year"].map({y: i for i, y in enumerate(years)})

    keep = [
        "event_id",
        "event_date",
        "academic_year",
        "tickets",
        "log_tickets",
        *FEATURE_COLUMNS,
    ]
    return df[keep].sort_values("event_date").reset_index(drop=True)


def formula(df: pd.DataFrame) -> str:
    """The OLS formula.

    `city` is a pricing-time feature and it is NOT a separate term here, on purpose. Every
    venue sits in exactly one city, so each city dummy is just the sum of that city's venue
    dummies: put both in and the design is rank-deficient. statsmodels would not complain —
    it would hand back numbers, split arbitrarily between the two sets of dummies, which is
    worse than an error. City is already in the model; the venue dummies carry it.

    `assert_city_is_nested_in_venue` checks that this is still true of the data, so if the
    programme ever gains a venue that moves city, or venues get pooled into an "other"
    bucket, the assumption fails loudly instead of quietly biasing the venue effects.
    """
    assert_city_is_nested_in_venue(df)
    terms = [
        "log_capacity",
        "log_guarantee",
        "log_marketing",
        "lead_to_announce_days",
        "C(dow)",
        "C(brand)",
        "C(venue)",
        # Brand-specific growth trend: a main effect plus an interaction, which is exactly
        # "each brand grew at its own rate" (SPEC.md 8.9).
        "year_index",
        "year_index:C(brand)",
    ]
    return "log_tickets ~ " + " + ".join(terms)


def assert_city_is_nested_in_venue(df: pd.DataFrame) -> None:
    """Every venue must sit in exactly one city, or `city` needs its own term."""
    spread = df.groupby("venue")["city"].nunique()
    bad = sorted(spread[spread > 1].index)
    if bad:
        raise ValueError(
            f"venue(s) {bad} appear in more than one city, so `city` is no longer implied by "
            f"`venue` and must be added to `formula` as C(city)."
        )


# --------------------------------------------------------------------------------------
# Look-ahead discipline (SPEC.md 8.2)
# --------------------------------------------------------------------------------------


def assert_no_lookahead(events: pd.DataFrame, transactions: pd.DataFrame) -> str:
    """Two checks, one by name and one by value. Raises on failure; returns what it did.

    **By name**: no feature is an outcome column. Cheap, and catches the obvious accident
    of dropping `sell_through` into the design matrix.

    **By value** — the one that actually matters: rebuild the whole feature frame using
    only the transactions that existed at each event's decision moment (its first paid
    sale), and require the result to be identical. Any feature that secretly reads later
    sales changes when the later sales are removed, whatever it is called. A name check
    can be defeated by renaming a column; this cannot.
    """
    banned = sorted(set(FEATURE_COLUMNS) & set(OUTCOME_COLUMNS))
    if banned:
        raise AssertionError(f"look-ahead: {banned} are outcomes, not pricing-time features")

    full = build_features(events, transactions)

    paid = transactions[~transactions["is_comp"]]
    decision = transactions["event_id"].map(paid.groupby("event_id")["purchased_at"].min())
    truncated = transactions[transactions["purchased_at"] <= decision]
    replayed = build_features(events, truncated)

    cols = ["event_id", *FEATURE_COLUMNS]
    a = full[cols].reset_index(drop=True)
    b = replayed[cols].reset_index(drop=True)
    if not a.equals(b):
        differing = [c for c in cols if not a[c].equals(b[c])]
        raise AssertionError(
            f"look-ahead: {differing} change when the data is truncated to the decision "
            f"moment, so they read sales that had not happened yet (SPEC.md 8.2)"
        )
    return (
        f"PASS — {len(FEATURE_COLUMNS)} features, none of them an outcome column, and all "
        f"{len(a)} rows reproduce identically from transactions truncated at each event's "
        f"first sale."
    )


# --------------------------------------------------------------------------------------
# Splits (SPEC.md 8.1)
# --------------------------------------------------------------------------------------


def grouped_kfold(groups: pd.Series, k: int = N_FOLDS, seed: int = CV_SEED) -> list[tuple]:
    """K folds in which every row of a group lands in exactly one fold.

    Ten lines, no sklearn. The whole idea: shuffle the *groups*, deal them round-robin into
    k piles, then a row's fold is its group's pile. Because the assignment is per group,
    a group can never be split across train and test — which is the entire point of
    SPEC.md 8.1. Splitting by row instead would put the same event in both halves and the
    model would be scored on an event it had already seen.

    Returns [(train_positions, test_positions), ...] as integer positions into `groups`.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    unique = groups.unique()
    shuffled = np.random.default_rng(seed).permutation(len(unique))
    fold_of_group = {unique[g]: i % k for i, g in enumerate(shuffled)}
    fold = groups.map(fold_of_group).to_numpy()
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]


def final_year_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the first four academic years, test on the fifth (SPEC.md 5.5)."""
    last = df["year_index"].max()
    return np.flatnonzero(df["year_index"] < last), np.flatnonzero(df["year_index"] == last)


# --------------------------------------------------------------------------------------
# Fitting and scoring
# --------------------------------------------------------------------------------------


def fit(train: pd.DataFrame):
    """OLS on log tickets. Returns the fitted statsmodels result."""
    return smf.ols(formula(train), data=train).fit()


def predict(model, test: pd.DataFrame, smear: float = 1.0) -> np.ndarray:
    """Predictions back on the ticket scale.

    exp() of a log-scale prediction gives the conditional MEDIAN, not the mean — the
    log-normal mean is bigger by exp(sigma^2 / 2). `smear` is Duan's smearing estimator,
    the mean of exp(training residuals), which corrects for that without assuming the
    residuals are normal. It is left at 1.0 for the headline numbers because MAE and MAPE
    are both minimised by the median, so the uncorrected prediction is the coherent one;
    the report quotes what the correction would do.
    """
    return np.exp(model.predict(test).to_numpy()) * smear


def smearing_factor(model) -> float:
    """Duan's smearing estimator: mean(exp(residual)) on the training rows."""
    return float(np.mean(np.exp(model.resid)))


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """The naive forecast: mean tickets for this brand at this venue.

    Three-step fallback — brand+venue, then brand, then the whole training set — because a
    fold can easily contain a brand-venue pair the training rows never saw, and a baseline
    that returns NaN there flatters the model by quietly dropping the hardest rows.
    """
    by_pair = train.groupby(["brand", "venue"])["tickets"].mean()
    by_brand = train.groupby("brand")["tickets"].mean()
    overall = train["tickets"].mean()

    pairs = pd.MultiIndex.from_frame(test[["brand", "venue"]])
    out = pd.Series(pairs.map(by_pair), index=test.index, dtype=float)
    out = out.fillna(test["brand"].map(by_brand))
    return out.fillna(overall).to_numpy()


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """MAE in tickets, MAPE in percent. Both on the ticket scale, never the log scale.

    Log-scale error is not quotable: "0.18 log points" means nothing to anyone deciding
    whether to book an artist. MAE is in tickets, which is the unit the decision is in.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(predicted - actual))),
        "MAPE_pct": float(np.mean(np.abs(predicted - actual) / actual) * 100),
    }


def _check_levels(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Refuse to score a fold whose test rows contain a category the fit never saw.

    statsmodels would raise something opaque from inside patsy; this says which column and
    which level, and what to do about it. It has never fired on the fixture — with four
    venues and ~117 events every fold sees every level — but on real data a venue used
    once will eventually land alone in a test fold, and that needs a decision (pool it into
    "other"), not a crash.
    """
    for col in ["brand", "venue", "dow", "city"]:
        unseen = sorted(set(test[col]) - set(train[col]))
        if unseen:
            raise ValueError(
                f"fold has {col} level(s) {unseen} that the training rows never saw. "
                f"Pool rare levels into 'other' in build_features before splitting."
            )


def evaluate_split(df: pd.DataFrame, train_idx, test_idx, label: str) -> tuple[dict, pd.DataFrame]:
    """Fit on train, score model and baseline on test. Returns (row, per-event predictions)."""
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    _check_levels(train, test)

    model = fit(train)
    predicted = predict(model, test)
    smeared = predict(model, test, smear=smearing_factor(model))
    naive = baseline_predict(train, test)

    row = {
        "split": label,
        "n_train": len(train),
        "n_test": len(test),
        **{f"model_{k}": v for k, v in scores(test["tickets"], predicted).items()},
        **{f"baseline_{k}": v for k, v in scores(test["tickets"], naive).items()},
        "smeared_MAE": scores(test["tickets"], smeared)["MAE"],
    }
    row["MAE_improvement_pct"] = 100 * (1 - row["model_MAE"] / row["baseline_MAE"])

    preds = test[["event_id", "event_date", "academic_year", "brand", "tickets"]].copy()
    preds["predicted"] = predicted
    preds["baseline"] = naive
    preds["split"] = label
    return row, preds


def cross_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grouped K-fold by event. Returns (per-fold table, out-of-fold predictions)."""
    rows, preds = [], []
    for i, (train_idx, test_idx) in enumerate(grouped_kfold(df["event_id"]), start=1):
        row, pred = evaluate_split(df, train_idx, test_idx, f"fold {i}")
        rows.append(row)
        preds.append(pred)

    folds = pd.DataFrame(rows)
    # Pooled, not the mean of the fold means: folds differ in size, and averaging the
    # averages would weight a 23-event fold the same as a 24-event one.
    out_of_fold = pd.concat(preds, ignore_index=True)
    pooled = {
        "split": "all folds pooled",
        "n_train": int(folds["n_train"].mean()),
        "n_test": len(out_of_fold),
        **{
            f"model_{k}": v
            for k, v in scores(out_of_fold["tickets"], out_of_fold["predicted"]).items()
        },
        **{
            f"baseline_{k}": v
            for k, v in scores(out_of_fold["tickets"], out_of_fold["baseline"]).items()
        },
        "smeared_MAE": float("nan"),
    }
    pooled["MAE_improvement_pct"] = 100 * (1 - pooled["model_MAE"] / pooled["baseline_MAE"])
    return pd.concat([folds, pd.DataFrame([pooled])], ignore_index=True), out_of_fold


def coefficient_table(model) -> pd.DataFrame:
    """Coefficients with 95% confidence intervals — effect sizes, not stars.

    SPEC.md 5.5 point 3: report effect sizes with uncertainty. On a log target a
    coefficient is roughly a proportional effect, so 0.35 on `log_capacity` reads as "a 10%
    bigger room sells about 3.5% more tickets".
    """
    ci = model.conf_int()
    out = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.to_numpy(),
            "std_err": model.bse.to_numpy(),
            "ci_low": ci[0].to_numpy(),
            "ci_high": ci[1].to_numpy(),
        }
    )
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def fig_fold_error(folds: pd.DataFrame, out_dir: Path) -> Path:
    """MAE per fold, model against the naive baseline."""
    per_fold = folds[folds["split"].str.startswith("fold")]
    x = np.arange(len(per_fold))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        x - 0.2, per_fold["baseline_MAE"], 0.4, label="baseline (brand-venue mean)", color="#999"
    )
    ax.bar(x + 0.2, per_fold["model_MAE"], 0.4, label="OLS forecast", color="#3d6bb3")
    ax.set_xticks(x)
    ax.set_xticklabels(per_fold["split"])
    ax.legend(fontsize=8, frameon=False)
    report.style(ax, "Held-out error by fold (grouped by event)", "", "MAE (tickets)")
    return report.save_fig(fig, out_dir, "phase2_fold_error.png")


def fig_predicted_vs_actual(oof: pd.DataFrame, holdout: pd.DataFrame, out_dir: Path) -> Path:
    """Out-of-fold and final-year predictions against what actually sold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, frame, title in (
        (ax1, oof, "Out-of-fold (grouped 5-fold)"),
        (ax2, holdout, "Final-year holdout"),
    ):
        for brand in sorted(frame["brand"].unique()):
            sub = frame[frame["brand"] == brand]
            ax.scatter(
                sub["tickets"],
                sub["predicted"],
                s=20,
                alpha=0.75,
                label=brand,
                color=report.BRAND_COLOURS.get(brand, report.GREY),
            )
        lo = min(frame["tickets"].min(), frame["predicted"].min()) * 0.8
        hi = max(frame["tickets"].max(), frame["predicted"].max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color="#b03030", linewidth=1.0, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        report.log_ticks(ax)
        ax.legend(fontsize=8, frameon=False)
        report.style(ax, title, "actual tickets", "predicted tickets")
    return report.save_fig(fig, out_dir, "phase2_predicted_vs_actual.png")


def fig_residuals(oof: pd.DataFrame, out_dir: Path) -> Path:
    """Where the model is wrong: by size of event, and by academic year."""
    resid = oof["predicted"] - oof["tickets"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.axhline(0, color="#b03030", linewidth=1.0)
    ax1.scatter(oof["tickets"], resid, s=18, alpha=0.7, color=report.GREY)
    ax1.set_xscale("log")
    report.style(ax1, "Out-of-fold residuals vs event size", "actual tickets", "predicted - actual")

    years = sorted(oof["academic_year"].unique())
    ax2.axhline(0, color="#b03030", linewidth=1.0)
    ax2.boxplot(
        [resid[oof["academic_year"] == y] for y in years],
        tick_labels=years,
        widths=0.55,
        medianprops={"color": "#b03030"},
    )
    ax2.tick_params(axis="x", labelsize=8)
    report.style(ax2, "Out-of-fold residuals by academic year", "", "predicted - actual")
    return report.save_fig(fig, out_dir, "phase2_residuals.png")


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def build(run: phaseio.Run) -> list[Path]:
    out_dir = run.out_dir
    events, transactions = run.events, run.transactions

    lookahead = assert_no_lookahead(events, transactions)
    df = build_features(events, transactions)

    folds, oof = cross_validate(df)
    train_idx, test_idx = final_year_split(df)
    holdout_row, holdout_preds = evaluate_split(df, train_idx, test_idx, "final-year holdout")

    full_model = fit(df)
    holdout_model = fit(df.iloc[train_idx])

    figures = [
        fig_fold_error(folds, out_dir),
        fig_predicted_vs_actual(oof, holdout_preds, out_dir),
        fig_residuals(oof, out_dir),
    ]

    pooled = folds.iloc[-1]
    metrics = pd.DataFrame([*folds.to_dict("records"), holdout_row])

    lines = [
        phaseio.provenance_header(
            "Phase 2 — Demand forecast",
            run.source,
            run.seed,
            phaseio.regenerate_command("phase2", run.source, run.seed),
        ),
        "",
    ]
    lines += [
        "Predict **final paid tickets** from what was known at pricing time. Plain OLS on",
        "`log(tickets)`, scored on held-out events against a naive brand-venue mean.",
        "",
        "## Headline",
        "",
        (
            f"- Grouped 5-fold CV, pooled: **MAE {pooled['model_MAE']:.1f} tickets** "
            f"({pooled['model_MAPE_pct']:.1f}% MAPE) vs baseline "
            f"{pooled['baseline_MAE']:.1f} ({pooled['baseline_MAPE_pct']:.1f}%) — "
            f"**{pooled['MAE_improvement_pct']:.0f}% better**"
        ),
        (
            f"- Final-year holdout: **MAE {holdout_row['model_MAE']:.1f} tickets** "
            f"({holdout_row['model_MAPE_pct']:.1f}% MAPE) vs baseline "
            f"{holdout_row['baseline_MAE']:.1f} ({holdout_row['baseline_MAPE_pct']:.1f}%) — "
            f"**{holdout_row['MAE_improvement_pct']:.0f}% better**"
        ),
        (
            f"- Sample: {len(df)} events that ran, "
            f"{df['academic_year'].min()} to {df['academic_year'].max()}, "
            f"median {df['tickets'].median():.0f} tickets"
        ),
        "",
        "The holdout is the honest number. It is a genuine forecast of a year the model",
        "never saw, and it has to extrapolate the brand-growth trend one year past its data.",
        "",
        "---",
        "",
        "## 1. Look-ahead check (SPEC.md §8.2)",
        "",
        f"`assert_no_lookahead`: {lookahead}",
        "",
        "The second half of that check is the one worth understanding. Feature names can be",
        "made to look innocent; behaviour cannot. So the whole feature frame is rebuilt from",
        "transactions truncated at each event's first sale, and required to come out",
        "identical. Any feature that quietly reads later sales moves when the later sales are",
        "taken away.",
        "",
        "## 2. Features",
        "",
        report.md_table(
            pd.DataFrame(FEATURE_NOTES, columns=["feature", "what it is", "known at pricing time"])
        ),
        "",
        "**Not included: every academic-calendar feature on the SPEC.md §5.4 list.**",
        "`preregistration.md` is still DRAFT, and §5.3 forbids testing those patterns before",
        "it is frozen. The calendar is a big real driver of student-event demand, so the",
        "errors below are an **upper bound** — the model is deliberately fighting with one",
        "hand behind its back. Re-run after the freeze and the gap should close.",
        "",
        "`marketing_spend` deserves its own warning. It predicts well precisely because it is",
        "endogenous: more is spent on events already believed in. Useful for forecasting,",
        "worthless as a causal statement, and **not** a valid instrument for price",
        "(SPEC.md §4.5) for exactly that reason.",
        "",
        "## 3. Held-out error",
        "",
        report.md_table(
            metrics[
                [
                    "split",
                    "n_train",
                    "n_test",
                    "model_MAE",
                    "model_MAPE_pct",
                    "baseline_MAE",
                    "baseline_MAPE_pct",
                    "MAE_improvement_pct",
                ]
            ],
            floatfmt="{:,.1f}",
        ),
        "",
        (
            f"Duan smearing factor on the holdout fit: **{smearing_factor(holdout_model):.4f}**."
            f" Applying it moves holdout MAE from {holdout_row['model_MAE']:.1f} to"
            f" {holdout_row['smeared_MAE']:.1f} tickets."
        ),
        "",
        "`exp()` of a log-scale fit predicts the",
        "median, not the mean; MAE and MAPE are minimised by the median, so the headline",
        "numbers leave it off deliberately. It matters for Phase 5, where the *mean* profit",
        "is the quantity of interest.",
        "",
        "## 4. Coefficients (fitted on all events)",
        "",
        (
            f"R-squared {full_model.rsquared:.3f}, adjusted {full_model.rsquared_adj:.3f}, "
            f"{int(full_model.df_model) + 1} parameters on {int(full_model.nobs)} events."
        ),
        "",
        report.md_table(coefficient_table(full_model), floatfmt="{:,.3f}"),
        "",
        "On a log target a coefficient is approximately a proportional effect: a coefficient",
        "of 0.25 on `log_capacity` means a 10% bigger room sells about 2.5% more. These are",
        "**predictive associations, not causal effects** — capacity, guarantee and marketing",
        "spend are all chosen by the same person, at the same time, using the same beliefs",
        "about how the night will go (SPEC.md §4.1). Read them as 'what an event like this",
        "usually does', never as 'what would happen if I changed this'.",
        "",
        "## 5. What this cannot do",
        "",
        (
            f"- **{int(full_model.nobs)} events is small** (SPEC.md §8.5). "
            f"{int(full_model.df_model) + 1} parameters on that sample is already at the limit"
        ),
        "  of what the data supports; this is exactly why there is no gradient booster here.",
        "- **Cancelled events are excluded**, so every number is conditioned on the event",
        "  having gone ahead (SPEC.md §8.4).",
        "- **Events are not independent** (SPEC.md §8.10): two nights a week apart eat each",
        "  other's audience. The cannibalisation features (`days_since_last_event`,",
        "  `events_in_trailing_14d`) are on the pre-registered list and are not in this model",
        "  yet. Standard errors above assume an independence the programme does not have.",
        "- **The year trend extrapolates.** The holdout fit sees years 0-3 and predicts year",
        "  4 with a straight line. That is the right test, and it is also the model's most",
        "  fragile assumption.",
        "",
        "---",
        "",
        "## Figures",
        "",
    ]
    lines += [f"- `{p.name}`" for p in figures]
    lines += ["", "".join(f"![{p.stem}]({p.name})\n" for p in figures)]

    print(f"look-ahead {lookahead}")
    print(
        f"CV pooled   MAE {pooled['model_MAE']:7.1f}  MAPE {pooled['model_MAPE_pct']:5.1f}%   "
        f"baseline MAE {pooled['baseline_MAE']:7.1f}  ({pooled['MAE_improvement_pct']:+.0f}%)"
    )
    print(
        f"holdout     MAE {holdout_row['model_MAE']:7.1f}  "
        f"MAPE {holdout_row['model_MAPE_pct']:5.1f}%   "
        f"baseline MAE {holdout_row['baseline_MAE']:7.1f}  "
        f"({holdout_row['MAE_improvement_pct']:+.0f}%)"
    )

    return [phaseio.write_report(lines, "phase2_demand_forecast.md", out_dir), *figures]


def main(argv: list[str] | None = None) -> int:
    return phaseio.run_phase(build, "pricing.phase2", __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
