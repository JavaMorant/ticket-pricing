"""Phase 4 — price elasticity of demand. SPEC.md section 4. This is the project.

    uv run python -m pricing.phase4 --synthetic
    uv run python -m pricing.phase4 --real          <- refused while the prereg is DRAFT

What this module does, in the order the report prints it
--------------------------------------------------------
1. **The naive pooled regression** `log Q ~ log P` across all tiers of all events.
   SPEC.md 4.1 predicts it comes out with the WRONG SIGN, because the operator set the
   prices using the same information that drove demand. We run it, we print it, and we
   explain it. We do not drop variables until the sign turns over: that is p-hacking your
   own business (SPEC.md 8.3).

2. **Event fixed effects** — every column demeaned by event, so beta is identified off
   within-event price variation only (SPEC.md 4.2). Everything constant inside an event
   (the artist, the date, the venue, the hype, the marketing spend) is absorbed. The
   dummy-variable version of the same regression is kept in `event_fixed_effects_dummies`
   and gives an identical slope; there is a test that asserts they agree to within 1e-9.

3. **The lead-time confound check** (SPEC.md 4.3). Early Bird is cheap AND early; the door
   price is expensive AND last-minute. Adding a lead-time control asks whether any price
   variation survives, and the diagnostics say how much. This is where the honest verdict
   comes from, and the criteria are numbers in `IDENTIFICATION_CRITERIA`, not a feeling.

4. **A cost-shifter instrument** (SPEC.md 4.5). The venue put its hire rate up and it went
   into the ticket price; the buyer neither knows nor cares. That moves price without
   moving demand — if it is strong enough to be worth using. The first-stage F decides,
   against a stated floor, and the exclusion argument is written out in the report.

The output is a markdown report with the numbers in it, plus a verdict of IDENTIFIED or
NOT-IDENTIFIED. **NOT-IDENTIFIED is a first-class result** (SPEC.md 4.4): it is the same
move as the deflated Sharpe, and the report says what would fix it — the randomised tier
experiment already running under `experiment/DESIGN.md`.

Method policy (DECISIONS.md, 2026-08-11 amendment): plain pandas, plain statsmodels OLS,
the within transform (subtract each event's own mean from every column) rather than one
dummy per event. Both are allowed by the amendment and both give the same slope; the
within version is used because of what it does to the *inference*. With ~120 events and
~400 rows, the dummy version estimates 119 parameters from 117 clusters, and a
cluster-robust covariance matrix built from 117 clusters has rank at most 116 — fewer
clusters than parameters is not a footing you want an interval to stand on. Demeaning
leaves two parameters against 117 clusters. Standard errors are clustered by event either
way, because two tiers of the same event are not two independent draws, and the intervals
use a t distribution with G-1 degrees of freedom rather than the normal.
"""

from __future__ import annotations

import re
import sys
from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pricing import phaseio

# The pre-registered feature list (SPEC.md 5.4) lives in `dataset.FEATURES`. Phase 4 uses
# two of its entries directly — `lead_time_days` as the confound control, `academic_year`
# as the SPEC.md 8.9 trend control in the instrument specs — but it does not import the
# list, because it tests none of them. Both controls survived the freeze (preregistration.md
# is FROZEN as of 1c7196d): "Lead time / days-to-event" and "Academic-year cohort effects"
# are bullets on the frozen list, and the reconciliation is pinned by
# `test_every_feature_slug_traces_to_a_bullet_in_the_frozen_preregistration`.

REPORT_FILENAME = "phase4_elasticity.md"

# --------------------------------------------------------------------------------------
# The verdict criteria. Numbers, in code, decided before the estimate was looked at.
# --------------------------------------------------------------------------------------
#
# A verdict of "identified" is a claim that this dataset can answer the question. It is
# not a claim about the p-value. Four things have to be true, and each one is a number:
#
# min_residual_price_share
#     Share of the variance of log price that SURVIVES event dummies and the lead-time
#     control. This is the identifying variation and nothing else is. At 0.02, two percent
#     of price movement is not explained by "which event is it" or "how far out was it" —
#     a mid-window correction, a ladder applied differently, a promo. Below that, beta is
#     being fitted off rounding and typing errors, whatever the standard error says.
#     Equivalently: VIF(log price) <= 50. The textbook VIF rule of thumb is 10, but a VIF
#     is only a statement about how much the standard error is inflated; whether the
#     resulting interval is usable is the NEXT criterion, and that is the one that matters.
#     A VIF above 10 is reported as a flag, not a failure.
# max_ci_width
#     Width of the 95% confidence interval on beta, clustered by event. An interval wider
#     than 1.0 (i.e. wider than +/- 0.5) cannot distinguish "raising prices costs me
#     nothing" from "raising prices costs me the room". That is not an elasticity, it is a
#     shrug with three decimal places.
# min_events
#     Events contributing at least two priced tiers. Fewer than 30 and the within
#     estimator is riding on a handful of ladders.
# require_negative_and_significant
#     The interval must lie entirely below zero. A positive or zero-straddling interval
#     from the within estimator means the confound is still in there; it does NOT mean
#     demand slopes upward.
#     THIS ONE IS NOT LIKE THE OTHER THREE, and the report says so in those words. The
#     first three are properties of the DESIGN — is there variation, is the interval
#     usable, are there enough ladders — and they can be checked without ever looking at
#     the sign of the answer. This one conditions on the sign. It is carried as a separate
#     kind of check (`Check.kind == "sign"`) and reported under its own heading, because a
#     criterion you can only ever pass when the estimate came out the way theory predicts
#     is not a test of identification, and pretending otherwise is how a project gets read
#     as p-hacked (SPEC.md 4.1, 8.3).
IDENTIFICATION_CRITERIA: dict[str, float | bool] = {
    "min_residual_price_share": 0.02,
    "max_ci_width": 1.00,
    "min_events": 30,
    "require_negative_and_significant": True,
}

# Reported as a flag rather than a failure — see the note above.
VIF_FLAG_LEVEL = 10.0

# Relevance floor for an instrument (SPEC.md 4.5). The first-stage F on the excluded
# instrument must clear this before the instrumental-variable estimate is computed at all.
# 10 is the standard rule of thumb for one instrument; below it the estimate is biased
# towards the very OLS estimate it is supposed to fix, so reporting it would be worse than
# reporting nothing.
FIRST_STAGE_F_FLOOR = 10.0

Z_ALPHA = 1.959963985  # 95% normal critical value


# --------------------------------------------------------------------------------------
# 1. The panels
# --------------------------------------------------------------------------------------


def build_panel(event_tier: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """The tier-level panel Phase 4 runs on: one row per (event x tier).

    Four filters, each with a reason:
      * cancelled events out    — SPEC.md 8.4, they did not happen. They are kept in the
                                  dataset and excluded here, so the exclusion is visible.
      * comp-only tiers out     — SPEC.md 8.7, a guestlist is not demand (its price is 0).
      * zero-sale tiers out     — log(0).
      * single-tier events out  — an event dummy with one observation fits that row
                                  perfectly and contributes nothing to beta.

    Adds `log_q`, `log_p` and `lead_term`. `lead_term` is log1p(days the tier opened
    before the event), which is the SPEC.md 4.3 control: log1p because the difference
    between 60 and 70 days out is worth far less than the difference between 0 and 10.
    """
    df = event_tier.copy()
    if events is not None and "cancelled" in events.columns:
        df = df.merge(events[["event_id", "cancelled"]], on="event_id", how="left")
        df = df[~df["cancelled"].fillna(False)].drop(columns=["cancelled"])
    df = df[(df["units_sold"] > 0) & (df["price"] > 0)].copy()
    df["log_q"] = np.log(df["units_sold"])
    df["log_p"] = np.log(df["price"])
    df["lead_term"] = np.log1p(df["lead_time_open_days"].clip(lower=0))
    df = df[df.groupby("event_id")["event_id"].transform("size") >= 2]
    return df.reset_index(drop=True)


def build_event_panel(event_tier: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """One row per event — the panel the instrument runs on.

    An instrument built from the cost stack (the venue's hire rate) varies ACROSS events,
    not within one. An event fixed effect would absorb it completely. So the 2SLS story is
    a different regression from the fixed-effects story, on a different panel, and saying
    which is which is half the point of SPEC.md 4.5.

    `log_p` is the unweighted mean of log price across the event's priced tiers — the
    LEVEL of the ladder. Deliberately not quantity-weighted: the weights would be the
    quantities that are also the left-hand side, so a weighted price is partly an outcome.
    """
    tiers = event_tier[(event_tier["units_sold"] > 0) & (event_tier["price"] > 0)].copy()
    tiers["log_p"] = np.log(tiers["price"])
    grouped = (
        tiers.groupby("event_id")
        .agg(units=("units_sold", "sum"), log_p=("log_p", "mean"), n_tiers=("units_sold", "size"))
        .reset_index()
    )

    keep = [
        c
        for c in [
            "event_id",
            "event_date",
            "brand",
            "venue",
            "city",
            "capacity",
            "cancelled",
            "venue_cost",
            "artist_guarantee",
            "marketing_spend",
            "security_cost",
            "production_cost",
            "fixed_cost_stack",
        ]
        if c in events.columns
    ]
    out = grouped.merge(events[keep], on="event_id", how="inner")
    if "cancelled" in out.columns:
        out = out[~out["cancelled"].fillna(False)].drop(columns=["cancelled"])
    out = out[out["units"] > 0].copy()

    out["log_q"] = np.log(out["units"])
    out["academic_year"] = phaseio.academic_year(out["event_date"])
    # Instrument candidates, both in logs so their coefficients are elasticities too.
    if "venue_cost" in out.columns:
        out["log_hire"] = np.log(out["venue_cost"].where(out["venue_cost"] > 0))
    if "artist_guarantee" in out.columns:
        out["log_guarantee"] = np.log(out["artist_guarantee"].where(out["artist_guarantee"] > 0))
    if "marketing_spend" in out.columns:
        out["log_marketing"] = np.log(out["marketing_spend"].where(out["marketing_spend"] > 0))
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 2. The estimators
# --------------------------------------------------------------------------------------


class Estimate(NamedTuple):
    """One elasticity estimate and everything needed to quote it honestly.

    `n_events` doubles as the number of CLUSTERS, because the clustering variable is the
    event. `n_params` is carried next to it on purpose: the comparison that matters for a
    cluster-robust interval is parameters against clusters, and the only way to have that
    argument in the report is to print both numbers.
    """

    name: str
    formula: str
    beta: float
    se: float
    ci_low: float
    ci_high: float
    n_rows: int
    n_events: int
    n_params: int

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def line(self) -> str:
        return (
            f"{self.beta:+.3f} (SE {self.se:.3f}, 95% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}])"
        )


def _fit_clustered(formula: str, panel: pd.DataFrame):
    """OLS with standard errors clustered by event, and a t interval on G-1 clusters.

    Two tiers of one event share the artist, the date and the marketing push, so their
    residuals are correlated and ordinary standard errors are too small. Clustering by
    event is the minimum honest adjustment (SPEC.md 8.10 notes the same problem across
    events, which clustering does NOT fix — see the report's caveats).

    `use_t=True` is the second half of that adjustment and it is easy to miss. A
    cluster-robust covariance matrix is built from G independent pieces, not from N rows,
    so the reference distribution is a t on G-1 = 116 degrees of freedom, not a normal.
    With 117 clusters the difference is small (1.981 against 1.960) — it is here because
    quoting a normal critical value off 117 clusters is exactly the kind of thing that
    gets asked about, and the honest answer should be "it is a t, on G-1".
    """
    return smf.ols(formula, data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["event_id"]}, use_t=True
    )


def _estimate(name: str, formula: str, panel: pd.DataFrame, shown: str | None = None) -> Estimate:
    """Fit `formula` on `panel` and package the one coefficient this project is about.

    `shown` is the formula to PRINT when the fitted formula is not the readable one — the
    within transform is fitted on demeaned columns with no intercept, and printing that
    without saying what it is would be less clear than printing the dummy form it equals.
    """
    fit = _fit_clustered(formula, panel)
    ci = fit.conf_int().loc["log_p"]
    return Estimate(
        name=name,
        formula=shown or formula,
        beta=float(fit.params["log_p"]),
        se=float(fit.bse["log_p"]),
        ci_low=float(ci.iloc[0]),
        ci_high=float(ci.iloc[1]),
        n_rows=int(fit.nobs),
        n_events=int(panel["event_id"].nunique()),
        n_params=len(fit.params),
    )


def naive_pooled(panel: pd.DataFrame) -> Estimate:
    """SPEC.md 4.1: log Q on log P, pooled across events. Expect the WRONG sign.

    Reported, not hidden. A positive beta here is the finding that motivates everything
    below it: it says price is correlated with the error term, which is what you would
    expect from prices the operator set using their own demand expectations.
    """
    return _estimate("naive pooled OLS", "log_q ~ log_p", panel)


def _demean(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The within transform: subtract each event's own mean from each of `columns`.

    Three lines, and it is exactly what a dummy per event does — the Frisch-Waugh-Lovell
    theorem says regressing demeaned y on demeaned x gives the same slope as putting the
    dummies in. `event_id` is left alone so the same frame can still be clustered on it.
    """
    out = panel.copy()
    for col in columns:
        out[col] = out[col] - out.groupby("event_id")[col].transform("mean")
    return out


def event_fixed_effects(panel: pd.DataFrame, control_lead_time: bool = True) -> Estimate:
    """SPEC.md 4.2: alpha_e absorbed by demeaning every column within its event.

    This is the same regression as one dummy per event and gives the same slope to within
    1e-9 (`test_the_within_transform_and_the_dummies_are_the_same_regression`).
    What differs is the count of parameters, and with cluster-robust standard errors that
    is not cosmetic:

        dummies       119 parameters (117 event dummies + log_p + lead_term + intercept)
                      estimated from 117 clusters. The cluster-robust "meat" matrix is a
                      sum of 117 outer products, so its rank is at most 116 — it is
                      singular, and there are ~3.4 rows per cluster to fit a dummy each.
        within        2 parameters against 117 clusters. No rank problem, and no dummy
                      soaking up a third of its own cluster's residual variation.

    Both are permitted by DECISIONS.md ("dummies or within-transform — whichever is easier
    to explain"). The dummy version stays in `event_fixed_effects_dummies` and its
    (wider) interval is printed next to this one in the report, so switching estimator
    cannot be a quiet way of getting a tighter interval.

    `control_lead_time=True` adds log1p(days the tier opened before the event) — SPEC.md
    4.3. Without it, the estimate charges the timing effect to the price, because early
    tiers are cheap AND early.
    """
    columns = ["log_q", "log_p", "lead_term"] if control_lead_time else ["log_q", "log_p"]
    rhs = "log_p + lead_term" if control_lead_time else "log_p"
    name = "event FE + lead-time control" if control_lead_time else "event FE only"
    return _estimate(
        name,
        f"log_q ~ {rhs} - 1",
        _demean(panel, columns),
        shown=f"log_q ~ {rhs} + C(event_id), fitted as the within transform",
    )


def event_fixed_effects_dummies(panel: pd.DataFrame, control_lead_time: bool = True) -> Estimate:
    """The literal dummy-variable version of `event_fixed_effects`. Kept for two jobs.

    1. It pins the equivalence: the slope must match the within estimate exactly, and a
       test asserts it. If they ever diverge, one of the two is wrong.
    2. Its clustered interval is the conservative comparison printed in the report. The
       raw cluster-robust variance is identical between the two; they differ only through
       statsmodels' finite-sample factor `(N-1)/(N-K) * G/(G-1)`, which counts all 117
       absorbed dummies in K and so inflates this one by about 19%. Which K belongs in
       that factor for an absorbed fixed effect is a genuinely unsettled question (it is
       the same disagreement as Stata's `areg` versus `xtreg, fe`), so the report prints
       both intervals rather than picking the flattering one.
    """
    if control_lead_time:
        return _estimate(
            "event FE + lead-time control (dummies)",
            "log_q ~ log_p + lead_term + C(event_id)",
            panel,
        )
    return _estimate("event FE only (dummies)", "log_q ~ log_p + C(event_id)", panel)


# --------------------------------------------------------------------------------------
# 3. The lead-time confound: diagnostics, then a verdict from numbers
# --------------------------------------------------------------------------------------


def collinearity_diagnostics(panel: pd.DataFrame) -> dict:
    """How much price variation is left once the event and the timing are accounted for?

    Three numbers matter:

    * `residual_price_share` — 1 - R2 from regressing log price on the lead-time control
      and the event dummies. This IS the identifying variation: the share of price
      movement that is neither "which event" nor "how far out". If it is near zero, price
      and timing are the same variable and no amount of clever inference fixes that.
    * `vif_price` — 1 / residual_price_share, by definition. The same statement in the
      units people quote it in.
    * `partial_r2_price` — of the demand variation the event dummies and lead time leave
      behind, how much does price explain? This is the "is it worth anything" number.
    """
    aux_fe = smf.ols("log_p ~ C(event_id)", data=panel).fit()
    aux_full = smf.ols("log_p ~ lead_term + C(event_id)", data=panel).fit()
    aux_lead = smf.ols("lead_term ~ log_p + C(event_id)", data=panel).fit()

    residual_share = float(1.0 - aux_full.rsquared)
    restricted = smf.ols("log_q ~ lead_term + C(event_id)", data=panel).fit()
    full = smf.ols("log_q ~ log_p + lead_term + C(event_id)", data=panel).fit()
    partial_r2 = float((restricted.ssr - full.ssr) / restricted.ssr) if restricted.ssr > 0 else 0.0

    return {
        "n_rows": len(panel),
        "n_events": int(panel["event_id"].nunique()),
        "tiers_per_event": float(len(panel) / max(panel["event_id"].nunique(), 1)),
        "price_share_explained_by_event": float(aux_fe.rsquared),
        "price_share_explained_by_event_and_lead": float(aux_full.rsquared),
        "residual_price_share": residual_share,
        "vif_price": float(1.0 / residual_share) if residual_share > 0 else float("inf"),
        "vif_lead": (
            float(1.0 / (1.0 - aux_lead.rsquared)) if aux_lead.rsquared < 1 else float("inf")
        ),
        "partial_r2_price": partial_r2,
        "corr_price_lead_within_event": float(
            np.corrcoef(
                smf.ols("log_p ~ C(event_id)", data=panel).fit().resid,
                smf.ols("lead_term ~ C(event_id)", data=panel).fit().resid,
            )[0, 1]
        ),
    }


class Check(NamedTuple):
    """One numeric criterion, its value, whether it passed, and WHAT KIND of check it is.

    `kind` is the honest bit. "design" criteria are properties of the data and the
    variation in it: they can be evaluated without ever looking at the sign of beta, and
    they are what decides whether this dataset CAN answer the question. "sign" is the one
    criterion that conditions on the answer, and it is labelled so that nobody has to
    discover that for themselves.
    """

    name: str
    value: float
    threshold: float
    passed: bool
    why: str
    kind: str = "design"
    # The comparison actually used above, carried rather than guessed from the name, so
    # the report's table prints the rule a reader could reproduce by hand. The sign check
    # is strict (`ci_high < 0`) and used to render as "<= 0.000", which disagrees with the
    # code at the boundary (review finding, 2026-08-12).
    comparator: str = ">="


class Verdict(NamedTuple):
    verdict: str  # "IDENTIFIED" or "NOT-IDENTIFIED"
    checks: list[Check]
    flags: list[str]

    @property
    def identified(self) -> bool:
        return self.verdict == "IDENTIFIED"

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def identification_verdict(
    estimate: Estimate,
    diagnostics: dict,
    criteria: dict | None = None,
) -> Verdict:
    """IDENTIFIED or NOT-IDENTIFIED, from `IDENTIFICATION_CRITERIA` and nothing else.

    The point of writing this as a function with a constant is that the answer does not
    depend on how the number felt on the day. Every criterion is a comparison you can
    check by hand from the report's own diagnostics table.

    The four criteria are NOT four of the same thing, and the returned checks say which is
    which. Three are `kind="design"` — residual price variation, interval width, event
    count — and none of them looks at the estimate's sign. The fourth is `kind="sign"`:
    it requires the interval to lie below zero, which means "identified" is a label this
    function can only ever attach to a negative estimate. That is defensible (a positive
    within-event elasticity is evidence that the confound survived the fixed effects, not
    evidence that demand slopes upward) but it is not the same kind of claim, so the
    report prints it under its own heading with that argument written out.

    NOT-IDENTIFIED is not a failure of the analysis. SPEC.md 4.4: it is the honest bullet,
    and it comes with the experiment that would fix it.
    """
    c = {**IDENTIFICATION_CRITERIA, **(criteria or {})}
    checks = [
        Check(
            "residual price variation",
            float(diagnostics["residual_price_share"]),
            float(c["min_residual_price_share"]),
            diagnostics["residual_price_share"] >= c["min_residual_price_share"],
            "share of log-price variance surviving event dummies + lead time — this is "
            "the only variation beta is estimated from",
        ),
        Check(
            "95% CI width",
            float(estimate.ci_width),
            float(c["max_ci_width"]),
            estimate.ci_width <= c["max_ci_width"],
            "an interval wider than this cannot tell a harmless price rise from a ruinous one",
            comparator="<=",
        ),
        Check(
            "events with 2+ priced tiers",
            float(estimate.n_events),
            float(c["min_events"]),
            estimate.n_events >= c["min_events"],
            "the within estimator needs ladders, not events",
        ),
    ]
    if c["require_negative_and_significant"]:
        checks.append(
            Check(
                "CI lies below zero",
                float(estimate.ci_high),
                0.0,
                estimate.ci_high < 0.0,
                "a positive or zero-straddling interval means the confound is still in "
                "there; it does not mean demand slopes upward. THIS CHECK CONDITIONS ON "
                "THE SIGN and is reported separately for that reason",
                kind="sign",
                comparator="<",
            )
        )

    flags = []
    if diagnostics["vif_price"] > VIF_FLAG_LEVEL:
        flags.append(
            f"VIF(log price) = {diagnostics['vif_price']:.1f}, above the textbook "
            f"rule-of-thumb {VIF_FLAG_LEVEL:.0f}. Flagged, not failed: a VIF only says the "
            f"standard error is inflated, and whether the resulting interval is usable is "
            f"the CI-width criterion."
        )
    if diagnostics["corr_price_lead_within_event"] < -0.9:
        flags.append(
            f"within-event correlation between log price and lead time is "
            f"{diagnostics['corr_price_lead_within_event']:.3f} — price and timing are "
            f"very nearly the same variable (SPEC.md 4.3)."
        )

    verdict = "IDENTIFIED" if all(chk.passed for chk in checks) else "NOT-IDENTIFIED"
    return Verdict(verdict, checks, flags)


# --------------------------------------------------------------------------------------
# 4. The cost-shifter instrument (SPEC.md 4.5)
# --------------------------------------------------------------------------------------


class IvResult(NamedTuple):
    """One instrumental-variable specification, run or refused, with its diagnostics."""

    label: str
    instrument: str
    controls: str
    n: int
    first_stage_f: float
    first_stage_coef: float
    relevant: bool
    beta: float | None
    se: float | None
    ci_low: float | None
    ci_high: float | None
    ols_beta: float
    balance_p: float | None
    trend_share: float
    note: str


def _control_columns(controls: str, df: pd.DataFrame) -> list[str]:
    """The dataframe columns a patsy control string names: "C(venue) + C(academic_year)".

    Used to drop incomplete rows BEFORE residualising. Anything else is a silent bug: see
    `_residualise`.
    """
    return [c for c in df.columns if re.search(rf"\b{re.escape(c)}\b", controls)]


def _residualise(df: pd.DataFrame, column: str, controls: str) -> np.ndarray:
    """Partial the controls out of one column (Frisch-Waugh-Lovell).

    Everything downstream is then a one-variable regression, which is why the IV estimator
    below is three lines of arithmetic instead of a matrix identity.

    `missing="raise"` is the whole point of this line. statsmodels' default is to DROP
    incomplete rows, which returns a residual vector shorter than the frame it was asked
    about — and then `z~ . p~` is a dot product of two different sets of events, silently.
    A real export with one missing venue is all it takes (review finding, 2026-08-12).
    Callers drop their own rows explicitly first; this raises if one slips through.
    """
    frame = df.assign(_y=pd.to_numeric(df[column], errors="coerce"))
    fit = smf.ols(f"_y ~ {controls}", data=frame, missing="raise").fit()
    return np.asarray(fit.resid, dtype=float)


def iv_estimate(
    df: pd.DataFrame,
    y: str,
    p: str,
    z: str,
    controls: str,
    label: str,
    f_floor: float = FIRST_STAGE_F_FLOOR,
    proxies: tuple[str, ...] = ("log_guarantee", "log_marketing"),
) -> IvResult:
    """Just-identified instrumental-variables estimate of beta, done by hand.

    With ONE instrument and one endogenous regressor, two-stage least squares collapses to
    a ratio, and it is worth writing it that way because you can then say what it is out
    loud: **the effect of the cost shifter on quantity, divided by its effect on price.**
    If the venue's rate card pushed prices up 5% and sales fell 4%, the elasticity implied
    is -0.8. That is the whole estimator.

        1. partial the controls out of y, p and z            (Frisch-Waugh-Lovell)
        2. first stage: p~ on z~. F = t^2. If F < floor, STOP and report the refusal.
        3. beta = (z~ . y~) / (z~ . p~)
        4. robust standard error from the structural residual e = y~ - beta * p~

    Step 2's floor is not decoration. A weak instrument biases the estimate back towards
    the OLS estimate it exists to fix, so a weak-instrument IV number is worse than no
    number: it looks like a correction and behaves like the disease.

    Residuals for the standard error use the ACTUAL price, not the fitted one. Running
    two OLS regressions and reading the second one's standard errors is the classic 2SLS
    mistake — those residuals are computed against fitted prices and come out too small.
    """
    # Every column this estimate touches, including the controls: one incomplete row (a
    # missing venue on a real export) has to leave the sample HERE, once, or the three
    # residual vectors below stop describing the same events.
    frame = df.dropna(subset=[y, p, z, *_control_columns(controls, df)]).copy()
    n = len(frame)
    y_r = _residualise(frame, y, controls)
    p_r = _residualise(frame, p, controls)
    z_r = _residualise(frame, z, controls)

    # Parameters spent on the controls (including their intercept). The residualised
    # regressions below are fitted on n rows but only n - k_controls of them are free,
    # and both the F and the standard error have to know that.
    k_controls = int(smf.ols(f"{p} ~ {controls}", data=frame).fit().df_model) + 1

    first = smf.ols("p ~ z - 1", data=pd.DataFrame({"p": p_r, "z": z_r})).fit(cov_type="HC1")
    # t^2 from a no-intercept regression on residualised variables counts n observations,
    # not n - k_controls - 1, so it overstates the F. Scaling by the true df ratio is the
    # one-line correction (review finding, 2026-08-12); it can only move the F down.
    f_stat = float(first.tvalues["z"] ** 2) * max(n - k_controls - 1, 1) / n
    first_coef = float(first.params["z"])
    relevant = f_stat >= f_floor

    ols_beta = float(smf.ols(f"{y} ~ {p} + {controls}", data=frame).fit().params[p])

    beta = se = ci_low = ci_high = None
    if relevant:
        denom = float(z_r @ p_r)
        beta = float((z_r @ y_r) / denom)
        resid = y_r - beta * p_r
        # Robust (heteroskedasticity-consistent) variance for a just-identified IV.
        # Clustering is not offered here: this panel is one row per event, so every
        # cluster would have exactly one member and clustering would change nothing.
        # The n/(n-k) factor is the usual small-sample correction; k counts the control
        # regression's parameters plus the price coefficient.
        k = k_controls + 1
        var = float(np.sum((z_r**2) * (resid**2)) / denom**2) * n / max(n - k, 1)
        se = float(np.sqrt(var))
        ci_low, ci_high = beta - Z_ALPHA * se, beta + Z_ALPHA * se

    balance_p = exclusion_balance_p(frame, z, controls, proxies)
    trend_share = instrument_trend_share(frame, z)

    note = (
        f"first-stage F = {f_stat:.1f} >= {f_floor:.0f}: relevant, estimate computed"
        if relevant
        else (
            f"first-stage F = {f_stat:.1f} < {f_floor:.0f}: TOO WEAK. No estimate is "
            f"reported — a weak-instrument estimate is biased towards the OLS number it "
            f"is meant to correct"
        )
    )
    return IvResult(
        label=label,
        instrument=z,
        controls=controls,
        n=n,
        first_stage_f=f_stat,
        first_stage_coef=first_coef,
        relevant=relevant,
        beta=beta,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        ols_beta=ols_beta,
        balance_p=balance_p,
        trend_share=trend_share,
        note=note,
    )


def exclusion_balance_p(
    df: pd.DataFrame, z: str, controls: str, proxies: tuple[str, ...]
) -> float | None:
    """A NECESSARY-condition check on the exclusion restriction. Never a sufficient one.

    Exclusion cannot be tested with one instrument — that is the nature of the assumption,
    and anyone who tells you their instrument "passed the exclusion test" is describing
    something else. What CAN be checked is whether the instrument, after the controls, is
    still correlated with things we know move demand: how big the artist was (the
    guarantee is the observable proxy) and how hard the event was marketed.

    A small p-value here is evidence AGAINST exclusion. A large one is not evidence for
    it. The report says so in those words.
    """
    usable = [c for c in proxies if c in df.columns and c != z and df[c].notna().any()]
    if not usable:
        return None
    # A proxy with holes in it (a marketing spend of zero logs to NaN) would otherwise
    # give a shorter residual vector than the others and build a ragged frame.
    df = df.dropna(subset=[z, *usable])
    frame = pd.DataFrame({"z": _residualise(df, z, controls)})
    for col in usable:
        frame[col] = _residualise(df, col, controls)
    fit = smf.ols("z ~ " + " + ".join(usable), data=frame).fit()
    return float(fit.f_pvalue) if np.isfinite(fit.f_pvalue) else None


def instrument_trend_share(df: pd.DataFrame, z: str) -> float:
    """How much of the instrument (net of venue) is just the passage of time?

    SPEC.md 8.9 is the reason this exists. Hire rates drift up year on year and so does a
    growing brand's demand, so an instrument with a strong time trend is an instrument
    correlated with the growth trend — which is a demand shifter. Either control for the
    academic year (and lose the trend part of the first stage) or admit exclusion fails.
    """
    if "academic_year" not in df.columns or "venue" not in df.columns:
        return float("nan")
    df = df.dropna(subset=[z, "venue", "academic_year"])
    resid = _residualise(df, z, "C(venue)")
    frame = pd.DataFrame({"r": resid, "academic_year": df["academic_year"].to_numpy()})
    return float(smf.ols("r ~ C(academic_year)", data=frame).fit().rsquared)


def cost_shifter_specs(event_panel: pd.DataFrame) -> list[IvResult]:
    """The three instrument specifications the report walks through.

    I  — the PRE-SPECIFIED one. Venue dummies (the hire rate is a venue's rate card, so
         the level differences between venues are not variation, they are the venues) plus
         academic-year dummies, because SPEC.md 8.9 says control the trend before believing
         anything that moves with time.
    II — the same instrument WITHOUT the year control. Not a candidate. It is in the report
         because it shows what dropping the trend control buys you: a much stronger first
         stage and an estimate that fails the smell test.
    III— the artist guarantee. Fails exclusion by construction — a bigger guarantee means a
         bigger artist means more demand — and is included precisely because it has the
         strongest first stage of the three. A strong first stage proves relevance and
         says nothing at all about validity.
    """
    specs = []
    if "log_hire" in event_panel.columns:
        specs.append(
            iv_estimate(
                event_panel,
                "log_q",
                "log_p",
                "log_hire",
                "C(venue) + C(academic_year)",
                "I. venue hire rate, trend controlled (PRE-SPECIFIED)",
            )
        )
        specs.append(
            iv_estimate(
                event_panel,
                "log_q",
                "log_p",
                "log_hire",
                "C(venue)",
                "II. venue hire rate, NO trend control (diagnostic only)",
            )
        )
    if "log_guarantee" in event_panel.columns:
        specs.append(
            iv_estimate(
                event_panel,
                "log_q",
                "log_p",
                "log_guarantee",
                "C(venue) + C(academic_year)",
                "III. artist guarantee (INVALID by construction — shown deliberately)",
            )
        )
    return specs


# --------------------------------------------------------------------------------------
# 5. Run everything
# --------------------------------------------------------------------------------------


def run(run_data: phaseio.Run) -> dict:
    """Every Phase 4 number, in one dict. `report()` renders it; tests assert on it."""
    panel = build_panel(run_data.event_tier, run_data.events)
    event_panel = build_event_panel(run_data.event_tier, run_data.events)

    pooled = naive_pooled(panel)
    fe_only = event_fixed_effects(panel, control_lead_time=False)
    fe_lead = event_fixed_effects(panel, control_lead_time=True)
    # The same regression written with dummies. Same slope; a wider interval, for the
    # reason in `event_fixed_effects_dummies`. Reported alongside, never instead.
    fe_lead_dummies = event_fixed_effects_dummies(panel, control_lead_time=True)
    diagnostics = collinearity_diagnostics(panel)
    verdict = identification_verdict(fe_lead, diagnostics)
    instruments = cost_shifter_specs(event_panel)

    beta_true = None
    if run_data.truth is not None:
        beta_true = float(run_data.truth["beta_true"])

    return {
        "source": run_data.source,
        "seed": run_data.seed,
        "panel": panel,
        "event_panel": event_panel,
        "pooled": pooled,
        "fe_only": fe_only,
        "fe_lead": fe_lead,
        "fe_lead_dummies": fe_lead_dummies,
        "diagnostics": diagnostics,
        "verdict": verdict,
        "instruments": instruments,
        "beta_true": beta_true,
        # The headline elasticity other phases use, and the honest label that must travel
        # with it. Phase 6 refuses to print a point uplift when this says NOT-IDENTIFIED.
        "beta": fe_lead.beta,
        "identified": verdict.identified,
    }


# --------------------------------------------------------------------------------------
# 6. The report
# --------------------------------------------------------------------------------------


def _estimate_table(results: dict) -> str:
    rows = [
        ("1. naive pooled OLS (SPEC 4.1)", results["pooled"]),
        ("2. event FE only (SPEC 4.2)", results["fe_only"]),
        ("3. event FE + lead-time control (SPEC 4.3)", results["fe_lead"]),
        ("3b. the same, written with event dummies", results["fe_lead_dummies"]),
    ]
    out = [
        (
            "| estimator | beta | SE (clustered by event) | 95% CI (t, G-1 df) | rows | "
            "clusters | params |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for label, est in rows:
        out.append(
            f"| {label} | **{est.beta:+.3f}** | {est.se:.3f} | "
            f"[{est.ci_low:+.3f}, {est.ci_high:+.3f}] | {est.n_rows} | {est.n_events} | "
            f"{est.n_params} |"
        )
    for iv in results["instruments"]:
        beta = f"**{iv.beta:+.3f}**" if iv.beta is not None else "_not run_"
        se = f"{iv.se:.3f}" if iv.se is not None else "—"
        ci = f"[{iv.ci_low:+.3f}, {iv.ci_high:+.3f}]" if iv.beta is not None else "—"
        out.append(f"| 4. IV — {iv.label} | {beta} | {se} | {ci} | {iv.n} | {iv.n} | — |")
    if results["beta_true"] is not None:
        out.append(
            f"| **TRUE value (synthetic only)** | **{results['beta_true']:+.3f}** | — | — | — | "
            "— | — |"
        )
    return "\n".join(out)


def _criteria_table(checks: list[Check]) -> list[str]:
    lines = ["| criterion | value | threshold | pass |", "|---|---|---|---|"]
    for chk in checks:
        fmt = "{:.0f}" if chk.name.startswith("events") else "{:.3f}"
        lines.append(
            f"| {chk.name} | {fmt.format(chk.value)} | {chk.comparator} "
            f"{fmt.format(chk.threshold)} | {'PASS' if chk.passed else '**FAIL**'} |"
        )
    return lines


def _which_interval_binds(results: dict) -> str:
    """Say whether the within/dummy choice could have changed the verdict. Measured, not assumed.

    Switching estimator narrowed the interval, so the report has to answer the obvious
    question — "did that switch buy you the verdict?" — with the arithmetic rather than a
    reassurance. If both widths sit the same side of the criterion, they do not.
    """
    within, dummies = results["fe_lead"], results["fe_lead_dummies"]
    threshold = float(IDENTIFICATION_CRITERIA["max_ci_width"])
    if max(within.ci_width, dummies.ci_width) <= threshold:
        return (
            f"**The choice cannot have bought the verdict:** both widths ({within.ci_width:.3f} "
            f"and {dummies.ci_width:.3f}) are inside the {threshold:.2f} CI-width criterion, and "
            f"both intervals lie entirely below zero "
            f"({within.ci_high:+.3f} and {dummies.ci_high:+.3f}). Every criterion lands the same "
            "way on either interval."
        )
    return (
        f"**The choice DOES move the verdict, so read the wider one as the binding number:** the "
        f"two widths ({within.ci_width:.3f} and {dummies.ci_width:.3f}) fall on opposite sides of "
        f"the {threshold:.2f} CI-width criterion. The verdict above is quoted on the within "
        "interval; on the dummy interval it would not pass, and that has to be the number a "
        "reader is given."
    )


def _verdict_block(results: dict) -> str:
    verdict: Verdict = results["verdict"]
    design = [c for c in verdict.checks if c.kind == "design"]
    sign = [c for c in verdict.checks if c.kind == "sign"]

    lines = [f"## Verdict: **{verdict.verdict}**", ""]
    lines += [
        (
            "The criteria are not four of the same thing, so they are not printed as one "
            "table. Three of them are properties of the **design** — they ask whether this "
            "dataset can answer the question at all, and none of them looks at what the "
            "estimate came out as. The fourth asks about the **sign**, and that is a "
            "different kind of claim."
        ),
        "",
        "### Design criteria — can this data answer the question?",
        "",
    ]
    lines += _criteria_table(design)
    lines.append("")
    for chk in design:
        lines.append(f"- *{chk.name}* — {chk.why}")

    if sign:
        lines += [
            "",
            "### A sign check, stated as such",
            "",
        ]
        lines += _criteria_table(sign)
        lines += [
            "",
            (
                "**This criterion conditions on the sign of the answer, and that has to be said "
                "out loud.** A within-event estimate of +0.30 with a tight interval and plenty of "
                "residual price variation would be labelled NOT-IDENTIFIED by this rule, even "
                "though the design criteria above all passed. Identification is a property of the "
                "design and the variation; the sign of the estimate is not."
            ),
            "",
            (
                "The reason it is still a criterion is a specific one, and it is not 'we expected "
                "a negative number'. Demand curves slope down — that is the one thing about this "
                "problem nobody argues with, and the entire purpose of the fixed effects is to "
                "remove a confound (SPEC.md 4.1: the operator priced off the same demand signal "
                "that drove sales) whose known direction is to push the estimate UP. So a "
                "surviving positive estimate is not a discovery about demand; it is a measurement "
                "that the confound is still in there, and the honest label for a number "
                "contaminated by a confound you were trying to remove is 'not identified'."
            ),
            "",
            (
                "Why that is not the same as p-hacking the sign: p-hacking is *searching* over "
                "specifications until the sign comes out right and then reporting only the "
                "winner. Everything searched here is printed — the pooled estimate with the "
                "wrong sign is the first row of the table, the FE-only estimate that is too "
                "elastic is the second, all three instrument specifications are shown including "
                "the invalid one, and the criteria were fixed in "
                "`IDENTIFICATION_CRITERIA` before the estimate was looked at. The test is "
                "whether a failed sign check would be *reported* as NOT-IDENTIFIED rather than "
                "quietly re-specified around, and SPEC.md 4.4 plus the "
                "`NOT-IDENTIFIED` branch of this report are what commit to that."
            ),
            "",
            "\n".join(f"- *{c.name}* — {c.why}" for c in sign),
        ]
    if verdict.flags:
        lines.append("")
        lines.append("**Flags (reported, not fatal):**")
        for flag in verdict.flags:
            lines.append(f"- {flag}")
    lines.append("")
    lines.append(
        "The criteria are in `IDENTIFICATION_CRITERIA` at the top of `src/pricing/phase4.py`, "
        "and they were written before the estimate was looked at. That ordering is the whole "
        "value of them."
    )
    lines.append("")

    if verdict.identified:
        est = results["fe_lead"]
        lines += [
            "### What that means",
            "",
            (
                f"Within-event price variation identifies an elasticity of **{est.line()}**: a 10% "
                f"price rise sells about {abs(est.beta) * 10:.0f}% fewer tickets, with a plausible "
                f"range of {abs(est.ci_high) * 10:.0f}% to {abs(est.ci_low) * 10:.0f}%."
            ),
            "",
            (
                "This is identified off the price movement that is neither the event nor the timing "
                "— ladders applied differently across events, tiers opened at different lead times, "
                "prices corrected mid-window. **It is still observational.** It rests on the "
                "assumption that whatever made those ladders differ was not itself a demand signal. "
                "The randomised experiment in `experiment/DESIGN.md` is what removes that "
                "assumption, and it is running now."
            ),
        ]
    else:
        failed = ", ".join(chk.name for chk in verdict.failures)
        lines += [
            "### NOT-IDENTIFIED is the result, not a missing result",
            "",
            f"Failed criteria: **{failed}**.",
            "",
            "SPEC.md 4.4, in the spec's own words:",
            "",
            (
                "> *Documented why price elasticity is not cleanly identifiable from prices set by "
                "the operator, and specified the release-tier randomisation required to recover it.*"
            ),
            "",
            (
                "Within an event, tier price is nearly the same variable as time-to-event: Early "
                "Bird is cheap **and** early, the door price is expensive **and** last-minute. "
                "Once the timing is controlled there is not enough price movement left to measure "
                "a demand response. No amount of estimator choice fixes that; the variation is not "
                "in the data."
            ),
            "",
            (
                "**What would fix it, and is already running:** `experiment/DESIGN.md` — the autumn "
                "2026 randomised tier-pricing experiment. Whole events are assigned a HIGH (x1.15) "
                "or LOW (x0.85) ladder by a pre-committed cryptographic coin, blocked on brand and "
                "venue, with the assignment revealed only after the baseline ladder is written down. "
                "That produces price variation the operator's demand expectations did not cause, "
                "which is the only kind that identifies this."
            ),
            "",
            (
                "Until it reads out, the fixed-effects number above is a **provisional, "
                "observational estimate**, and every downstream number that uses it (Phase 6's "
                "uplift) is reported conditional on an assumed elasticity rather than as a point "
                "estimate."
            ),
        ]
    return "\n".join(lines)


def _instrument_block(results: dict) -> str:
    specs: list[IvResult] = results["instruments"]
    if not specs:
        return (
            "## 4. Cost-shifter instrument (SPEC.md 4.5)\n\n"
            "No cost-shifter column was available in `events`, so no instrument was tried.\n"
        )
    lines = [
        "## 4. Cost-shifter instrument (SPEC.md 4.5)",
        "",
        (
            "The idea: find something that moves **price** but not **demand**. Demand-side hunches "
            "move both, which is the whole problem — but the cost stack only moves price. The venue "
            "put its hire rate up; that went into the ticket price; the buyer neither knows nor "
            "cares what the venue charged us."
        ),
        "",
        (
            "This is a different regression on a different panel: one row per event, not one per "
            "tier. A cost shifter varies **across** events, so an event fixed effect would absorb it "
            "entirely. Fixed effects and instruments are two different answers to the same question, "
            "not two steps of one answer."
        ),
        "",
        (
            f"Relevance floor: first-stage **F >= {FIRST_STAGE_F_FLOOR:.0f}** on the excluded "
            "instrument. Below it, no estimate is computed at all."
        ),
        "",
        (
            "| spec | instrument | controls | first-stage F | dP/dZ | IV beta | OLS beta | "
            "balance p | trend share |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for iv in specs:
        beta = f"**{iv.beta:+.3f}**" if iv.beta is not None else "not run"
        bal = f"{iv.balance_p:.3f}" if iv.balance_p is not None else "—"
        lines.append(
            f"| {iv.label} | `{iv.instrument}` | `{iv.controls}` | {iv.first_stage_f:.1f} | "
            f"{iv.first_stage_coef:+.3f} | {beta} | {iv.ols_beta:+.3f} | {bal} | "
            f"{iv.trend_share:.3f} |"
        )
    lines += [
        "",
        (
            "`dP/dZ` is the first stage: log price per log pound of hire rate. `balance p` tests "
            "whether the instrument, after its controls, still predicts demand-side observables "
            "(the artist guarantee as a proxy for how big the act was, and marketing spend). "
            "`trend share` is the share of the instrument — net of which venue it is — explained by "
            "the academic year."
        ),
        "",
        "### Reading the exclusion restriction honestly",
        "",
        (
            "**Exclusion cannot be tested with one instrument.** That is what makes it an "
            "assumption. A small balance p-value is evidence *against* exclusion; a large one is "
            "not evidence *for* it. What can be argued is the mechanism, and here it is:"
        ),
        "",
        (
            "- **Venue hire rate — passes on the mechanism.** A buyer sees a poster with a price on "
            "it. Nothing on that poster says what the room cost us. The hire rate moves our price "
            "and reaches demand through no other channel."
        ),
        (
            "- **...but only conditional on the year.** SPEC.md 8.9: hire rates drift up year on "
            "year, and so does a growing brand's demand. An instrument with a time trend is an "
            "instrument correlated with the growth trend, and the growth trend is a demand shifter. "
            "Spec II is in the table to show exactly that — drop the year control and the first "
            "stage gets much stronger and the estimate goes the wrong way. That is not a better "
            "instrument, it is a broken one with more power."
        ),
        (
            "- **Artist guarantee — fails outright.** A bigger guarantee means a bigger artist means "
            "more people come. It moves price *and* demand. Spec III has the strongest first stage "
            "in the table and is the least usable specification on it, which is the lesson: "
            "relevance is testable and validity is not, so the strong number is the one to distrust."
        ),
    ]
    prespec = specs[0]
    if prespec.relevant:
        lines += [
            "",
            (
                f"**Outcome:** the pre-specified instrument clears the relevance floor "
                f"(F = {prespec.first_stage_f:.1f}) and gives {prespec.beta:+.3f}. Compare it with "
                f"the OLS estimate on the same panel ({prespec.ols_beta:+.3f}): the gap is what the "
                f"endogeneity was worth."
            ),
        ]
    else:
        lines += [
            "",
            (
                f"**Outcome: 2SLS is NOT reported.** The pre-specified instrument is excludable but "
                f"weak — first-stage F = {prespec.first_stage_f:.1f}, below the floor of "
                f"{FIRST_STAGE_F_FLOOR:.0f}. Controlling the year trend is what makes it excludable "
                f"and it is also what takes most of its strength away, and there is no version of "
                f"this that gets both. The scaffold stays in the repo because the moment a real "
                f"hire-rate shock lands — a venue that puts its rate card up 25% in one year — it "
                f"becomes the best identification in the project."
            ),
        ]
    return "\n".join(lines)


def report(results: dict, run_data: phaseio.Run) -> str:
    """The whole Phase 4 report as markdown."""
    d = results["diagnostics"]
    pooled, fe_only, fe_lead = results["pooled"], results["fe_only"], results["fe_lead"]
    regenerate = phaseio.regenerate_command("phase4", results["source"], results["seed"])

    parts = [
        phaseio.provenance_header(
            "Phase 4 — Price elasticity of demand", run_data.source, run_data.seed, regenerate
        ),
        "",
        "**The question:** if I raise ticket prices 10%, how many fewer tickets do I sell?",
        "",
        "**The problem:** I set the prices. SPEC.md section 4 is about nothing else.",
        "",
        "## All four estimates, including the embarrassing one",
        "",
        _estimate_table(results),
        "",
        "## 1. Why the naive regression is wrong (SPEC.md 4.1)",
        "",
        (
            f"Pooled across every tier of every event, log quantity on log price gives "
            f"**{pooled.line()}**."
        ),
        "",
        (
            "Positive. Higher prices 'cause' more sales. That is not a bug and it is not "
            "fixed by dropping variables until the sign flips — doing that is p-hacking your "
            "own business (SPEC.md 8.3). It is what you should expect from prices an operator "
            "set using the same information that drove demand: big artist, good date, "
            "freshers' week, so we charged more **and** more people came. Price is correlated "
            "with the error term. Textbook simultaneity."
            if pooled.beta > 0
            else "Negative here, which is not the same as identified — see the diagnostics "
            "below. A pooled regression on operator-set prices has no claim to identify "
            "anything; if the sign happens to come out right, that is luck about which "
            "omitted variables cancelled, not evidence."
        ),
        "",
        "## 2. Event fixed effects (SPEC.md 4.2)",
        "",
        (
            "One fixed effect per event. Everything constant within an event — the artist, the "
            "date, the venue, the hype, the marketing spend, the weather — is absorbed, and beta "
            "is left to be identified off **within-event** price variation: the tier ladder."
        ),
        "",
        (
            f"That alone gives **{fe_only.beta:+.3f}**. Adding the lead-time control gives "
            f"**{fe_lead.beta:+.3f}**. The difference between those two numbers is section 3."
        ),
        "",
        "### How the fixed effect is fitted, and why it matters for the interval",
        "",
        (
            "The regression is run as the **within transform** — subtract each event's own mean "
            "from `log_q`, `log_p` and `lead_term`, then fit with no intercept — rather than by "
            "putting a dummy per event on the right-hand side. Frisch-Waugh-Lovell says those are "
            "the same regression, and on this panel they are: rows 3 and 3b of the table above "
            f"give {fe_lead.beta:+.9f} and {results['fe_lead_dummies'].beta:+.9f}, a difference of "
            f"{abs(fe_lead.beta - results['fe_lead_dummies'].beta):.1e} — floating-point noise. "
            "There is a test that asserts it."
        ),
        "",
        (
            f"What is not the same is the parameter count, and with standard errors clustered by "
            f"event that is not cosmetic. The dummy version estimates "
            f"**{results['fe_lead_dummies'].n_params} parameters from "
            f"{fe_lead.n_events} clusters**. A cluster-robust covariance matrix is a sum of one "
            f"outer product per cluster, so its rank is at most G-1 = {fe_lead.n_events - 1} — "
            f"with {results['fe_lead_dummies'].n_params} parameters it is singular, and each "
            f"event dummy is being fitted off the "
            f"{d['tiers_per_event']:.1f} rows in its own cluster. The within version estimates "
            f"**{fe_lead.n_params} parameters from {fe_lead.n_events} clusters**, which is the "
            "footing cluster-robust inference actually assumes."
        ),
        "",
        (
            f"Both intervals are printed because the switch narrows the interval and that should "
            f"be visible rather than discovered: within gives "
            f"[{fe_lead.ci_low:+.3f}, {fe_lead.ci_high:+.3f}] (width "
            f"{fe_lead.ci_width:.3f}), dummies give "
            f"[{results['fe_lead_dummies'].ci_low:+.3f}, "
            f"{results['fe_lead_dummies'].ci_high:+.3f}] (width "
            f"{results['fe_lead_dummies'].ci_width:.3f}). The raw cluster-robust variance is "
            "identical; the whole gap is statsmodels' finite-sample factor "
            "`(N-1)/(N-K) * G/(G-1)`, which counts all the absorbed dummies in K. Whether they "
            "belong there is genuinely unsettled — it is the same disagreement as Stata's `areg` "
            "versus `xtreg, fe` — so the verdict below is quoted on the within interval and the "
            "wider one is one row above it."
        ),
        "",
        _which_interval_binds(results),
        "",
        (
            "Both intervals use a **t distribution on G-1 = "
            f"{fe_lead.n_events - 1} degrees of freedom** — critical value "
            f"{fe_lead.ci_width / (2 * fe_lead.se):.3f}, read straight off the fitted interval, "
            f"against the normal's {Z_ALPHA:.3f}. Cluster-robust inference has as many "
            "independent pieces as there are clusters, not as there are rows."
        ),
        "",
        "## 3. The lead-time confound (SPEC.md 4.3)",
        "",
        (
            "Within an event, tier price is nearly the same variable as time-to-event. Early Bird "
            "is cheap **and** early; the door price is expensive **and** the night itself. Two "
            "stories are tangled: (a) Early Bird sells more because it is cheaper — the price "
            "effect we want; (b) Early Bird sells more because eager buyers buy early and it is the "
            "only thing on sale."
        ),
        "",
        "| diagnostic | value | reading |",
        "|---|---|---|",
        f"| panel rows | {d['n_rows']} | one per (event x tier) |",
        f"| events with 2+ priced tiers | {d['n_events']} | {d['tiers_per_event']:.2f} tiers each |",
        (
            f"| price variance explained by event alone | {d['price_share_explained_by_event']:.3f} | "
            "what the fixed effects take out |"
        ),
        (
            f"| price variance explained by event + lead time | "
            f"{d['price_share_explained_by_event_and_lead']:.3f} | what is left is the estimator's "
            "entire diet |"
        ),
        (
            f"| **residual price variation** | **{d['residual_price_share']:.3f}** | the identifying "
            "variation, and nothing else is |"
        ),
        f"| VIF(log price) | {d['vif_price']:.1f} | 1 / the row above, by definition |",
        f"| VIF(lead time) | {d['vif_lead']:.1f} | the same collinearity from the other side |",
        (
            f"| within-event corr(log price, lead term) | {d['corr_price_lead_within_event']:+.3f} | "
            "-1.0 would mean price IS timing |"
        ),
        (
            f"| partial R2 of price | {d['partial_r2_price']:.3f} | of the demand variation the "
            "event and timing leave behind, how much price explains |"
        ),
        "",
        _verdict_block(results),
        "",
        _instrument_block(results),
        "",
        "## What this cannot identify, whatever the verdict says (SPEC.md 5.6)",
        "",
        (
            "- **Not the elasticity of a different price level.** Every estimate here is local to "
            "the ladders actually used. Extrapolating to a doubling of prices is not a modelling "
            "choice, it is a fabrication."
        ),
        (
            "- **Not separate elasticities per tier.** Early-bird buyers and door buyers are "
            "different people with different willingness to pay, and a single beta averages them. "
            "Splitting it would need more within-event price variation than exists."
        ),
        (
            "- **Not freshers' week versus the loan instalment versus the weather.** They arrive "
            "together every year, for five years. Nobody could separate them with ~120 events, so "
            "they are reported as one grouped start-of-semester effect and explicitly not "
            "decomposed."
        ),
        (
            "- **Not cannibalisation.** Two events a week apart eat each other's audience "
            "(SPEC.md 8.10). Clustering by event does not fix that; the observations are not "
            "independent across events either, and the standard errors above quietly assume they "
            "are."
        ),
        (
            "- **Not anything about buyers who never bought.** The panel only contains tiers that "
            "sold at least one ticket, at events that happened. Cancelled events are excluded and "
            "counted."
        ),
        "",
        "## Provisional while `fee_treatment` is UNKNOWN (SPEC.md 8.6)",
        "",
        (
            "Elasticity acts on **the price the buyer sees**, face value plus booking fee. Until the "
            "per-platform fee treatment is confirmed against a real order confirmation, `price` is a "
            "guess of the buyer's price, and every number in this report inherits that."
        ),
        "",
        "## Reproduce",
        "",
        "```",
        regenerate,
        "```",
        "",
        (
            "Companions: `docs/EXPLAINERS.md` (why these methods) and `docs/VIVA.md` (the questions "
            "this invites, with answers)."
        ),
    ]
    return "\n".join(parts) + "\n"


def summary_lines(results: dict) -> list[str]:
    """The terminal summary — the four numbers and the verdict."""
    lines = [
        f"PHASE 4  elasticity  [{results['source']}]",
        "",
        f"  naive pooled   log Q ~ log P                 {results['pooled'].beta:+.3f}"
        + ("   <-- WRONG SIGN (SPEC 4.1)" if results["pooled"].beta > 0 else ""),
        f"  event FE       + C(event_id)                 {results['fe_only'].beta:+.3f}",
        (
            f"  event FE+lead  + lead_term + C(event_id)     {results['fe_lead'].beta:+.3f}"
            f"   CI [{results['fe_lead'].ci_low:+.3f}, {results['fe_lead'].ci_high:+.3f}]"
            f"   (within: {results['fe_lead'].n_params} params, "
            f"{results['fe_lead'].n_events} clusters)"
        ),
        (
            f"    same, with event dummies                   "
            f"{results['fe_lead_dummies'].beta:+.3f}"
            f"   CI [{results['fe_lead_dummies'].ci_low:+.3f}, "
            f"{results['fe_lead_dummies'].ci_high:+.3f}]"
            f"   ({results['fe_lead_dummies'].n_params} params, "
            f"{results['fe_lead_dummies'].n_events} clusters)"
        ),
    ]
    for iv in results["instruments"]:
        got = f"{iv.beta:+.3f}" if iv.beta is not None else "not run (weak)"
        lines.append(f"  IV {iv.label[:44]:<44s} {got}  F={iv.first_stage_f:.1f}")
    if results["beta_true"] is not None:
        lines.append(f"  BETA_TRUE (synthetic)                        {results['beta_true']:+.3f}")
    lines += [
        "",
        f"  residual price variation after FE + lead     {results['diagnostics']['residual_price_share']:.3f}",
        f"  VERDICT                                      {results['verdict'].verdict}",
    ]
    for chk in results["verdict"].failures:
        lines.append(f"    FAIL {chk.name}: {chk.value:.3f} vs {chk.threshold:.3f}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = phaseio.phase_parser("pricing.phase4", "Phase 4 — price elasticity (SPEC.md 4)")
    parser.add_argument("--no-write", action="store_true", help="print it, write nothing")
    args = parser.parse_args(argv)
    try:
        run_data = phaseio.resolve(args)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    results = run(run_data)
    print("\n".join(summary_lines(results)))
    if not args.no_write:
        path = phaseio.write_report(report(results, run_data), REPORT_FILENAME, run_data.out_dir)
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
