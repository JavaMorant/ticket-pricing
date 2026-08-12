"""Synthetic ground truth: a fake five-year event programme whose demand model is KNOWN.

Why this file exists
--------------------
Every phase after 0 estimates something. An estimator you have never run on data whose
answer you already know is a guess with a p-value attached. So before the real exports
land, we manufacture a dataset where the true elasticity is a constant we chose, and we
make the dataset carry the two properties that decide whether Phase 4 is honest:

  1. **The naive pooled regression must come out WRONG-SIGNED.** SPEC.md 4.1 predicts
     beta > 0 from real data because the operator set the prices using the same
     information that drove demand. If our fixture did not reproduce that, the fixture
     would be easier than reality and any estimator would look fine on it.

  2. **The within-event regression must recover the true beta.** SPEC.md 4.2. Same data,
     same prices, one extra term (event fixed effects) — and the sign flips and the
     magnitude lands. That contrast IS the Phase 4 result, rehearsed on data where we can
     check the answer.

Those two are written as tests in tests/test_synthetic.py and they are the acceptance
tests the Phase 4 estimator will later have to pass, unchanged.

The data-generating process, in full
------------------------------------
Read this next to SPEC.md 4. Nothing here is hidden in a helper.

Per event e:

    artist_tier_e      1..4 (4 = headliner)
    hire_index_{v,y}   venue v's hire rate in academic year y — the COST SHIFTER
    shock_e            = a*(artist_tier - 2.5) + growth(brand, year) + calendar(date) + eta

    log Pbar_e         = log P0
                       + lambda * shock_e          <-- THE ENDOGENEITY. The operator
                                                       charges more when they expect more.
                       + kappa  * log hire_index   <-- cost shifter: moves price, and
                                                       appears nowhere in demand.
                       + c * (log capacity - mean) + pricing noise

Per tier t within event e (rung k of the ladder, Early Bird -> ... -> Door):

    log P_et           = log Pbar_e + step_k + jitter_et
    lead_et            = days from the tier's on-sale to the event (Early Bird is the
                         longest, Door is 0) — drawn with a random start AND random
                         spacing per event, so the same price appears at different lead
                         times across events (SPEC.md 4.3, route 2)
    log Q_et           = alpha_e + BETA_TRUE * log P_et + delta * log1p(lead_et) + eps_et

alpha_e is not written down as a parameter: it falls out of allocating the event's total
demand T_e across its tiers with a softmax of the tier terms. Anything constant within
the event — the artist, the date, the venue, the hype, the marketing spend, T_e itself —
lands inside alpha_e, which is exactly what an event fixed effect absorbs.

The event's own total responds to its average price with the SAME BETA_TRUE, so the truth
is one elasticity everywhere. The pooled regression still gets the sign wrong, because
`shock_e` is omitted from it and shock drives price (via lambda) and demand (directly).
That is textbook omitted-variable bias, not a rigged fixture.

Three traps are baked in on purpose
-----------------------------------
* **The lead-time confound (SPEC.md 4.3).** `delta > 0` and lead time is nearly collinear
  with tier position, so event fixed effects ALONE overstate the elasticity. You need the
  lead-time control too, and the only reason it is identified is `price_jitter_sd` — the
  operator does not apply the same ladder multiplier every time. Set `lead_confound` to 0
  in `params` and plain event FE recovers beta; leave it on and it does not. Tunable knob,
  because "how bad is this confound" is a question you should be able to answer with a
  number rather than a shrug.
* **The growth trend (SPEC.md 8.9).** Both brands grow year on year, at different rates.
  Anything correlated with time — a venue you started using recently, a night of the week
  you moved to — will inherit that growth unless academic year is controlled for.
* **A valid and an invalid instrument (SPEC.md 4.5).** `venue hire index` moves price and
  nothing else: valid. `artist_guarantee` moves price AND demand, because a bigger
  guarantee means a bigger artist: it fails the exclusion restriction outright. Both are
  in the fixture so Phase 4 can be shown to reject one and keep the other.

Privacy
-------
Everything here is invented. Brands are "Brand A"/"Brand B", venues are V1..V4, cities
are "City A"/"City B", and every id is prefixed FAKE. No real event, artist, venue or
buyer appears, so this file and anything derived from it are safe to commit.

Usage
-----
    uv run python -m pricing.synthetic --synthetic           summary to stdout
    uv run python -m pricing.synthetic --synthetic --write   + parquet fixtures
    uv run python -m pricing.synthetic --real                refused while the
                                                             pre-registration is DRAFT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pricing import dataset, tables

# --------------------------------------------------------------------------------------
# Where synthetic output goes
# --------------------------------------------------------------------------------------
# results/          real-data results — gitignored until the pre-publication gate
# results/synthetic/ synthetic results — committable, because none of it is real
RESULTS_DIR = dataset.REPO_ROOT / "results"
SYNTHETIC_DIR = RESULTS_DIR / "synthetic"
FIXTURES_DIR = SYNTHETIC_DIR / "fixtures"

DEFAULT_SEED = 20260811


# --------------------------------------------------------------------------------------
# 1. The knobs. Every number the generator uses is here, named, with its units.
# --------------------------------------------------------------------------------------
#
# Units: everything called a shock, a loading or a step is in LOG points, so 0.10 is
# "about 10%". Money is in pounds.
PARAMS: dict[str, float] = {
    # --- the answer we are going to try to recover ---------------------------------
    "beta_true": -0.80,  # BETA_TRUE: true price elasticity of demand
    # --- the confound (SPEC.md 4.3): tunable, set to 0.0 to switch it off ------------
    "lead_confound": 0.08,  # delta, on log1p(days the tier opened before the event)
    # --- the endogeneity (SPEC.md 4.1): why the pooled regression comes out wrong ----
    "price_shock_loading": 0.55,  # lambda: log price per 1 log point of demand shock
    "capacity_price_loading": 0.30,  # bigger room, bigger artist, higher price
    "cost_shifter_loading": 0.35,  # kappa: log price per 1 log point of hire index
    # --- demand-side variance --------------------------------------------------------
    "artist_tier_effect": 0.30,  # log demand per billing-tier step
    "sigma_shock": 0.28,  # eta: the part of the event shock nobody can predict
    "sigma_tier_noise": 0.10,  # eps: tier-level demand noise
    "base_sellthrough": 0.55,  # sell-through of a median event at the reference price
    "max_sellthrough": 0.985,  # a sellout is a sellout; caps T_e at capacity
    # --- price ladder ----------------------------------------------------------------
    "reference_price": 9.00,  # P0, buyer price incl. fee, of a median event mid-ladder
    "tier_step": 0.16,  # log price added per rung up the ladder
    "door_step": 0.30,  # extra log price on the door tier
    "price_event_noise_sd": 0.06,  # event-level pricing wobble
    "price_jitter_sd": 0.10,  # tier-level pricing wobble. THIS is what makes
    # beta identifiable once lead time is controlled:
    # price variation that is NOT lead time.
    # --- brand growth over the five years (SPEC.md 8.9 trap) -------------------------
    "brand_a_growth": 0.16,  # log demand per academic year, centred on year 3
    "brand_b_growth": 0.10,
    # --- purchase-time process (SPEC.md 6.2) -----------------------------------------
    "curve_a": 2.0,  # Beta(a, b) over the tier's window. A unimodal density
    "curve_b": 2.6,  # means an S-shaped CUMULATIVE curve, which is the
    # one-line stand-in for the Bass curve Phase 3 fits.
    "multi_ticket_share": 0.28,  # share of tickets bought on someone else's order
    # --- programme shape --------------------------------------------------------------
    "n_events": 120,
    "n_cancelled": 3,  # SPEC.md 8.4 — the events that did not happen
    "cancelled_demand_multiplier": 0.18,  # they were selling badly. That is why they went.
    "comp_event_share": 0.6,  # share of events that issue a guestlist
    "other_platform_share": 0.12,  # DECISIONS.md: ~15 of ~165 events are not Fixr
    "promo_code_share": 0.02,
}

BRANDS = ("Brand A", "Brand B")

# Venue -> (capacity, city, base hire cost). Capacities span 150..1500 as specified.
VENUES: dict[str, tuple[int, str, float]] = {
    "V1": (150, "City A", 300.0),
    "V2": (400, "City A", 700.0),
    "V3": (800, "City B", 1400.0),
    "V4": (1500, "City B", 2600.0),
}

# Two naming conventions, because real exports carry both and normalize.tier_ordinal has
# to cope with both ("Tier 1" and "Release 2" are the same rung). One convention per
# event: event_tier groups on the raw tier NAME, so mixing spellings inside one event
# would split a single tier into two panel rows.
LADDER_STYLES: tuple[tuple[str, ...], ...] = (
    ("Early Bird", "Tier 1", "Tier 2", "Tier 3"),
    ("First Release", "Release 2", "Release 3", "Release 4"),
)
DOOR_NAME = "Door"
COMP_NAMES = ("Guestlist", "Comp", "Artist Guest", "Press")

# Guarantee by artist billing tier. Correlated with demand BY CONSTRUCTION, which is
# precisely why it is not a usable instrument (SPEC.md 4.5).
GUARANTEE_BY_TIER = {1: 400.0, 2: 900.0, 3: 2000.0, 4: 4500.0}
DEAL_BY_TIER = {1: "guarantee", 2: "guarantee", 3: "door_split", 4: "guarantee_or_split"}


class SyntheticData(NamedTuple):
    """The three derived tables, in tables.py's exact schemas, plus what made them."""

    events: pd.DataFrame
    transactions: pd.DataFrame
    event_tier: pd.DataFrame
    truth: dict


# --------------------------------------------------------------------------------------
# 2. Calendar. The academic year is the spine of this business.
# --------------------------------------------------------------------------------------


def _academic_year_label(date: pd.Timestamp) -> str:
    """Academic-year label: "2023/24" for anything from Sept 2023 to Aug 2024."""
    start = date.year if date.month >= 9 else date.year - 1
    return f"{start}/{str(start + 1)[-2:]}"


def _calendar_shock(date: pd.Timestamp) -> float:
    """Log-demand effect of WHERE in the academic year the event sits.

    These are the SPEC.md 5.4 patterns, planted so Phases 1-2 have something real to find
    and so the FDR correction has real effects to keep alongside the noise. They are
    deliberately coarse and deliberately grouped: freshers' week here means "start of
    semester", the same undecomposable bundle SPEC.md 5.6 says you cannot take apart.
    """
    m, d = date.month, date.day
    if (m == 9 and d >= 18) or (m == 10 and d <= 12):
        return 0.40  # freshers / start of semester (grouped: loan drop, weather, all of it)
    if m == 12 and d <= 15:
        return 0.15  # end-of-term release
    if (m == 12 and d > 15) or (m == 1 and d <= 10):
        return -0.55  # vacation: everyone is at their parents'
    if m == 1 or (m == 2 and d <= 1):
        return -0.35  # January exams
    if m == 5 or (m == 6 and d <= 5):
        return -0.30  # summer exams
    if m == 6:
        return 0.25  # ball season
    return 0.0  # ordinary term week


DAY_OF_WEEK_SHOCK = {2: -0.15, 3: 0.0, 4: 0.10, 5: 0.15}  # Wed, Thu, Fri, Sat


def _event_dates(rng: np.random.Generator, n_events: int) -> list[pd.Timestamp]:
    """~n_events/5 dates per academic year, Sept-June, Wed-Sat, skipping deep vacation."""
    per_year = int(np.ceil(n_events / 5))
    dates: list[pd.Timestamp] = []
    for year_index in range(5):
        season_start = pd.Timestamp(year=2021 + year_index, month=9, day=15)
        # Candidate slots: every Wed/Thu/Fri/Sat in the 40 weeks after mid-September.
        slots = [
            season_start + pd.Timedelta(days=7 * w + dow - season_start.dayofweek)
            for w in range(40)
            for dow in (2, 3, 4, 5)
        ]
        slots = [s for s in slots if _calendar_shock(s) > -0.5]  # no deep-vacation events
        chosen = rng.choice(len(slots), size=per_year, replace=False)
        dates.extend(slots[i] for i in sorted(chosen))
    return sorted(dates)[:n_events]


# --------------------------------------------------------------------------------------
# 3. Events: metadata, the demand shock, the cost stack, and the operator's price
# --------------------------------------------------------------------------------------


def _build_event_truth(rng: np.random.Generator, p: dict) -> pd.DataFrame:
    """One row per event: everything the generator knows and the analyst does not.

    The columns that are NOT in the events table (demand_shock, log_hire_index,
    log_base_price) are the ground truth. They are returned in `truth`, never written to
    a derived table — that is the whole point of the exercise.
    """
    n = int(p["n_events"])
    dates = _event_dates(rng, n)

    # Venue hire rates move by academic year: a new manager, a rate card, a refit. This is
    # the cost shifter. It is drawn INDEPENDENTLY of everything on the demand side, which
    # is what makes it a candidate instrument and what the exclusion test should confirm.
    years = [f"{y}/{str(y + 1)[-2:]}" for y in range(2021, 2026)]
    hire_index: dict[tuple[str, str], float] = {}
    for venue in VENUES:
        level = 1.0
        for i, year in enumerate(years):
            level *= float(np.exp(rng.normal(0.03, 0.11)))
            if venue == "V3" and i == 3:
                level *= 1.25  # V3 put its hire rate up 25% in year 4. A real shock.
            hire_index[(venue, year)] = level

    rows = []
    for i, date in enumerate(dates):
        brand = BRANDS[i % 2] if rng.random() < 0.8 else BRANDS[(i + 1) % 2]
        # Brand A is the smaller-room brand, Brand B the bigger-room one. Venue choice is
        # correlated with brand, exactly as in a real programme.
        venue = str(rng.choice(["V1", "V2", "V3"] if brand == "Brand A" else ["V2", "V3", "V4"]))
        capacity, city, base_hire = VENUES[venue]
        year = _academic_year_label(date)
        year_index = years.index(year)

        artist_tier = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.28, 0.12]))
        # SPEC.md 8.9. Both brands grow, at different rates. Centred on the middle year so
        # growth pushes the early years down as much as the late years up: an uncentred
        # trend would park the last two years against the sell-out ceiling, where the
        # trend stops being visible in the data that is supposed to demonstrate it.
        rate = p["brand_a_growth"] if brand == "Brand A" else p["brand_b_growth"]
        growth = rate * (year_index - 2)

        # THE DEMAND SHOCK. Everything the operator can see when pricing, plus the part
        # nobody can see. Both go into demand; only the visible parts should really go
        # into price, but lambda below applies to the whole thing — an operator's pricing
        # hunch is correlated with the unobservable too, which is the harder, truer case.
        shock = (
            p["artist_tier_effect"] * (artist_tier - 2.5)
            + growth
            + _calendar_shock(date)
            + DAY_OF_WEEK_SHOCK[date.dayofweek]
            + rng.normal(0.0, p["sigma_shock"])
        )

        log_hire = float(np.log(hire_index[(venue, year)]))
        mean_log_capacity = float(np.mean(np.log([c for c, _, _ in VENUES.values()])))
        log_base_price = (
            np.log(p["reference_price"])
            + p["price_shock_loading"] * shock
            + p["cost_shifter_loading"] * log_hire
            + p["capacity_price_loading"] * (np.log(capacity) - mean_log_capacity)
            + rng.normal(0.0, p["price_event_noise_sd"])
        )

        guarantee = GUARANTEE_BY_TIER[artist_tier] * (capacity / 400) ** 0.4
        guarantee *= float(np.exp(rng.normal(0.0, 0.15)))

        rows.append(
            {
                "event_key": f"FAKE-EV-{i + 1:03d}",
                "event_name": f"FAKE {brand} Night {i + 1:03d}",
                "event_date": date.strftime("%Y-%m-%d"),
                "academic_year": year,
                "year_index": year_index,
                "brand": brand,
                "venue": venue,
                "city": city,
                "capacity": capacity,
                "platform": "fixr",  # overwritten below for a minority of events
                "artist_tier": artist_tier,
                "demand_shock": shock,
                "calendar_shock": _calendar_shock(date),
                "brand_growth": growth,
                "log_hire_index": log_hire,
                "log_base_price": float(log_base_price),
                "artist_guarantee": round(guarantee, 2),
                "deal_structure": DEAL_BY_TIER[artist_tier],
                "venue_cost": round(base_hire * hire_index[(venue, year)], 2),
                "security_cost": round(150 + 0.9 * capacity * np.exp(rng.normal(0, 0.10)), 2),
                "production_cost": round(200 + 0.6 * capacity * np.exp(rng.normal(0, 0.15)), 2),
                "staffing_cost": round(100 + 0.4 * capacity * np.exp(rng.normal(0, 0.10)), 2),
                # Marketing spend tracks expected demand: the operator spends more on the
                # events they already believe in. Another endogenous regressor, and a
                # reason Phase 2 must not read it as "marketing causes sales".
                "marketing_spend": round(
                    (150 + 1.2 * capacity) * np.exp(0.30 * shock + rng.normal(0, 0.20)), 2
                ),
                "cancelled": False,
            }
        )

    truth = pd.DataFrame(rows)

    # A minority of events sold on TBC.xyz / Youni rather than Fixr (DECISIONS.md).
    other = rng.random(len(truth)) < p["other_platform_share"]
    truth.loc[other, "platform"] = rng.choice(["tbc", "youni"], size=int(other.sum()))

    # SPEC.md 8.4: a few events were pulled. They are in the data, flagged, with the thin
    # presale that got them pulled. Every downstream phase has to decide what to do with
    # them, which is the point of putting them here.
    cancelled = rng.choice(len(truth), size=int(p["n_cancelled"]), replace=False)
    truth.loc[cancelled, "cancelled"] = True

    return truth


def _costs_frame(event_truth: pd.DataFrame) -> pd.DataFrame:
    """Only the canonical cost columns — the shape adapters.adapt_costs would produce."""
    from pricing.adapters import COST_FIELDS

    return event_truth[list(COST_FIELDS)].copy()


# --------------------------------------------------------------------------------------
# 4. Tier ladders: prices, on-sale schedule, and the true quantity
# --------------------------------------------------------------------------------------


def _ladder(rng: np.random.Generator, p: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """One event's ladder: names, log-price offsets from the event's base, on-sale leads.

    Offsets are centred so the event's base price stays the geometric mean of the ladder.

    The leads are the interesting part. The first rung opens 45-95 days out and the last
    advance rung opens 3-20 days out, with the gaps in between drawn at random — so "Tier
    2" is 40 days out at one event and 12 days out at another. That cross-event variation
    in tier TIMING is SPEC.md 4.3 route 2, and without it there is no hope of separating
    the price effect from the timing effect.
    """
    n_rungs = int(rng.choice([2, 3, 4, 5], p=[0.15, 0.35, 0.35, 0.15]))
    has_door = bool(rng.random() < 0.80)
    style = LADDER_STYLES[int(rng.integers(len(LADDER_STYLES)))]
    # A door-less event spends all its rungs on advance releases, so the ladder can want
    # more advance names than a style provides. Clamp to what exists rather than silently
    # returning a short list of names against a long list of offsets.
    n_advance = min(max(1, n_rungs - 1 if has_door else n_rungs), len(style))

    names = list(style[:n_advance])
    offsets = [p["tier_step"] * k for k in range(n_advance)]
    if has_door:
        names.append(DOOR_NAME)
        offsets.append(p["tier_step"] * (n_advance - 1) + p["door_step"])

    # On-sale leads, in days before the event.
    first_lead = float(rng.uniform(45, 95))
    last_lead = float(rng.uniform(3, 20))
    if n_advance > 1:
        gaps = rng.uniform(0.5, 1.5, n_advance - 1)
        gaps = gaps / gaps.sum() * (first_lead - last_lead)
        leads = first_lead - np.concatenate([[0.0], np.cumsum(gaps)])
    else:
        leads = np.array([first_lead])
    leads = np.round(leads).astype(int)
    for k in range(1, len(leads)):  # rounding must never create a tie
        leads[k] = min(leads[k], leads[k - 1] - 1)
    if has_door:
        leads = np.append(leads, 0)  # the door opens on the night

    offsets = np.asarray(offsets, dtype=float)
    return names, offsets - offsets.mean(), leads


def _round_price(target_buyer_price: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a target buyer price into (face value, booking fee), both realistically round.

    Order matters. Real face values are round numbers, so rounding happens FIRST and the
    quantity is generated from the price that results. Generating Q from an unrounded
    price and then reporting a rounded one would be classical measurement error in the
    regressor, which attenuates beta towards zero — the fixture would then fail its own
    recovery test for a reason that has nothing to do with the estimator.
    """
    face = np.round(target_buyer_price / 1.10 * 2) / 2  # nearest 50p
    face = np.clip(face, 4.0, 22.5)  # keeps the buyer price inside £4.00 .. £24.75
    fee = np.round(0.10 * face, 2)
    return face, fee


def _tier_table(rng: np.random.Generator, event_truth: pd.DataFrame, p: dict) -> pd.DataFrame:
    """One row per (event x tier): the price, the on-sale lead, and the TRUE units."""
    beta = p["beta_true"]
    delta = p["lead_confound"]
    rows = []

    for event in event_truth.itertuples():
        names, offsets, leads = _ladder(rng, p)
        jitter = rng.normal(0.0, p["price_jitter_sd"], len(names))
        target = np.exp(event.log_base_price + offsets + jitter)
        face, fee = _round_price(target)
        buyer_price = face + fee  # this, and only this, is what demand responds to

        lead_term = np.log1p(leads)
        noise = rng.normal(0.0, p["sigma_tier_noise"], len(names))

        # Tier-level demand terms. The softmax below turns them into shares, and the
        # event's total turns the shares into units. Everything the softmax divides out is
        # constant within the event, so it lands in alpha_e and an event fixed effect eats
        # it whole -- which is why this construction still satisfies
        #     log Q_et = alpha_e + beta*log P_et + delta*log1p(lead_et) + eps_et
        # exactly, and why the within-event estimator can recover beta.
        v = beta * np.log(buyer_price) + delta * lead_term + noise
        share = np.exp(v - v.max())
        share = share / share.sum()

        # Event total. It responds to the event's AVERAGE price with the same beta, so
        # there is one true elasticity in this world, not two.
        mean_log_price = float(np.mean(np.log(buyer_price)))
        sell = p["base_sellthrough"] * np.exp(
            event.demand_shock + beta * (mean_log_price - np.log(p["reference_price"]))
        )
        if event.cancelled:
            sell *= p["cancelled_demand_multiplier"]
        total = event.capacity * min(sell, p["max_sellthrough"])

        units = np.maximum(3, np.round(total * share)).astype(int)

        for k, name in enumerate(names):
            rows.append(
                {
                    "event_key": event.event_key,
                    "event_date": pd.Timestamp(event.event_date),
                    "tier_name": name,
                    "rung": k,
                    "is_door": name == DOOR_NAME,
                    "face_price": float(face[k]),
                    "booking_fee": float(fee[k]),
                    "buyer_price": float(buyer_price[k]),
                    "lead_open_days": int(leads[k]),
                    "units": int(units[k]),
                    "platform": event.platform,
                    "log_price_true": float(np.log(buyer_price[k])),
                    "lead_term_true": float(lead_term[k]),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 5. Transactions: the S-shaped purchase-time process
# --------------------------------------------------------------------------------------


def _window(row) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(open, close) timestamps for one tier.

    Advance tiers go on sale at 10:00 on their lead day and come off when the next rung
    opens; the last advance rung runs until 18:00 on the night. The door tier is the night
    itself, 20:00 to 23:40 — kept before midnight on purpose, because
    tables._lead_time_days floors both sides to the DATE and a 00:30 sale would land on
    the day after the event.
    """
    event_date = row.event_date
    if row.is_door:
        return event_date + pd.Timedelta(hours=20), event_date + pd.Timedelta(hours=23, minutes=40)
    open_ts = event_date - pd.Timedelta(days=int(row.lead_open_days)) + pd.Timedelta(hours=10)
    close_ts = event_date - pd.Timedelta(days=int(row.close_lead_days)) + pd.Timedelta(hours=10)
    if row.close_lead_days == 0:
        close_ts = event_date + pd.Timedelta(hours=18)
    return open_ts, close_ts


def _purchase_times(
    rng: np.random.Generator, n: int, open_ts: pd.Timestamp, close_ts: pd.Timestamp, p: dict
) -> np.ndarray:
    """n purchase timestamps across a tier's window, S-shaped in cumulative terms.

    A Beta(a, b) density over the window is unimodal, so its cumulative is the S the sales
    curve actually has: slow at first, steepest in the middle, tapering at the close. It
    is a one-liner and Phase 3 will fit a proper diffusion curve to it.

    The first ticket is pinned to the instant the tier opens. That is realistic (there is
    always an on-sale burst) and it is load-bearing: event_tier.window_open is the FIRST
    SALE, so pinning it makes lead_time_open_days exactly the scheduled lead rather than
    a noisy version of it. Noise there would leak into the lead-time control.
    """
    span = (close_ts - open_ts).total_seconds()
    frac = np.sort(rng.beta(p["curve_a"], p["curve_b"], size=n))
    frac[0] = 0.0
    return open_ts + pd.to_timedelta(frac * span, unit="s")


def _transactions_frame(
    rng: np.random.Generator, tier_table: pd.DataFrame, event_truth: pd.DataFrame, p: dict
) -> pd.DataFrame:
    """Per-ticket rows in canonical columns — the shape an adapter would hand over."""
    # Each advance tier closes when the next rung opens. Computed per event, in ladder
    # order, with the last advance rung running to the night (close lead 0).
    tier_table = tier_table.sort_values(["event_key", "rung"]).copy()
    next_lead = tier_table.groupby("event_key")["lead_open_days"].shift(-1)
    tier_table["close_lead_days"] = next_lead.fillna(0).astype(int)

    comp_events = set(
        event_truth.loc[rng.random(len(event_truth)) < p["comp_event_share"], "event_key"]
    )

    parts: list[pd.DataFrame] = []
    for row in tier_table.itertuples():
        n = int(row.units)
        open_ts, close_ts = _window(row)
        times = pd.Series(_purchase_times(rng, n, open_ts, close_ts, p))

        # Orders: a per-ticket export repeats order_id when one order holds several
        # tickets, and those tickets share a timestamp. Every row is one ticket, so
        # quantity stays 1 and units_sold is the row count.
        starts_order = rng.random(n) > p["multi_ticket_share"]
        starts_order[0] = True
        order_num = np.cumsum(starts_order)
        times = times.groupby(order_num).transform("first")

        parts.append(
            pd.DataFrame(
                {
                    "order_id": [f"FAKE-ORD-{row.event_key[-3:]}-{o:05d}" for o in order_num],
                    "event_key": row.event_key,
                    "tier_name": row.tier_name,
                    "price_paid": row.face_price,
                    "booking_fee": row.booking_fee,
                    # Microsecond resolution, matching what pd.to_datetime gives a real
                    # export's timestamp strings — so the fixture's dtypes are identical
                    # to the ones the live pipeline will produce, not merely similar.
                    "purchased_at": times.to_numpy().astype("datetime64[us]"),
                    "quantity": 1,
                    "promo_code": None,  # filled in once, after the concat below
                    "platform": row.platform,
                }
            )
        )

    # Comps and guestlist (SPEC.md 8.7). Price 0, so normalize.add_comp_flag catches them
    # by price as well as by name, and build_event_tier keeps them out of units_sold.
    for event in event_truth.itertuples():
        if event.event_key not in comp_events:
            continue
        n = int(rng.integers(2, 16))
        event_date = pd.Timestamp(event.event_date)
        parts.append(
            pd.DataFrame(
                {
                    "order_id": [f"FAKE-ORD-{event.event_key[-3:]}-C{i:04d}" for i in range(n)],
                    "event_key": event.event_key,
                    "tier_name": str(rng.choice(COMP_NAMES)),
                    "price_paid": 0.0,
                    "booking_fee": 0.0,
                    "purchased_at": (
                        event_date
                        - pd.to_timedelta(rng.integers(0, 7, n), unit="D")
                        + pd.Timedelta(hours=12)
                    )
                    .to_numpy()
                    .astype("datetime64[us]"),
                    "quantity": 1,
                    "promo_code": None,
                    "platform": event.platform,
                }
            )
        )

    out = pd.concat(parts, ignore_index=True)

    # Promo codes last, in one pass over the whole frame. Assigning a plain list of
    # str-or-None lets pandas infer the column's dtype exactly as it would when reading a
    # real export, which is what keeps the fixture's schema identical to the live one
    # rather than merely similar. Comps never carry a code — they were never bought.
    # SPEC.md 4.3: on real data these codes are gold, because a discount given for a
    # reason unrelated to demand is price variation you did not choose.
    coded = (rng.random(len(out)) < p["promo_code_share"]) & (out["price_paid"] > 0)
    out["promo_code"] = ["FAKEPROMO" if c else None for c in coded]
    return out


# --------------------------------------------------------------------------------------
# 6. generate()
# --------------------------------------------------------------------------------------


def generate(seed: int = DEFAULT_SEED, params: dict | None = None) -> SyntheticData:
    """Build the three derived tables plus the ground truth that produced them.

    The tables are built by calling tables.build_events / build_transactions /
    build_event_tier on canonical raw frames — the same functions the real ingest uses.
    That is deliberate: it makes schema equality with the real pipeline true by
    construction rather than by a list of column names kept in sync by hand.

    `params` overrides any key in PARAMS, e.g. generate(1, {"lead_confound": 0.0}).
    """
    p = {**PARAMS, **(params or {})}
    rng = np.random.default_rng(seed)

    event_truth = _build_event_truth(rng, p)
    tier_truth = _tier_table(rng, event_truth, p)
    tx_raw = _transactions_frame(rng, tier_truth, event_truth, p)

    events = tables.build_events(_costs_frame(event_truth))
    transactions = tables.build_transactions(tx_raw, events)
    event_tier = tables.build_event_tier(transactions)

    truth = {
        "seed": seed,
        "beta_true": p["beta_true"],
        "params": p,
        # One row per event: the demand shock and the operator's base price. NEITHER is in
        # the events table, and that asymmetry is the experiment.
        "event_truth": event_truth,
        # One row per (event x tier): the true log price, lead term and units.
        "tier_truth": tier_truth,
        "instruments": {
            # Moves price, absent from demand. The one to try in 2SLS.
            "valid": ["log_hire_index (events.venue_cost / base hire for the venue)"],
            # Moves price AND demand: a bigger guarantee means a bigger artist means more
            # people come. Fails the exclusion restriction by construction (SPEC.md 4.5).
            "invalid": ["artist_guarantee", "marketing_spend"],
        },
        "expected": {
            # The two acceptance properties. Asserted in tests/test_synthetic.py.
            "naive_pooled_beta": "> 0 (wrong sign — SPEC.md 4.1)",
            "within_event_beta_with_lead_control": p["beta_true"],
            "within_event_beta_without_lead_control": (
                "more negative than beta_true while lead_confound > 0 (SPEC.md 4.3)"
            ),
        },
    }
    return SyntheticData(events, transactions, event_tier, truth)


# --------------------------------------------------------------------------------------
# 7. The two acceptance checks
# --------------------------------------------------------------------------------------
#
# These are NOT the Phase 4 estimator. They are the minimum regressions needed to state
# the two properties as tests, kept here so the fixture ships with its own definition of
# correct. Phase 4 will write the real thing (clustered standard errors, IV, diagnostics)
# and must reproduce both numbers on this fixture before it is allowed near real data.


def analysis_panel(event_tier: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """The event_tier rows that can carry a regression, plus log columns.

    Four filters, each with a reason:
      * cancelled events out          — SPEC.md 8.4, they did not happen
      * comp-only tiers out           — SPEC.md 8.7, comps are not demand (price is null)
      * zero-sale tiers out           — log(0)
      * single-tier events out        — an event fixed effect with one observation fits it
                                        perfectly and contributes nothing to beta
    """
    df = event_tier.copy()
    if events is not None and "cancelled" in events.columns:
        df = df.merge(events[["event_id", "cancelled"]], on="event_id", how="left")
        df = df[~df["cancelled"].fillna(False)].drop(columns=["cancelled"])
    df = df[(df["units_sold"] > 0) & (df["price"] > 0)].copy()
    df["log_q"] = np.log(df["units_sold"])
    df["log_p"] = np.log(df["price"])
    df["lead_term"] = np.log1p(df["lead_time_open_days"].clip(lower=0))
    return df[df.groupby("event_id")["event_id"].transform("size") >= 2].reset_index(drop=True)


def naive_pooled_beta(panel: pd.DataFrame) -> float:
    """SPEC.md 4.1: log Q on log P, pooled across events. Expect the WRONG sign."""
    return float(smf.ols("log_q ~ log_p", data=panel).fit().params["log_p"])


def within_event_beta(panel: pd.DataFrame, control_lead_time: bool = True) -> float:
    """SPEC.md 4.2: the same regression plus a dummy per event.

    `C(event_id)` is one dummy per event — literally alpha_e from the spec, and the
    easiest version to explain out loud. With ~120 events and ~400 rows it costs
    milliseconds; the within-transform (demean every column by event) gives the identical
    slope and is the thing to switch to only if that ever stops being true.

    `control_lead_time=True` adds log1p(days the tier opened before the event). Leave it
    off and the estimate absorbs the timing effect too, because early tiers are cheap AND
    early (SPEC.md 4.3).
    """
    formula = "log_q ~ log_p + C(event_id)"
    if control_lead_time:
        formula = "log_q ~ log_p + lead_term + C(event_id)"
    return float(smf.ols(formula, data=panel).fit().params["log_p"])


# --------------------------------------------------------------------------------------
# 8. CLI
# --------------------------------------------------------------------------------------


def write_fixtures(data: SyntheticData, out_dir: Path = FIXTURES_DIR) -> list[Path]:
    """Write the three tables to parquet, plus the scalar ground truth as JSON.

    Uses tables.write_table, so the fail-closed PII allowlist runs on synthetic data too —
    if the generator ever grows a column the real pipeline would refuse, we find out here.

    The DataFrames inside `truth` are NOT written: they are regenerable from the seed in
    one call, and a file that looks like a data export is exactly what .gitignore's
    repo-wide *.parquet rule is there to stop drifting into git.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        tables.write_table(data.events, "events", out_dir=out_dir),
        tables.write_table(data.transactions, "transactions", out_dir=out_dir),
        tables.write_table(data.event_tier, "event_tier", out_dir=out_dir),
    ]
    scalars = {k: v for k, v in data.truth.items() if not isinstance(v, pd.DataFrame)}
    json_path = out_dir / "ground_truth.json"
    json_path.write_text(json.dumps(scalars, indent=2, default=str))
    paths.append(json_path.resolve())
    return paths


def summary(data: SyntheticData) -> str:
    """What was generated, and the two properties that make it worth generating."""
    events, transactions, event_tier, truth = data
    panel = analysis_panel(event_tier, events)
    pooled = naive_pooled_beta(panel)
    within = within_event_beta(panel, control_lead_time=True)
    within_naive = within_event_beta(panel, control_lead_time=False)

    span = f"{events['event_date'].min().date()} .. {events['event_date'].max().date()}"
    cancelled = int(events["cancelled"].sum())
    comps = int(transactions["is_comp"].sum())
    prices = f"£{event_tier['price'].min():.2f} .. £{event_tier['price'].max():.2f}"
    rooms = f"{events['capacity'].min()} .. {events['capacity'].max()}"

    lines = [
        f"SYNTHETIC FIXTURE  seed={truth['seed']}  (nothing here is real)",
        "",
        f"  events        {len(events):>7,}   {span}   ({cancelled} cancelled)",
        f"  transactions  {len(transactions):>7,}   {comps:,} comps",
        f"  event_tier    {len(event_tier):>7,}   {len(panel):,} usable panel rows",
        f"  buyer price   {prices}",
        f"  capacity      {rooms}   venues {sorted(events['venue'].unique())}",
        f"  brands        {sorted(events['brand'].unique())}",
        "",
        f"  BETA_TRUE                                     {truth['beta_true']:+.3f}",
        (
            f"  naive pooled      log Q ~ log P               {pooled:+.3f}"
            "   <-- WRONG SIGN if positive (SPEC 4.1)"
        ),
        (
            f"  event FE only     + C(event_id)               {within_naive:+.3f}"
            "   <-- lead-time confound (SPEC 4.3)"
        ),
        (
            f"  event FE + lead   + lead_term + C(event_id)   {within:+.3f}"
            "   <-- recovers BETA_TRUE (SPEC 4.2)"
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pricing.synthetic", description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="use the seeded fixture")
    parser.add_argument("--real", action="store_true", help="use data/derived/ (currently refused)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--write", action="store_true", help=f"write parquet to {FIXTURES_DIR}")
    args = parser.parse_args(argv)

    if args.real:
        try:
            dataset.load_real()
        except (RuntimeError, FileNotFoundError) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        print("real tables loaded — but this module only generates synthetic data.")
        return 0

    data = generate(seed=args.seed)
    print(summary(data))
    if args.write:
        print()
        for path in write_fixtures(data):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
