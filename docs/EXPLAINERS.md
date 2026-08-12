# EXPLAINERS

One section per phase. Plain English: what the method is, why we chose it, and what can
go wrong with it. Written so that reading this file and nothing else is enough to defend
the repo out loud.

Companion: `docs/VIVA.md` — the same material as questions you should be able to answer
cold. Nothing here is a number from real data; the real numbers arrive after
`preregistration.md` is frozen and the exports land.

---

## Phase 0.5 — Synthetic ground truth (`src/pricing/synthetic.py`)

### What it is

A fake five-year event programme: 120 events, two brands, four venues, capacities 150 to
1,500, tier ladders of two to five rungs from Early Bird to Door, prices from £4.40 to
£24.75, ~34,000 ticket transactions, guestlist comps, three cancelled events, and a full
cost stack per event. Everything is invented — "Brand A", "V1", "FAKE-EV-014" — so it is
safe to commit and safe to publish.

The point is not the data. The point is that **we chose the answer before we generated
it.** The true price elasticity of demand is a constant in the file:

```
BETA_TRUE = -0.80        # a 10% price rise sells about 8% fewer tickets
```

Every later phase can therefore be tested rather than believed.

### Why we built it before anything else

Two reasons, and they are both about honesty rather than convenience.

**1. The real data is not allowed in yet.** `preregistration.md` is still marked DRAFT.
SPEC.md §5.3 says the pattern list is written from operating memory *before first data
contact*, because a list written after you have seen the data is not a pre-registration,
it is a memory of what looked significant. So the real path is wired
(`pricing/dataset.py: load_real()`) and closed: it reads the Status line of
`preregistration.md` and refuses, printing why. Meanwhile every phase can be built,
tested and debugged end to end on the fixture.

**2. An untested estimator is a guess with a p-value attached.** Phase 4 is going to
produce a number that goes on a CV. The only way to know the estimator works is to run it
on data whose answer you already know. That is what this file is for.

### The one thing it has to reproduce

SPEC.md §4.1 predicts that regressing log quantity on log price across the real events
will give a **positive** coefficient — higher prices "causing" more sales. Not a bug: the
operator set those prices using the same information that drove demand. Big artist, good
date, freshers' week → charged more *and* more people came.

So the generator does exactly that. Each event gets a demand shock (artist billing tier,
brand growth, where it sits in the academic year, plus noise), and the operator's price
loads on that shock with a coefficient `lambda = 0.55`. Nothing else is needed. The three
regressions on the fixture:

```
BETA_TRUE                                     -0.800
naive pooled      log Q ~ log P               +0.698   <-- wrong sign (SPEC 4.1)
event FE only     + C(event_id)               -1.260   <-- lead-time confound (SPEC 4.3)
event FE + lead   + lead_term + C(event_id)   -0.815   <-- recovers BETA_TRUE (SPEC 4.2)
```

That middle line matters as much as the last. **Event fixed effects alone are not
enough.** Early Bird is cheap *and* early; the door price is expensive *and* last-minute.
Fixed effects remove everything constant within the event, but the timing effect varies
*within* the event exactly as price does, so the FE estimate charges the timing effect to
the price and comes out 57% too elastic. Only when lead time is controlled does the
estimate land on the truth.

### How it is built (the whole thing, in order)

Per event: draw a date in the academic year, a brand, a venue, an artist billing tier,
and a demand shock. Look up the venue's hire-rate index for that academic year. Set the
event's base price from the shock, the hire index and the room size. Draw a cost stack.

Per tier within the event: walk up the ladder with a log price step per rung and a jitter
term; give each rung an on-sale date drawn with a random start (45–95 days out) and random
spacing; then

```
log Q_et = alpha_e + BETA_TRUE * log P_et + delta * log1p(lead_et) + eps_et
```

`alpha_e` is never written down as a parameter. The tier terms go through a softmax to
become shares of the event's total demand, and the event total is set by the demand shock
and the event's average price. Everything that division removes is constant within the
event, so it lands in `alpha_e` — which is precisely what an event dummy absorbs. This is
why the construction and the estimating equation agree exactly.

Per ticket: a purchase time drawn from a Beta density across the tier's window. A
unimodal density means an **S-shaped cumulative** curve, which is what ticket sales
actually look like and what Phase 3 will fit a diffusion curve to. The first ticket of
every tier is pinned to the on-sale instant, so `window_open` in the panel is the
scheduled on-sale and the lead-time control is measured without noise.

The three tables are then built by calling `tables.build_events`,
`build_transactions` and `build_event_tier` — the same functions the real ingest calls.
Schema equality with the live pipeline is therefore true by construction, and the tests
assert it column for column and dtype for dtype.

### What else is planted

| Trap | Where | Why |
|---|---|---|
| Lead-time confound (§4.3) | `lead_confound = 0.08` | So nobody concludes event FE is sufficient. Tunable: set it to 0 and plain FE recovers beta. |
| Brand growth trend (§8.9) | `brand_a_growth = 0.16`, `brand_b_growth = 0.10` per year | Anything correlated with time inherits it. Control for academic year before believing a calendar effect. |
| A valid instrument (§4.5) | venue hire index, moves ±10–25% a year | Moves price, absent from demand. The 2SLS candidate. |
| An invalid instrument (§4.5) | `artist_guarantee` | Bigger guarantee → bigger artist → more demand. Fails exclusion by construction. |
| Endogenous marketing | `marketing_spend` loads on the shock | The operator spends more on events they already believe in. Not "marketing causes sales". |
| Survivorship (§8.4) | 3 cancelled events, thin presale | They are in the data, flagged. Every phase has to decide what to do with them. |
| Comps (§8.7) | ~60% of events issue 2–15 | Flagged, excluded from `units_sold`, counted in `units_comp`. |
| Face vs fee (§8.6) | `fee_treatment = "UNKNOWN"` on every row | The platform treatment is unresolved, and the fixture carries that unresolvedness. |

### What can go wrong with this method

**The fixture can be easier than reality.** It is a model, and the estimator was written
against the same model. If real prices move for reasons this generator does not contain —
a competitor's night, a promo code, a mid-window correction — the estimator may pass here
and fail there. Passing these tests means the estimator is not broken; it does not mean
the real elasticity is identified.

**Identification is baked in.** `price_jitter_sd = 0.10` is the operator not applying the
same ladder multiplier every time, and it is the only price variation orthogonal to lead
time. It is what makes the fixed-effects-plus-lead-time regression identified at all. If
the real data has no such jitter — if every event used the identical ladder on the
identical schedule — then price and lead time are perfectly collinear and the answer is
SPEC.md §4.4: **say it is not identified and specify the experiment that would fix it.**
The knob exists so that "how much residual price variation do I need?" is a question with
a number for an answer, not a shrug.

**The generator is not a forecast.** Nothing here is calibrated to real Brand A or Brand B
performance. The parameters were chosen to make the statistical properties visible, not to
predict anything. No number from this file belongs in a CV bullet, an interview answer, or
a pricing decision.

**Small-sample honesty applies to the fixture too.** 120 events give ~400 usable panel
rows, and the recovered beta wanders by about ±0.08 across seeds. The tests allow ±0.15.
That spread is a real feature of the sample size, not sloppiness — and it is the same
spread the real estimate will have.

**Rounding and censoring are present and mild.** Face values are rounded to 50p and clipped
to £4.00–£22.50, which binds on about 2.5% of tiers at each end; sell-through is capped at
98.5% so sellouts exist. Both are realistic. Both attenuate slightly. Neither is large
enough to move the recovered beta outside tolerance, and the capacity cap is absorbed
entirely by the event fixed effect.

### The guard on real data

`pricing/dataset.py` refuses to read `data/derived/` while `preregistration.md` says
DRAFT, and says why:

```
refused: real data is OFF LIMITS: preregistration.md is DRAFT.
  why : SPEC.md 5.3 — the pattern list is written from operating memory BEFORE
        first data contact...
  fix : edit the Status line in preregistration.md to say FROZEN once the list
        is final, then re-run.
```

It is a tripwire, not a security control — one word in a markdown file opens it. That is
the point: the freeze is a human act, and this only records it. `dataset.FEATURES` holds
the pre-registered pattern list as slugs, with a comment saying its final contents come
from the frozen file and not from the code.

---

## Phase 1 — Descriptives (`src/pricing/phase1.py`)

### What it is

Counts, distributions and shapes, and nothing else. Events per brand, venue and academic
year; the distribution of capacity sell-through; revenue per event; the shape of the tier
ladder; comp rates; the sellout rate. Five figures and a markdown report, regenerated by

```
uv run python -m pricing.phase1 --synthetic
```

On the synthetic fixture: 117 events that ran plus 3 that were pulled, 32,999 paid tickets,
579 comps, 402 paid tiers, median sell-through 51%, sellout rate 3.4%, median revenue per
event about £1,900, ladders of two to five rungs with the top rung typically 1.6x the
opening price and the first rung opening around 69 days out.

### Why look before you model

Two reasons, and only one of them is "sanity checking".

The obvious one: half the mistakes in this project are arithmetic, not statistics. If comps
are being counted as demand, or the tier ordinal is upside down, or three events are
duplicated by a bad join, every model downstream is wrong in a way no cross-validation will
reveal — because the model will happily learn the broken thing. Looking at the marginal
distributions catches that in ten minutes.

The less obvious one, and the reason this phase exists as its own numbered step in the
spec: **the descriptives decide what is worth modelling at all.** Ladder span is the
example. Phase 4 can only identify a price elasticity if tier price and lead time are not
the same variable, and the only thing that separates them is whether the ladder timing
varied across events (SPEC.md §4.3). That is a descriptive fact. It sits in the "ladder
span per event" table, and if that table had shown every event opening on the same schedule
the honest move would be to skip Phase 4 and write the §4.4 paragraph instead.

### The thing this phase deliberately refuses to do

**Nothing here is tested.** No p-value, no multiple-comparison correction, no significance
claim, no out-of-sample confirmation, no model. That is the precise claim, and the precise
version matters, because the loose version — "no §5.4 pattern appears in Phase 1" — is not
true of this report and was written that way until the 2026-08-12 review caught it.

What the report does show is descriptive cross-tabs by **academic year** (sellout rate,
median sell-through), **brand** (median sell-through, median revenue) and **venue** (median
tickets). All three are lines on the §5.4 list. They are not smuggled in: SPEC.md §8.9
requires knowing the growth trend and the programme's composition *before* any later
calendar effect can be believed. A night you only started running recently proxies for
"recent" in any model; a brand that moved into bigger rooms shows falling sell-through
while it is growing. Skipping these tables would not be more disciplined, it would just
leave the §8.9 risk unquantified.

The line between the two is: reading a median off a cross-tab is describing; comparing it
to a null and claiming the difference is real is testing. Only the second one needs a
frozen `preregistration.md`.

What *is* genuinely absent is the calendar crossed with an outcome: no tickets by term
week, no freshers' week, no exam period, no loan instalment, no tickets by day of week.
Day-of-week and academic-year *counts of events* are in the report — composition, not
demand.

The reason is the trap in SPEC.md §5.2, and Phase 1 is exactly where it springs. If you
plot tickets by academic week today, you will see something. You will remember it. And the
"pre-registered" list you write afterwards will be a list of what you remember seeing —
which is not a pre-registration, it is a description of the noise in your own data dressed
up as a prior. That damage is not recoverable by being careful later.

So the discipline is enforced by three tests rather than by good intentions, and the split
is deliberate — the first one alone could never fail, because Phase 1 builds no calendar
features at all:

- `test_report_crosses_no_calendar_feature_with_an_outcome` — no calendar word in a *table
  row*. A guard rail against a future edit, not against today's code.
- `test_report_body_contains_no_inference_vocabulary_at_all` — no `p-value`, `p =`,
  `significan…`, `FDR`, `Benjamini`, `confidence interval`, `correlat…`, `regression` or
  `coefficient` anywhere in the body of the report. This is the claim the report actually
  makes, and it is a claim a careless edit really could break.
- `test_report_discloses_the_cross_tabs_it_does_show` — the preamble must name the year /
  brand / venue cross-tabs and cite §8.9. Claim and content, pinned together, so nobody can
  quietly tighten the wording back to the overreaching version.

### Definitions fixed here, inherited everywhere

`event_outcomes()` lives in this module and is imported by Phases 2 and 3, so "tickets
sold" means one thing in the whole repo:

- **tickets** — paid units only. Comps and guestlist excluded (SPEC.md §8.7).
- **sell_through** — tickets / capacity. The demand measure.
- **room_fill** — (tickets + comps) / capacity. The fire-marshal measure. Reported
  separately because mixing them up is how a 60%-sold night becomes a "sellout".
- **revenue_buyer** — what buyers paid, face + fee. Provisional: `fee_treatment` is still
  `UNKNOWN` for every platform (SPEC.md §8.6).
- **sellout** — sell_through ≥ 0.95. A judgement call, not a fact, and the threshold is a
  named constant so it can be argued with.

### What can go wrong with this method

- **A descriptive table is still a choice.** Excluding cancelled events from every table
  but one is a decision that conditions everything downstream on the event having gone
  ahead. It is stated in the report rather than buried, but it does not stop being a
  limitation because it was declared.
- **Medians hide bimodality.** Every distribution here is summarised by percentiles. A
  programme that is really two programmes — a 150-cap room and a 1,500-cap room — has a
  median that describes neither. That is why the figures show distributions and not just
  the summary table.
- **The sellout rate depends entirely on the threshold.** At 0.95 the fixture reports 3.4%;
  at 0.90 it would report 4.2%. Any number of this kind should be quoted with its
  definition attached or not quoted at all.
- **It is very easy to look one table too far.** The whole risk of this phase is the
  analyst, not the code.

---

## Phase 2 — Demand forecast (`src/pricing/phase2.py`)

### What it is

Given only what was known when the prices were set, predict how many tickets an event will
finally sell. Plain OLS on `log(tickets)` with nine pricing-time features, validated on
held-out events against a naive baseline.

```
uv run python -m pricing.phase2 --synthetic
```

On the synthetic fixture:

| split | model MAE | model MAPE | baseline MAE | baseline MAPE | improvement |
|---|---|---|---|---|---|
| grouped 5-fold CV, pooled | 42.7 | 14.5% | 68.0 | 25.8% | 37% |
| final-year holdout | 53.0 | 16.6% | 80.1 | 25.2% | 34% |

The baseline is the mean tickets of the same brand at the same venue, computed on training
rows only. It is the right baseline because it is what the business actually does: "the
last three nights in that room did about 300, so call it 300."

### Why plain OLS on a log target

Three reasons, in order of weight.

**Small n.** 117 events. The model already spends 13 parameters. A gradient booster on 117
rows will fit the noise beautifully and there is no held-out set large enough to catch it
(SPEC.md §8.5). The constraint is written into DECISIONS.md as a binding requirement, and
it is the correct engineering call, not a compromise.

**Logs, because the outcome is multiplicative.** Ticket sales span 57 to 1,001 in this
fixture. A 50-ticket error is catastrophic at a 150-cap room and a rounding error at a
1,500-cap one. Modelling `log(tickets)` makes the error proportional, which is both how the
business thinks about it and what makes the residuals roughly homoscedastic. It also makes
every coefficient read as a percentage effect, which is what you want when explaining the
model out loud.

**Explainability is a deliverable.** The whole model is one formula, printable in a line,
with a coefficient table and confidence intervals. That matters more here than a point of
MAPE, because the model's job in an interview is to be defended, not to win a leaderboard.

### The two validations, and why both

**Grouped 5-fold by event** (SPEC.md §8.1). Every row here is one event, so grouping is
trivially satisfied — but the split function is written and tested as a proper grouped
split anyway, because Phases 3 and 4 reuse it on tier-level rows where a random split
really would put the same event in train and test. It is ten lines and no sklearn:

```
shuffle the groups -> deal them round-robin into k piles -> a row's fold is its group's pile
```

**Final-year holdout** (SPEC.md §5.5). Fit on the first four academic years, predict the
fifth. This is strictly harder than cross-validation and it is the honest number, because
it is the only one that matches how the model would be used: predicting a season that has
not happened. It also forces the brand-growth trend to *extrapolate* one year past its
data, which is the model's most fragile assumption and deserves to be tested rather than
hidden. The holdout MAE being worse than the CV MAE (53 vs 43) is exactly what should
happen; if it were not, something would be leaking.

### Look-ahead, tested rather than asserted

SPEC.md §8.2 is not a rule about column names, it is a rule about *when you knew things*.
So the check has two halves:

1. **By name** — no feature is on the outcome list. Catches the obvious accident of
   dropping `sell_through` into the design matrix.
2. **By value** — the whole feature frame is rebuilt from transactions *truncated at each
   event's first paid sale* (the decision moment) and required to come out identical. Any
   feature that secretly reads later sales moves when the later sales are removed, whatever
   it is called.

The second check is the one that matters, and it is why `lead_to_announce_days` — the only
feature touching `purchased_at` — is legitimate: it is the minimum timestamp, so truncating
at the minimum cannot change it. A test plants a leak (`sell_through` as a feature) and
proves the check catches it; a check nobody has seen fail is decoration.

### What can go wrong with this method

- **No calendar features.** Every SPEC.md §5.4 academic-calendar pattern is absent, because
  `preregistration.md` is DRAFT and §5.3 forbids testing them before it is frozen. The
  calendar is a large, real driver of student-event demand, so **these errors are an upper
  bound**. That is the price of doing the ordering honestly, and it is worth paying.
- **`marketing_spend` is endogenous, and it is the biggest coefficient in the model.** More
  is spent on events already believed in. That makes it an excellent predictor and a
  worthless causal statement — and, for the same reason, an invalid instrument for price in
  Phase 4 (SPEC.md §4.5). The workbook figure is also *realised* spend, which can react to
  a weak presale; the budgeted figure would be cleaner and should replace it.
- **Every coefficient is an association.** Capacity, guarantee and marketing are all chosen
  by the same person at the same time using the same beliefs about the night. Read the
  table as "what an event like this usually does", never as "what would happen if I changed
  this".
- **Events are not independent** (SPEC.md §8.10). Two nights a week apart eat each other's
  audience. The cannibalisation features are pre-registered and not yet in the model, and
  the standard errors assume an independence the programme does not have.
- **exp() of a log fit predicts the median, not the mean.** Duan's smearing factor
  (mean of exp(residuals), 1.011 here) corrects it. The headline numbers deliberately leave
  it off, because MAE and MAPE are minimised by the median. Phase 5 wants the *mean* profit
  and will need it.
- **Conditioned on the event happening.** Cancelled events are excluded, so this forecasts
  demand for events that went ahead (SPEC.md §8.4).

---

## Phase 3 — Sales curve (`src/pricing/phase3.py`)

### What it is

How tickets accumulate against days-to-event, and what a partial presale forecasts.

```
uv run python -m pricing.phase3 --synthetic
```

For every event, cumulative paid sales at each lead day are divided by the event's final
total, giving a curve from 0 to 1. The median of those curves, per segment, is the sales
curve. The forecast is one division:

```
forecast = tickets sold by day d / median share sold by day d
```

On the synthetic fixture, held out by event:

| decision day | median share sold | MAE | MAPE | "assume no more sales" MAPE |
|---|---|---|---|---|
| −7 | 72% | 42.7 | 15.7% | 29.0% |
| −3 | 79% | 35.0 | 12.9% | 20.3% |
| −1 | 83% | 31.5 | 11.2% | 16.7% |

This is "presale velocity", and operationally it is the most useful thing in the repo: it
is what tells you on the Monday whether to spend on marketing or leave it alone.

### Why an empirical curve and not Bass diffusion

SPEC.md §6.2 offers Bass "or something simpler that works". The empirical median curve is
the simpler thing, and here it is also the better one.

Bass has three parameters (p, q, m) fitted per event by non-linear least squares, and `m` —
market potential — *is the quantity being forecast*. Fitting a three-parameter non-linear
curve, from a partial series, at roughly 24 events per segment, in order to read off a
number that a single division already gives you, is machinery for its own sake. It can also
fail to converge, and then you have to decide what to do about that.

The empirical curve makes exactly one assumption and states it in a sentence: **this
event's presale is shaped like the median of previous events in its segment.** That
assumption is directly checkable — the report prints the p10-to-p90 spread of the curves at
each lead day, and it is wide (about 0.39 at day −7). The assumption is doing real work and
is only roughly true. A parametric curve would not fix that; it would hide it behind a
goodness-of-fit statistic.

One small piece of luck worth knowing: each event's cumulative share is non-decreasing, and
the pointwise median of non-decreasing functions is non-decreasing. So the median curve is
a valid cumulative curve for free — no smoothing, no monotone constraint, nothing to tune.

### The segmentation trap

The obvious segmentation is brand × sold-out-or-not. Half of it is illegal.

**On day −7 you do not know whether the event will sell out.** A forecast that conditions on
the sellout flag is conditioning on the answer. So the report carries both, clearly
separated: `brand` (known months ahead — this is the forecaster) and `brand × sellout`
(descriptive only, shown because whether a sold-out night's curve is shaped differently is
worth knowing, and labelled as something nobody could have forecast with).

The same discipline governs the evaluation: each fold's median curve is built from the
other four folds, so no event is ever forecast by a curve it helped draw. That is SPEC.md
§8.1 in its sales-curve costume, and it is easy to miss because there is no "model" being
fitted — just a median, which feels innocent.

### Two findings worth reporting even though they are negative

**Brand segmentation buys nothing.** Averaged over the three decision days, the
brand-segmented curve scores −0.11 MAPE points against the pooled one: noise. The honest
conclusion is to use the pooled curve. A split that buys nothing still costs degrees of
freedom, and on real data it would eventually pick up noise and call it a segment.

**At day −7 the presale forecast is worse than the Phase 2 pricing-time model** (15.7% vs
14.5% MAPE) and only overtakes it at day −3. That is a real result on this fixture, and it
is the sort of thing that would have been embarrassing to assert from memory. The right
reading is not "which model wins" — it is that the two use different information and should
be combined, the pricing-time forecast as the prior and the presale as the update. That is
Phase 5's job.

### What can go wrong with this method

- **A tier drop after the forecast day breaks the shape.** The median is taken over events
  whose ladders opened on their own schedules; an event that releases a new tier at day −5
  gets a jump the median curve does not have. Conditioning the curve on ladder timing is
  the obvious next refinement.
- **Sellouts truncate.** A night that sells out at day −4 has a curve that flattens because
  there is nothing left to sell, not because demand stopped. Scaling those curves up
  estimates **sales**, never **demand** — a distinction that matters the moment Phase 6 asks
  what a higher price would have earned.
- **The accuracy mostly comes from the numerator.** By day −1 the median event has banked
  83% of its final total, so the forecast is scaling up a small remainder. The curve is not
  getting cleverer as the night approaches; there is simply less left to guess. Quoting the
  day −1 MAPE as if it were a modelling achievement would be dishonest.
- **Small segments.** The pooled fallback (below 12 training events) exists for real data,
  where a brand × venue × year segmentation would be very thin indeed.

---

## Phase 4 — Price elasticity of demand (`src/pricing/phase4.py`)

**This is the project.** Everything before it is assembly and everything after it is
arithmetic on top of one number. Read this section before any other.

### The question, and why the obvious answer is wrong

*If I put ticket prices up 10%, how many fewer tickets do I sell?*

The obvious move is to regress log quantity on log price across events. On the synthetic
fixture that gives:

```
naive pooled OLS      log Q ~ log P        +0.698   (SE 0.099, 95% CI [+0.502, +0.893])
```

**Positive.** Higher prices "cause" more sales, significantly. That is not a bug and it is
not a small-sample fluke — it is the signature of the thing that makes this project hard:
**I set the prices, using the same information that drove demand.** Big artist, good date,
freshers' week → I charged more *and* more people came. Price is correlated with the error
term. The estimated coefficient is the true elasticity plus a bias term, and the bias term
is bigger than the truth.

The generator plants this deliberately: the operator's price loads on the event's demand
shock with a coefficient of 0.55, and the demand shock also drives quantity. Omit the shock
and price picks up the shock's effect as its own. Textbook simultaneity, made on purpose so
that the estimator has to survive it.

**The wrong response is to drop variables until the sign turns negative.** That is
p-hacking your own business (SPEC.md 8.3). The right response is to find variation the
demand shock cannot explain.

### What event fixed effects identify — and what they do not

Each event sells several tiers at different prices: Early Bird, Tier 1, Tier 2, Door. Same
artist, same date, same room, same marketing — different prices. So put a dummy in for
every event:

```
log Q_et = alpha_e + beta * log P_et + gamma * X_et + eps_et
```

`alpha_e` absorbs **everything constant within the event**: the artist, the date, the
venue, the hype, the marketing spend, the weather, the competing night down the road. What
is left to identify beta is within-event price variation only.

That alone gives **−1.260**, against a truth of −0.800. Right sign, wrong number, 58% too
elastic — and if the fixture had not been built with a known answer, that is the number
that would have gone on the CV.

**Fixed effects are not enough, and the reason is SPEC.md 4.3.** Within an event, tier
price is nearly the same variable as time-to-event. Early Bird is cheap *and* early; the
door price is expensive *and* the night itself. Two stories are tangled together:

- **(a)** Early Bird sells more because it is cheaper — the price effect we want.
- **(b)** Early Bird sells more because eager buyers buy early and it is the only thing on
  sale — selection and timing.

Fixed effects remove everything constant within the event, but the timing effect *varies*
within the event exactly the way price does, so the estimate charges the timing effect to
price. Add the lead-time control and:

```
event FE + lead-time control                   -0.815   (SE 0.053, CI [-0.920, -0.711])
BETA_TRUE                                      -0.800
```

That middle line — the FE-only estimate — matters as much as the last one. It is the number
you get by doing the standard, respectable thing and stopping one step too early.

### Dummies or demeaning? The same slope, a different interval

The regression above is written with `alpha_e` as a dummy, but it is **fitted** as the
within transform: subtract each event's own mean from `log_q`, `log_p` and `lead_term`,
then fit with no intercept. Frisch–Waugh–Lovell says those are the same regression, and on
the fixture they agree to about 1e−16 — there is a test.

They are not the same for *inference*, and this is the fix that came out of the 2026-08-12
review. With 117 events and 402 rows:

| form | parameters | clusters | SE | 95% CI | width |
|---|---|---|---|---|---|
| event dummies | 119 | 117 | 0.063 | [−0.939, −0.691] | 0.248 |
| within transform | 2 | 117 | 0.053 | [−0.920, −0.711] | 0.209 |

The dummy form asks for **more parameters than there are clusters**. A cluster-robust
covariance matrix is a sum of one outer product per cluster, so its rank is at most
G − 1 = 116; with 119 parameters it is singular, and each event dummy is being fitted off
the 3.4 rows inside its own cluster. That is not a footing an interval can stand on. The
within form has two parameters against 117 clusters, which is what cluster-robust inference
assumes in the first place.

**The switch narrows the interval, so the report prints both.** The raw cluster-robust
variance is identical between the two; the entire 19% gap is statsmodels' finite-sample
factor `(N−1)/(N−K) · G/(G−1)`, which counts all 117 absorbed dummies in K. Whether they
belong there is genuinely unsettled — it is the same disagreement as Stata's `areg` versus
`xtreg, fe`. So the report shows both intervals and states, computed rather than asserted,
whether the choice could have changed the verdict. Here it could not: 0.209 and 0.248 are
both far inside the 1.00 CI-width criterion and both intervals lie entirely below zero.

### What makes it identified at all, and how we decide

Once the event fixed effects and the lead-time control are in, **93.5%** of the variance of
log price is explained by "which event is it" and "how far out was it". The identifying
variation is the remaining **6.5%** — ladders applied slightly differently across events,
tiers opened at different lead times, prices corrected mid-window. That number is the whole
game, so the verdict is written as a function of it and three other numbers.

**But the four criteria are not four of the same thing**, and the report prints them under
two headings for that reason (also a 2026-08-12 review finding). Three are properties of
the **design** — they can be evaluated without ever looking at what the estimate came out
as, and they decide whether this dataset *can* answer the question:

| design criterion | value | threshold | why |
|---|---|---|---|
| residual price variation | 0.065 | ≥ 0.02 | below this, beta is fitted off rounding errors |
| 95% CI width (clustered, within) | 0.209 | ≤ 1.00 | a wider interval cannot tell a harmless price rise from a ruinous one |
| events with 2+ priced tiers | 117 | ≥ 30 | the within estimator needs ladders, not events |

The fourth is a **sign check, and it is labelled as one**:

| sign criterion | value | threshold | why |
|---|---|---|---|
| CI lies below zero | −0.711 | < 0 | a straddling interval means the confound is still in there |

This one conditions on the answer. A within-event estimate of +0.30 with a tight interval
and abundant residual variation would be called NOT-IDENTIFIED by this rule even though
every design criterion passed — and identification is a property of the design and the
variation, not of whether the sign came out the way theory predicts. Say that before
someone else does.

The reason it is still a criterion is specific, and it is not "we expected a negative
number". The confound this whole phase exists to remove has a **known direction**: the
operator priced off the same demand signal that drove sales, which pushes the estimate up
(SPEC.md 4.1 — the pooled regression comes out at +0.698). So a positive estimate that
survives the fixed effects is not evidence that demand slopes upward; it is a measurement
that the confound is still in there, and "not identified" is the honest label for a number
still carrying the bias you were trying to remove.

Why that is not p-hacking the sign: p-hacking is *searching* over specifications until the
sign comes out right and reporting only the winner. Every specification tried here is
printed — the wrong-signed pooled estimate is row 1, the over-elastic FE-only estimate is
row 2, all three instrument specs including the invalid one are in the table, and the
thresholds were fixed in `IDENTIFICATION_CRITERIA` before the estimate was looked at. The
real test is whether a failed sign check would be *reported* as NOT-IDENTIFIED rather than
quietly re-specified around, and SPEC.md 4.4 plus the NOT-IDENTIFIED branch of the report
are the commitment to that.

All four pass on the fixture, so the verdict is `IDENTIFIED`. **The criteria are constants
at the top of `phase4.py`, written before the estimate was looked at, and they are
arguments to the verdict function so a reviewer can re-run it at their own thresholds.**
That ordering is the entire value of them.

The VIF of log price is 15.4 — above the textbook rule of thumb of 10. It is reported as a
**flag, not a failure**, and the reason is worth being able to say out loud: a VIF is only a
statement about how much the standard error is inflated. Whether the resulting interval is
usable is a different question, and the CI-width criterion is the one that asks it. At 402
panel rows the interval is ±0.10, which is usable. A VIF of 15 with a tight interval and a
VIF of 15 with a useless one are not the same situation and one number cannot tell them
apart.

### Standard errors: clustered by event

Two tiers of the same event share the artist, the date and the marketing push, so their
residuals are correlated and ordinary standard errors are too small. Every estimate above
is clustered by event. On the fixture the effect is visible: plain 0.043 → HC1 0.051 →
clustered by event 0.063 (on the dummy form).

**Clusters, not rows, are the sample size for this interval.** A cluster-robust covariance
matrix is built from G = 117 independent pieces, so two things follow and both are
implemented. First, the reference distribution is a **t on G − 1 = 116 degrees of freedom**
(critical value 1.981), not the normal 1.960 — statsmodels only does this if you pass
`use_t=True`, and it is easy to leave off. Second, the number of parameters has to stay
well below G, which is why the fixed effect is fitted by demeaning rather than with 117
dummies (see "Dummies or demeaning?" above). Both were wrong before the 2026-08-12 review:
the interval used a normal critical value off a covariance matrix with 119 parameters and
117 clusters.

**What clustering does not fix**: events are not independent of each other either — two
nights a week apart eat the same student budget (SPEC.md 8.10) — and the intervals above
quietly assume they are.

### When 2SLS is the better answer, and what happened here

The stronger identification is a **cost shifter** (SPEC.md 4.5): something that moves price
and reaches demand through no other channel. Demand-side hunches move both, which is the
whole problem, but the cost stack only moves price. The venue put its hire rate up; that
went into the ticket price; **the buyer neither knows nor cares what the room cost me.**

That is a different regression on a different panel — one row per event, not one per tier,
because a cost shifter varies *across* events and an event fixed effect would absorb it
completely. With one instrument, two-stage least squares collapses to a ratio worth saying
out loud: **the effect of the hire rate on quantity, divided by its effect on price.**

Three specifications, and the result is a clean negative:

| spec | instrument | controls | first-stage F | IV beta |
|---|---|---|---|---|
| I (pre-specified) | log venue hire | venue + academic year | **2.8** | **not run** |
| II (diagnostic) | log venue hire | venue only | 15.6 | +0.366 |
| III (invalid) | log artist guarantee | venue + academic year | 96.5 | +1.008 |

The F is the squared first-stage t, scaled by `(n − k − 1) / n` for the parameters the
controls already spent — the residualised regression is fitted on n rows but only
n − k − 1 of them are free, and without the scaling the F is overstated in exactly the
direction that lets a weak instrument through the floor.

- **Spec I is excludable but too weak.** F = 2.8 is far below the relevance floor of 10, so
  no estimate is computed. A weak instrument biases the estimate back towards the OLS
  estimate it exists to fix, so a weak-instrument number is worse than no number: it looks
  like a correction and behaves like the disease.
- **Spec II is relevant and not excludable.** Drop the academic-year control and the first
  stage jumps to 15.6 — and the estimate comes out wrong-signed. The reason is SPEC.md 8.9:
  hire rates drift up year on year and so does a growing brand's demand, so an uncontrolled
  hire index is partly a time trend, and the time trend is a demand shifter. Controlling the
  year is what makes it excludable and it is also what takes most of its strength away.
  There is no version of this that gets both.
- **Spec III is the lesson.** The artist guarantee has by far the strongest first stage
  (F = 97) and is the least usable instrument in the table: a bigger guarantee means a
  bigger artist means more demand, so it fails exclusion outright and returns +1.008.
  **Relevance is testable and validity is not.** The strong number is the one to distrust.

**Exclusion cannot be tested with one instrument** — that is what makes it an assumption.
The report checks a necessary condition (is the instrument, after its controls, still
correlated with demand-side observables?) and states plainly that passing it is not
evidence *for* exclusion. The scaffold stays in the repo because the day a real hire-rate
shock lands — a venue that puts its rate card up 25% in one year — it becomes the best
identification in the project.

### What this cannot identify, whatever the verdict says

- **Not the elasticity at a different price level.** Every estimate is local to the ladders
  actually used. Extrapolating to a doubling of prices is not a modelling choice, it is a
  fabrication.
- **Not separate elasticities per tier.** Early-bird buyers and door buyers are different
  people with different willingness to pay, and one beta averages them. Separating them
  needs more within-event price variation than exists.
- **Not freshers' week versus the loan instalment versus the weather** (SPEC.md 5.6). They
  arrive together, every year, for five years. Nobody could separate them with ~120 events.
  They are reported as one grouped start-of-semester effect and explicitly not decomposed.
- **Not cannibalisation.** Two events a week apart are not independent observations.
- **Nothing about buyers who never bought**, or about the three events that were cancelled
  and are excluded here (SPEC.md 8.4). Every number is conditional on the night happening.

### What can go wrong with this method

**The estimate is observational, and it rests on an assumption nobody can test.** Even at
−0.815 with a tight interval, identification requires that whatever made ladders differ
across events was not itself a demand signal. If I applied a steeper ladder to nights I
expected to sell out, the residual variation is not clean and the number is wrong in a way
no diagnostic here would catch. That is precisely what the randomised experiment in
`experiment/DESIGN.md` removes, and it is running now.

**The fixture can be easier than reality.** The generator plants the endogeneity and the
lead-time confound, but real prices move for reasons it does not contain — a promo code, a
competitor, a mis-priced tier corrected in public. Passing these tests means the estimator
is not broken. It does not mean the real elasticity is identified.

**`fee_treatment` is UNKNOWN** (SPEC.md 8.6). Elasticity acts on the price the buyer sees,
face value plus booking fee. Until that is confirmed against one real order confirmation,
`price` is a guess of the buyer's price and every number downstream inherits it.

**On real data, the honest answer may well be NOT-IDENTIFIED.** That is a first-class
result, not a missing one, and the report template treats it as one:

> *Documented why price elasticity is not cleanly identifiable from prices set by the
> operator, and specified the release-tier randomisation required to recover it.*

---

## Phase 5 — Break-even, P(clear) and deal structure (`src/pricing/phase5.py`)

### What it is

Three numbers per booking, and one decision:

```
contribution per ticket = what you keep from one paid ticket (face value; the fee is the platform's)
break-even attendance   = fixed cost stack / contribution per ticket
P(clear)                = P(tickets sold > break-even attendance)
```

then, because the same artist can be booked three ways (SPEC.md 6.3), which contract to
sign. Writing `Net` for what the night makes before the artist is paid — ticket
contribution minus every fixed cost except the artist:

| structure | your profit | who carries the risk |
|---|---|---|
| fixed guarantee | `Net − F` | you, all of it |
| door split | `(1 − s)·Net` | shared, both directions |
| guarantee OR s% of door, whichever is greater | `Net − max(F, s·Net)` | you keep the worse of the two |

### The thing worth noticing about the third row

It pays **the minimum of the other two, on every single night**:

```
payoff_guarantee_or_split == minimum(payoff_fixed_guarantee, payoff_door_split)
```

exactly, for any attendance, any fee and any share — there is a test that asserts it
pointwise across the whole range. Below the crossover (`Net = F/s`) it behaves like the
flat fee; above it, like the split. The artist takes the better of their two outcomes on
every night, so the promoter takes the worse of theirs: the full downside of the guarantee
deal *and* the capped upside of the split deal, the two worst halves. **The only
compensation for that is a lower F**, and if the fee is the same as a straight guarantee,
you are being paid nothing for the difference. That is the standard live-music contract.

### Why the distribution is bootstrapped and not fitted

Phase 5 fits no demand model. It takes **Phase 2's out-of-fold predictions** — every event
scored by a model that never saw it, folds grouped by event (SPEC.md 8.1) — and resamples
their residuals with replacement.

- The **centre** of each event's distribution is its out-of-fold prediction, so the
  simulation is centred on what a model that had never seen the night would have said, not
  on a fit that already knew the answer.
- The **spread** is the spread of mistakes the forecast actually made. A normal
  distribution fitted to those residuals would be smoother and would understate the thing
  that matters most: the bad tail, which in this business is a wet Wednesday in exam
  season, not a symmetric wobble.
- Simulated attendance is capped at capacity, which is why the profit distribution is
  asymmetric even though the residual distribution nearly is.

Two demand models in one repo means two answers to "how many will come", and the day they
disagree is the day the report stops being trustworthy. There is one.

**And the claim is pinned, which it was not until the 2026-08-12 review.** A sabotage
replaced `rng.choice(residuals)` with `rng.normal(0, residuals.std())` and the whole suite
stayed green: the report said "no normal distribution is fitted to anything", the explainer
said it, a viva answer was entirely about it, and nothing checked it. The test that fixes it
is set arithmetic — an empirical bootstrap can only ever produce `exp(log_pred + r)` for an
`r` that is literally an element of the residual array, so the distinct simulated ticket
counts must be exactly that set. A parametric draw produces a continuum and fails at once.
The draws are made in one function, `bootstrap_units`, which both the per-event simulation
and the fee curve call, so there is one place to sabotage and one place that is pinned.

### How many draws, and the uncertainty draws cannot fix

40,000 per event, and the report **measures** the resulting precision instead of asserting
it. Every run simulates the whole programme twice, on two bootstrap seeds over the same
residuals, and prints how far the numbers moved: on the fixture the worst event's
5th-percentile night shifts about £30 and the median event £0.

That replaced a claim that was simply false. The old constant was 4,000 draws and the
report stated "5th percentile stable to about a pound"; re-running at 4,000 moves the worst
event by £184. A checkable numeric claim that nobody had checked is the exact failure this
project is supposed to be about not making.

The useful part is which uncertainty that is **not**. Monte Carlo error shrinks like
1/√draws, so it is the cheap one and more draws buy it down. The residual distribution is
**atomic** — one residual per event, ~120 of them — so a simulated 5th percentile can only
ever land on one of ~120 values, and resampling ten million times adds no information about
how bad a night can get. That is why the median move across seeds is £0 while the worst
event's is £30: between seeds it is usually the same atom and occasionally one step away,
and the size of that step is set by the data. Quote these percentiles as an order of
magnitude.

### The decision rule

> Take the guarantee when `E[profit | guarantee] > E[profit | split]` **and** the left tail
> is survivable.

Risk-adjusted, not expected-value, and *survivable* is a number rather than a feeling:
`MAX_SURVIVABLE_LOSS`, one night, written down so it can be argued with. Where the
guarantee wins on the mean but its 5th-percentile night is worse than that, the
recommendation flips to the split — you are buying insurance, and its price is the expected
profit you hand over.

### The number to walk into the phone call with

`max_guarantee_at_target`: the largest fee that still leaves a 50% chance of clearing. It
is the (1 − target) quantile of simulated `Net` — one line, no solver. The report prints the
whole fee curve for the booking with the most headroom, which turns *"can you do £5,000?"*
into a probability:

```
FAKE-EV-072    fee £0 → 100%    £3,650 → 82%    £5,470 → 44%    £7,300 → 19%
```

### What can go wrong with this method

- **The synthetic P&L is not a P&L.** The generator was built to make statistical
  properties visible and its cost stack was never calibrated against its ticket prices: on
  the fixture the median break-even is 1.1x the median tickets sold, so almost every
  simulated night loses money and P(clear) sits near zero. Read the *shapes* — how the three
  structures order themselves, how the tail behaves — and wait for the real cost stack for
  the levels. The report says this in a box rather than quietly showing a friendlier
  example.
- **The residual pool is small.** It has as many draws as there are events. The 5th
  percentile is read off a few dozen bad nights and should be quoted as an order of
  magnitude, not a number.
- **Nights are simulated independently.** Two events a week apart eat each other's audience
  (SPEC.md 8.10), so adding these distributions into a portfolio would understate a bad
  month.
- **Everything is conditional on the night happening.** Cancelled events are excluded, and
  the count comes from the data rather than from a sentence. *Why* each one was pulled is
  **not checkable**: their presale histories were destroyed on-platform (DECISIONS.md,
  survivorship), so the rows that would say how badly they were selling do not exist. If
  poor sales are part of what gets a night pulled — and on the operator's account they
  sometimes are — the true left tail is worse than the one drawn here, by an amount nobody
  can currently quantify. The report used to say flatly that the pulled events "were pulled
  *because* they were selling badly"; that is a claim about the real programme this dataset
  cannot support, and it is gone. **A known, unmeasured bias stated as unmeasured beats a
  confident sentence with nothing behind it.**
- **No bar spend, no repeat custom.** For a promoter with a bar deal, tickets are not the
  whole margin, and a night that loses money on tickets can still be worth running.
- **The artist's share `s` is an assumption**, not data. It is in the contracts, and until
  they are read it is applied identically to every event so the structures compare like for
  like.

---

## Phase 6 — Counterfactual pricing backtest (`src/pricing/phase6.py`)

### What it is

The headline number of the project, and the one most likely to be a lie.

Take each event's actual ladder and re-price it on a **visible grid** — no optimiser, no
solver, nothing you cannot print:

- **level** × 0.70 … 1.30 in steps of 0.05 — move the whole ladder
- **spread** × 0.00 … 1.50 — stretch or flatten the gaps between rungs (1.00 = as it was)

91 candidate schedules per event. At each one, quantity moves as
`units × exp(beta × change in log price)`, the face value moves by the same proportion as
the buyer price, and the event's total is capped at its licensed capacity with every tier
scaled back proportionally. Each event keeps its own best grid point. **At (1.00, 1.00)
every change is zero and the counterfactual reproduces exactly what happened** — the
cheapest possible check that the machinery is not quietly adding something.

On the fixture, with beta = −0.815, that search gives **+5.3%** — and the report does not
lead with it, because 100% of events choose a level at the edge of the grid. That number is
the grid bound restated ("at least +5.3% within the ±30% range considered"), not something
the data found. The headline is the uplift at the experiment's pre-committed +15% dose,
**+2.6%**: one price change, no search, no corner. Projected either way, never realised —
nobody was charged these prices.

### The honesty rule that governs the whole module

**If Phase 4's verdict is NOT-IDENTIFIED, the headline uplift number does not exist.** Not
"is uncertain" — does not exist. An uplift is an elasticity multiplied by a price change,
so an uplift computed from an unidentified elasticity is an assumption wearing a percentage
sign. In that case the module reports an **uplift curve over assumed elasticities**, every
row labelled with the assumption it rests on, plus the point estimate at the fixed-effects
beta clearly marked PROVISIONAL — and points at `experiment/DESIGN.md` as the thing that
pins beta. There is a test that renders that branch and asserts the headline is absent.

**The rule has to hold on every line, not just the headline, and it did not.** Forcing
NOT-IDENTIFIED in the 2026-08-12 review produced a report that refused a headline on line 8,
labelled the point estimate PROVISIONAL on line 14 — and printed
`- programme uplift: **+5.3%**` on line 48 with no qualifier at all. The old test only
checked that one exact headline string was absent, so it passed. A section heading does not
survive a copy-paste; a bold number does. The fix is structural on both sides:

- In the NOT-IDENTIFIED branch the bold comes off every programme-level percentage and the
  word PROVISIONAL goes on the same line as the number, not three paragraphs above it.
- The test is now a scan rather than a string match: every bold percentage in the rendered
  NOT-IDENTIFIED report must either be a row of the assumed-beta curve (labelled with its
  assumption in the first cell) or sit on a line containing PROVISIONAL. And a second test
  asserts the bold *returns* when the verdict is IDENTIFIED, so the qualifier stays a
  consequence of the verdict rather than becoming formatting.

### Two results in the output that are arithmetic, not discoveries

Both are stated in the report, because a reader who spots them before you do will assume
you did not understand your own model.

**1. The optimum sits on the edge of the grid.** At beta = −0.815, 100% of events pick the
top level. That is `price^(1+beta)`: with inelastic demand (`|beta| < 1`) revenue rises with
price without limit, so the model wants infinity and stops only where the grid stops. **The
±30% bound is an assumption about what a student night can be re-priced by without becoming
a different product — it is not something the data implies.** The model has no mechanism for
a brand becoming known as expensive, for a competitor undercutting, or for the bar spend of
the people who stopped coming. That is why the report also gives the uplift at the
experiment's pre-committed **+15%** dose (+2.6%): a price change the business has already
agreed it can live with, so the number has a use rather than being a corner solution.

**2. The ladder's shape is chosen by the same single number.** At the estimated beta the
model wants to flatten the ladder to one price (modal spread ×0.00) — with `|beta| < 1` the
objective is concave in log price, so spreading prices apart can only lose money. Flip to
`|beta| > 1` and it turns convex and the model stretches the ladder as far as the grid
allows instead; the fixture's own curve shows exactly that at beta = −1.2. Neither is
advice. One beta for every rung means tiers have no separate willingness to pay. **This is
not a recommendation to abolish early-bird pricing** — it is the model naming the question
it cannot answer.

**3. The curve is U-shaped and touches zero at beta = −1.** At unit elasticity, revenue does
not respond to the level of prices at all. Either side of it the model wants to move prices
in *opposite directions* — raise them if demand is inelastic, cut them if it is elastic. So
the double-digit uplift at the elastic end of the curve and the double-digit uplift at the
inelastic end are not the same claim at different confidence: one of them is telling you to
cut prices and the other to raise them, and they are contradictory pieces of advice that
happen to carry a similar-looking number. That is exactly why an unidentified beta cannot
produce a headline, and why the experiment matters more than any estimator choice.

The two numbers in that sentence used to be typed into the report by hand. They were right
on one seed and wrong on the next, which meant the prose contradicted the table two
paragraphs above it — caught in the 2026-08-12 review. They are now read off
`results["curve"]` by `_contradiction_sentence`, with the rows chosen by a rule (the
extremes of the assumed range) rather than by which pair looked most striking, and a test
compares the sentence against the table on two seeds. **In a report whose selling point is
that every number regenerates, a hand-typed number in the prose is the worst defect
available**, and that is worth carrying as a general rule rather than as one fix.

### The ground-truth check (synthetic only)

The chosen ladders are pushed back through the **generator's own** demand process — softmax
reallocation across tiers, event total responding to the average log price, the generator's
3-ticket floor and its sell-out cap — using the true elasticity rather than the estimated
one:

```
model-implied (constant elasticity, tier by tier)   +5.3%
the generator's true demand process                 +8.7%
ratio                                                0.61
```

Same sign, same order of magnitude, pinned as a test across seeds with a deliberately wide
tolerance: agreement to three decimal places would only prove that two nearly identical
formulas are nearly identical. The gap has a specific cause — the model moves each tier by
its own price change, so it is implicitly quantity-weighted, while the generator moves the
event total by the unweighted average change. Flattening a ladder raises the cheap rungs
(most of the units) and cuts the expensive ones, so the quantity-weighted view sees a bigger
average rise and predicts a bigger loss of sales. **The model is the conservative one of the
two, which is the direction to be wrong in.**

**What it proves:** applying one elasticity tier by tier does not throw the answer away.
**What it does not prove:** that the real world is a constant-elasticity world. Nothing run
on a fixture can prove that.

### What can go wrong with this method

- **Buyers moving between tiers.** Change the gaps and some Early Bird buyers become door
  buyers, or stop coming. That needs a tier-specific elasticity the data cannot support.
- **Time.** A ladder is a schedule, not just a set of prices. This grid re-prices the rungs;
  it does not re-time them. Phase 3's sales curve is what says whether a tier had time to
  sell.
- **Everything outside the room.** Bar spend, repeat custom, brand reputation and the value
  of a full room to the next booking are all outside the objective, and all four argue for
  lower prices than the model wants.
- **It is model-implied throughout.** "Projected", on the CV and in conversation, before
  anyone asks. Claiming a realised uplift that was never run is the same sin as a fabricated
  Sharpe ratio.
