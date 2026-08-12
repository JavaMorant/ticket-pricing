# VIVA

Interview-style questions per phase, with the answers. **No number from this repo is
quoted in an interview until these vivas are passed** (DECISIONS.md, 2026-08-11
amendment). Read the answer, close the file, say it out loud, then check.

Style note: these are written the way SPEC.md §10 frames them — a question a trading
interviewer would actually ask, and the shortest honest answer that survives a follow-up.
All nine of §10's questions are answered: five under Phase 4, the other four in the last
section, which also carries the survivorship answer DECISIONS.md marks non-deferrable.

---

## Phase 0.5 — Synthetic ground truth

### 1. Why would anyone generate fake data for a project whose whole selling point is that the data is real and proprietary?

Two reasons.

First, ordering. The pre-registered pattern list has to be written before I look at the
data — otherwise it is not a pre-registration, it is a description of what looked
significant. That list is still a draft, so the real data is off limits. I still wanted
the whole pipeline built and tested, so I built it against a fixture. The real-data loader
is written and wired; it reads the status line of `preregistration.md` and refuses.

Second, and more important: Phase 4 produces a number that goes on my CV. The only way to
know an estimator works is to run it on data where I already know the answer. So I set the
true elasticity to −0.8, generated a programme around it, and made the acceptance test for
Phase 4 "recover −0.8 from this". If the estimator can't do that on data I built, it has
no business being pointed at data I can't check.

### 2. Isn't that circular? You built the generator and the estimator, so of course they agree.

Partly, and I'd say so up front. It rules out one class of error, not all of them: it
proves the estimator isn't broken, not that the real elasticity is identified.

What stops it being wholly circular is that the generator is written as a *story about the
business*, not as the estimating equation rearranged. The operator sets prices from their
demand expectations; the venue puts its hire rate up and that gets passed into the ticket
price; Early Bird is cheap and early. The estimating equation then has to dig the
elasticity out of that. I can also break the fixture on purpose — there's a knob that
switches the lead-time confound off — and watch which estimators start and stop working.
That's the useful part.

And I'd say the honest version of the limitation out loud: if real prices move for reasons
my generator doesn't contain, the estimator can pass here and fail there.

### 3. Your naive regression on this fixture gives +0.70. Talk me through why, in one breath.

Because I set the prices. Price loads on the event's demand shock with a coefficient of
0.55 — bigger artist, better date, I charge more — and the demand shock also drives
quantity directly. The pooled regression omits the shock, so price picks up the shock's
effect as well as its own. Textbook omitted-variable bias: the price coefficient is the
true −0.8 plus a positive term big enough to flip the sign.

The fix is not to drop variables until it turns negative. The fix is to identify off
variation the shock can't explain — within-event, where the shock is a constant.

### 4. So event fixed effects solve it?

Not on their own, and this is the part I'd fail an interview on if I hadn't checked. On
this fixture, event fixed effects alone give −1.26 against a truth of −0.8. Over-elastic
by more than half.

The reason is that the within-event price variation is nearly the same variable as the
timing. Early Bird is cheap and it's early; the door price is expensive and it's the night
itself. Fixed effects absorb everything constant within the event, but the timing effect
varies *within* the event exactly the way price does, so the estimate charges the timing
effect to price. Adding a lead-time control gives −0.815, which is the truth inside
sampling error.

What makes that regression identified at all is the small slice of price variation that
isn't timing — I don't apply the same ladder multiplier every time. If the real data turns
out to have none of that, price and lead time are collinear and the right answer is that
the elasticity isn't identified from tier structure, plus the randomised experiment that
would settle it.

### 5. What's in the fixture that isn't the elasticity?

Four things, all planted because they're traps I'd otherwise walk into on the real data.

A brand growth trend — both brands grow year on year at different rates, so anything
correlated with time inherits it. If I started using a venue recently, that venue will
look brilliant unless I control for academic year first.

A valid instrument and an invalid one. The venue hire index moves 10–25% a year, gets
passed into the ticket price, and the buyer neither knows nor cares — that's a cost
shifter and a 2SLS candidate. The artist guarantee looks like an instrument and isn't: a
bigger guarantee means a bigger artist means more demand, so it fails the exclusion
restriction outright. Both are in the fixture so I can show the test rejecting one.

Marketing spend that loads on the demand shock, so "marketing causes sales" is a claim the
data will happily support and shouldn't.

And three cancelled events with thin presales, flagged. Every phase has to decide what to
do with them rather than silently conditioning on success.

### 6. How do you know the fixture actually has the properties you claim?

They're tests, not comments. `tests/test_synthetic.py` asserts the pooled regression is
positive and the within-event-plus-lead regression lands within 0.15 of −0.8, on three
seeds each, plus that plain fixed effects are visibly biased while the confound is on and
correct once it's switched off. Those four are the acceptance tests the Phase 4 estimator
has to pass before it's allowed near real data.

The schema is tested the same way: the generator builds its tables by calling the same
three build functions the real ingest calls, and the tests compare the result column for
column and dtype for dtype against the Phase 0 fixture.

### 7. Why ±0.15 tolerance? That's not very tight.

Because that's what 120 events buys. The identifying variation is the price movement
that isn't lead time, which is a fraction of the total, and across eight seeds the
recovered beta moves by about ±0.08. I set the tolerance at roughly twice the observed
spread so the test fails on a broken estimator rather than on a draw.

It's also a preview of the real result: whatever elasticity I report from real data will
come with an interval of that order, and I'd rather quote it than a third decimal place.

### 8. Why plain OLS with event dummies rather than something better?

Because I can explain every line of it, and because 400 panel rows do not support anything
bigger. One dummy per event is literally the alpha-e in the spec — it costs milliseconds
at this size and the within-transform gives the identical slope if it ever stops being
cheap.

The constraint is deliberate and it's written down in the decision log: no gradient
boosters, no Bayesian machinery. With ~120 events, a flexible model would fit the noise
and I'd have no way to tell. Small n is the binding constraint on this whole project, and
being visibly honest about it is the signal.

---

## Phase 1 — Descriptives

### 1. You have five years of proprietary data and the first thing you did was make bar charts. Why?

Because half the ways this project could be wrong are arithmetic, not statistical, and no
amount of cross-validation catches them. If comps are counted as demand, or the tier
ordinal is upside down, or a bad join has duplicated three events, the model learns the
broken thing perfectly happily.

But the real reason is that the descriptives decide what is worth modelling. The one that
mattered: Phase 4 can only identify an elasticity if tier price and lead time are separable
across events, and that is a descriptive fact — did the ladder timing vary? The ladder-span
table answers it. If every event had opened on the same schedule I would have skipped
Phase 4 and written the "not identified" paragraph instead. I would rather find that out in
Phase 1 than after fitting something.

### 2. Your report shows events by day of week but nothing about tickets by day of week. That's an odd gap.

It's deliberate, and it's the most important thing in the phase.

Day of week is on my pre-registered pattern list, and that list isn't frozen yet. If I plot
tickets by day of week today, I'll see something, I'll remember it, and the list I write
afterwards will be a list of what I remember seeing. That isn't a pre-registration, it's a
description of the noise in my own data with a prior's hat on. It's the cardinal sin in
§5.2 — using the same data to generate and confirm a hypothesis — and you can't undo it by
being careful later.

Counts of events by day of week are fine, because that's programme composition, not demand.
I need it, because if I only started running Thursdays recently then "Thursday" will
secretly proxy for "recent" in any later model. There's a test that fails if a calendar word
ever appears in a results table in that report.

### 2b. But your report does show sell-through by brand and sellout rate by year, and brand and academic-year cohort are both on your §5.4 list. Isn't that the same sin?

No, and I had the claim worded too loosely until a review caught it, so let me be exact.
What's forbidden before the list is frozen is *testing*: a p-value, a correction, a
significance claim, an out-of-sample confirmation. There isn't one in that report — no
p-value, no FDR, no model, and there's a test that scans the body of the report for that
vocabulary and fails if any of it appears.

The cross-tabs by year, brand and venue are there because §8.9 makes them a prerequisite,
not an indulgence. Both brands grew year on year, so any later calendar effect will be
partly the growth trend unless I know what the trend was; and a brand that moved into
bigger rooms shows falling sell-through while it's actually growing. If I skipped those
tables I wouldn't be more disciplined, I'd just have an unquantified §8.9 risk.

The line is: reading a median off a cross-tab is describing. Comparing it to a null and
claiming the difference is real is testing. Only the second one needs the frozen list. What
is genuinely absent from that report is the *calendar* crossed with an outcome — no term
week, no freshers', no exam period, no loan date, no tickets by day of week.

### 3. Your sellout rate is 3%. Doesn't that just mean you're pricing too low?

It means one of three things and I can't tell you which from this table alone: prices are
low, the rooms are too big for the acts, or my sellout definition is wrong. That's exactly
why it's a descriptive and not a conclusion.

The definition is a judgement call I've written down: paid tickets at or above 95% of
licensed capacity. Real rooms run short of licence for comps and no-shows, so demanding
100% would report zero sellouts on nights that were visibly full. Move it to 90% and the
fixture reports 4.2% instead of 3.4%. Any number of that kind should be quoted with its
threshold attached or not quoted at all.

### 4. Why do you report two different utilisation numbers?

Because they answer different questions and conflating them is how a 60%-sold night becomes
a "sellout" in a deck. Sell-through is paid tickets over capacity — that's demand.
Room fill adds comps and guestlist — that's the fire marshal's number. Comps aren't demand
(§8.7): a guestlist spot is a cost, not a sale, and if I let them into sell-through then the
worse a night sells, the more comps I paper it with, and the healthier it looks.

### 5. What did Phase 1 actually change?

Three things. It confirmed the ladder timing varies across events, which is the only reason
Phase 4 is worth attempting. It showed sell-through has a long left tail — the median is
51% — which means the interesting risk question is the left tail of the profit
distribution, not the average, and that shaped Phase 5. And it made me define "tickets
sold" once, in one function, that every later phase imports, so the number can't quietly
mean two things in two places.

---

## Phase 2 — Demand forecast

### 1. MAE of 53 tickets on the holdout. Is that good?

It's 17% MAPE against a naive baseline's 25%, so a third better than what the business
actually does, which is "the last few nights in that room did about 300". That's the only
comparison that means anything — an MAE with no baseline is a number, not a result.

Two caveats I'd give before you ask. First, that's the *final-year* holdout, which is the
honest number; the cross-validated figure is 43 and it flatters me, because random folds let
the model interpolate a year it has partly seen. Second, it's an upper bound on the error,
not a floor: every academic-calendar feature is deliberately missing until my
pre-registration is frozen, and the calendar is a big driver in student events.

### 2. Why OLS? Everyone else brings a gradient booster.

117 events. The model already spends 13 parameters. A booster would fit the noise
beautifully and I'd have no held-out set big enough to catch it — small n is the binding
constraint on this whole project, and being visibly honest about it is the signal.

The other reason is that I have to defend every number in this repo out loud. The whole
model is one formula and a coefficient table with confidence intervals. That's worth more
to me than a point of MAPE I couldn't explain.

### 3. How do you know there's no look-ahead?

I don't assert it, I test it — and the test is behavioural, not a check on column names,
because you can defeat a name check by renaming a column.

I rebuild the entire feature frame from transactions truncated at each event's first paid
sale, which is the decision moment, and require it to come out identical. Anything that
secretly reads later sales moves when the later sales are removed. There's also a test that
plants a leak — it puts `sell_through` in the feature list — and proves the check catches
it. A check nobody has ever seen fail is decoration.

The one feature that touches purchase timestamps at all is lead-to-announce, and it's the
minimum timestamp, so truncating at the minimum can't change it.

### 4. Your marketing coefficient is 0.41. So doubling marketing spend gets you 40% more tickets?

No, and that's the most important sentence in the phase. Marketing spend is endogenous — I
spend more on the events I already believe in. The coefficient is picking up my own beliefs
about the night as much as any effect of the advertising. It predicts brilliantly and it
tells you nothing about what would happen if I changed it.

The same fact rules it out as an instrument in Phase 4, for exactly the reason a bigger
artist guarantee is ruled out: it moves with demand, so it fails the exclusion restriction
outright.

### 5. Why hold out the final year rather than just cross-validate?

Because the model's real job is to predict a season that hasn't happened, and random folds
don't test that. The final-year holdout forces the brand-growth trend to extrapolate one
year past its data, which is the model's most fragile assumption — and growth is the trap in
§8.9, the thing that will masquerade as a pattern if it isn't controlled.

I report both. The holdout being worse than the cross-validated number is exactly what
should happen. If it weren't, I'd go looking for a leak.

### 6. Why model log tickets rather than tickets?

Because the outcome is multiplicative. Sales run from 57 to 1,001 across these events, and
a 50-ticket error is a disaster in a 150-cap room and a rounding error in a 1,500-cap one.
Logs make the error proportional, which is how the business thinks about it, and they make
every coefficient read as a percentage effect.

The catch is that `exp()` of a log-scale prediction gives the conditional median, not the
mean — the mean is bigger by roughly exp(σ²/2). I report Duan's smearing factor, which is
1.011 here, and leave it out of the headline because MAE and MAPE are both minimised by the
median. Phase 5 wants the mean profit, so it needs the correction.

---

## Phase 3 — Sales curve

### 1. Explain presale velocity in one sentence.

Events sell in a predictable shape, so if I know a typical event has sold 72% of its final
total with a week to go, and this one has sold 400, I forecast about 550 — and on held-out
events that's accurate to about 16%, tightening to 11% by the day before.

### 2. Why not Bass diffusion? That's the standard model.

Because Bass has three parameters and one of them, market potential, is the thing I'm trying
to forecast. I'd be fitting a non-linear curve to a partial series, with about 24 events per
segment, to read off a number that a single division already gives me. And it can fail to
converge, which is then my problem too.

The empirical curve makes one assumption — this event is shaped like the median of previous
events — and I can check it directly by printing the spread of the curves. It's wide: p10 to
p90 is about 0.39 of the final total at day −7. That assumption is doing real work and it's
only roughly true. Bass wouldn't fix that, it would hide it behind a fit statistic.

One nice property: the pointwise median of monotone curves is monotone, so the median curve
is a valid cumulative curve for free. No smoothing, nothing to tune.

### 3. At day −7 your curve forecast is worse than your Phase 2 model. Why are you showing me this?

Because it's true, and because it's the interesting part. The pricing-time model knows the
room, the artist fee, the night and the brand's trajectory. A week out, that's still worth
more than watching the presale. The curve overtakes it at day −3 and wins clearly by day −1.

The right conclusion isn't "which model is better" — they use different information. It's
that they should be combined: the pricing-time forecast as the prior, the presale as the
update. That's Phase 5. If I'd asserted from memory that presale beats the prior at a week
out, I'd have been wrong, and I'd have found out in an interview instead of in a test.

### 4. Your segmentation by brand improves nothing. Why keep the code?

Because reporting that is the finding. Averaged over the three decision days, segmenting by
brand moves MAPE by about a tenth of a point — noise. So the honest recommendation is to use
the pooled curve.

I keep the mechanism because on real data a segment may genuinely differ, and I keep the
number in the report because a split that buys nothing still costs degrees of freedom and
will eventually pick up noise and call it a segment. Being able to say "I tried the obvious
split and it did nothing" is worth more than quietly deleting it.

### 5. An event sells out four days before the door. What does your forecast do?

It under-forecasts demand, and it should — but you have to be careful what you're claiming.
The curve flattens because there's nothing left to sell, not because people stopped wanting
tickets. Scaling that curve up estimates **sales**, not **demand**.

That distinction is harmless here and fatal in Phase 6, where the question is what a higher
price would have earned. A sold-out event's true demand is censored at capacity, and any
counterfactual that treats observed sales as demand will understate what the room could
have taken. It's on the limitations list for exactly that reason.

### 6. Where does the accuracy actually come from?

Mostly the numerator, and I say so in the report. By the day before, the median event has
banked 83% of its final total, so the forecast is scaling up a small remainder — the curve
isn't getting cleverer, there's just less left to guess. Quoting the day −1 MAPE as a
modelling achievement would be dishonest. The day −7 number is the one that's doing work,
and it's the one operational decisions actually get made on.

---

## Phase 4 — Elasticity

The first five are five of SPEC.md §10's nine questions — the ones the CV bullet invites,
so the answers are written to be said in one breath. §10's other four do not belong to a
single phase and are answered in **the last section of this file**.

### 1. Why not just regress quantity on price?

Because I set the prices. I priced off the same information that drove demand — big artist,
good date, freshers' week, so I charged more and more people came. Price is correlated with
the error term, so the coefficient is the elasticity plus a bias, and on my data the bias is
bigger than the elasticity: the pooled regression comes out **+0.70**, significantly
positive. Higher prices "causing" more sales.

I report that rather than hiding it, because it is the finding that motivates everything
else. And I don't fix it by dropping variables until the sign turns over — that's p-hacking
my own business.

### 2. So how *did* you identify elasticity?

Within-event variation. Each event sells several tiers at different prices — Early Bird,
Tier 1, Tier 2, Door — same artist, same date, same room, same marketing. Put a dummy in for
every event and everything constant within the night is absorbed; beta comes off the ladder.

But fixed effects alone are **not** enough, and that's the part I'd fail on if I hadn't
checked. Within an event, price is nearly the same variable as time-to-event: Early Bird is
cheap and early, the door price is expensive and last-minute. Fixed effects don't touch
that, because timing varies within the event exactly the way price does. On my fixture,
event FE alone gives **−1.26** against a truth of **−0.80** — over-elastic by more than
half. Add a lead-time control and it lands at **−0.815**.

What makes it identified at all is the 6.5% of price variation that survives both the event
dummies and lead time: ladders applied differently across events, tiers opened at different
lead times, prices corrected mid-window. Standard errors clustered by event.

### 3. What if it isn't identified?

Then I say so, and I say what would fix it. That's a written criterion, not a judgement
call: the residual price variation has to clear 2%, the clustered 95% interval has to be
narrower than 1.0 and to sit below zero, and there have to be at least 30 events with two or
more priced tiers. Those are constants at the top of the module, written before I looked at
the estimate.

If they fail, the bullet is: *"documented why price elasticity is not cleanly identifiable
from prices set by the operator, and specified the release-tier randomisation required to
recover it"* — and the randomisation isn't hypothetical, it's running. Whole events get a
HIGH or LOW ladder from a pre-committed cryptographic coin, blocked on brand and venue,
baseline recorded before the assignment is revealed.

### 4. Is that uplift real or modelled?

**Modelled.** Nobody was charged those prices. It's a model-implied uplift in contribution
margin from re-pricing ladders on a grid, and I say "projected" before anyone asks.

And I'd add the caveat before that one: with an inelastic estimate, the optimum sits on the
edge of whatever grid I allow, so the ±30% bound is my assumption about what a student night
can be re-priced by, not something the data implies. The number I'd actually act on is the
one at the experiment's +15% dose, which is a price change the business already agreed it
can live with.

### 5. What was your break-even on that booking?

Fixed stack over contribution per ticket. On the biggest booking in the synthetic
programme: a £16k stack against £13 a head is **1,239 tickets in a 1,500 room** — 83% of the
room before a penny of profit. And P(clear) is the number I care about more, because
break-even is a point and demand is a distribution: I bootstrap the demand forecast's
out-of-sample residuals and count how often we clear.

The number I take into the phone call is the inverse: the largest fee that still leaves a
50/50 chance of clearing.

### 6. Your fixed-effects estimate is −0.815 and the truth is −0.80. Isn't that just the fixture agreeing with itself?

Partly, and I'd say so first. It rules out one class of error — the estimator being broken —
and not the class that matters, which is whether the real data has clean variation.

What stops it being wholly circular is that the generator is written as a story about the
business, not as the estimating equation rearranged: the operator prices off their demand
expectations, the venue's rate card gets passed into the ticket price, Early Bird is cheap
and early. And there's a knob that turns the lead-time confound off, so I can watch which
estimators start and stop working. The FE-only estimate being wrong on this fixture is the
useful part — it's the mistake I'd otherwise have shipped.

### 7. You had a cost-shifter instrument. Why isn't that your headline?

Because it failed honestly, in both directions, and the pair of failures is more informative
than either.

The venue hire rate is a clean cost shifter on the mechanism — the buyer neither knows nor
cares what the room cost me. But hire rates drift up year on year and so does a growing
brand's demand, so uncontrolled it's partly a time trend, and a time trend is a demand
shifter: run it that way and the first stage is strong (F = 15.6) and the estimate comes out
wrong-signed. Control for the academic year and it's excludable but weak — F = 2.8, below the
relevance floor of 10 — so I report no estimate at all, because a weak instrument is biased
back towards the OLS estimate it exists to fix.

And I keep the artist guarantee in the table as the counterexample: it has the strongest
first stage of the three, F = 97, and it's the least usable, because a bigger guarantee
means a bigger artist means more demand. Relevance is testable; validity isn't. The strong
number is the one to distrust.

### 8. Your VIF is 15. Isn't that a fail?

It's a flag, and it's reported. A VIF only tells you how much the standard error is
inflated — it says nothing about whether the interval you end up with is usable. With 402
panel rows the clustered interval is ±0.10, which is usable, so the criterion I bind on is
the interval width, not the VIF. A VIF of 15 with a tight interval and a VIF of 15 with a
useless one are not the same situation, and one number can't tell them apart.

The number I'd actually defend is the one underneath it: 6.5% of the variance of log price
survives the event fixed effects and the lead-time control. That's the entire diet of the
estimator, and if it fell under 2% I'd call it not identified.

### 9. You've got 117 events and you're clustering on event. How many parameters are in that regression?

Two — and that's the point of the question. I fit the fixed effect by demeaning within
event, not with 117 dummies, so the regression is `log_p` and `lead_term` against 117
clusters. If I'd left the dummies in it would be 119 parameters from 117 clusters, and a
cluster-robust covariance matrix is a sum of 117 outer products, so its rank is at most 116.
It's singular. You can still read a number off it, and that number is what I used to report.

Two things follow from clusters being the sample size. The interval is a t on G−1 = 116
degrees of freedom, not a normal — 1.981 against 1.960, small here but it's the honest
reference distribution. And the two forms give exactly the same slope, −0.8152975 either
way, so nothing about the point estimate is at stake.

The uncomfortable half of the answer: switching narrowed my interval, from 0.248 wide to
0.209. The raw cluster-robust variance is identical — the whole gap is the finite-sample
factor `(N−1)/(N−K)`, which counts the absorbed dummies in K, and whether they belong there
is the same argument as Stata's `areg` versus `xtreg, fe`. So I print both intervals and I
print whether the choice could have changed the verdict. It couldn't: both widths are inside
the 1.00 criterion by a factor of four and both intervals sit entirely below zero.

### 10. Your identification criteria include "the interval must lie below zero". So you're only ever "identified" when you get the answer you wanted?

That's the right question and the report answers it under its own heading, because three of
those criteria are one kind of thing and that one isn't.

Residual price variation, interval width and event count are properties of the **design**.
I can evaluate all three without ever looking at what beta came out as, and they're what
decides whether this data can answer the question at all. The sign check conditions on the
answer. A +0.30 estimate with a tight interval and plenty of residual variation would be
called NOT-IDENTIFIED by my rule even though every design criterion passed — and
identification is a property of the design, not of the sign.

Why I keep it anyway: the confound I'm removing has a known direction. The operator priced
off the same demand signal that drove sales, and that pushes the estimate up — the pooled
regression is +0.698. So a positive estimate surviving the fixed effects isn't evidence that
demand slopes upward; it's a measurement that the confound is still in there. "Not
identified" is the honest label for a number still carrying the bias I was trying to remove.

And the difference between that and p-hacking is what gets *reported*. P-hacking is
searching specifications until the sign comes out and showing only the winner. My
wrong-signed pooled estimate is row 1 of the table, the over-elastic FE-only estimate is
row 2, all three instruments are in there including the one that's invalid by construction,
and the thresholds were constants in the file before I looked at the estimate. The test is
whether a failed sign check gets published as NOT-IDENTIFIED — and SPEC §4.4 plus the
NOT-IDENTIFIED branch of the report are the commitment to that.

---

## Phase 5 — Break-even, P(clear) and deal structure

### 1. Walk me through the three deal structures.

Flat guarantee: I pay a fixed fee and keep everything above it, so I carry all the demand
risk both ways. Door split: the artist takes a percentage of net, so the risk is shared —
capped downside, capped upside. And "guarantee **or** a percentage of the door, whichever is
greater", which is the standard contract.

That third one pays me the **minimum of the other two on every single night** — that's an
identity, not an approximation, and there's a test that asserts it pointwise. Below the
crossover it behaves like the flat fee; above it, like the split. The artist takes the
better of their two outcomes, so I take the worse of mine: the full downside of the
guarantee deal and the capped upside of the split. The only compensation for that is a lower
fee, and if the fee is the same as a straight guarantee I'm being paid nothing for the
difference.

### 2. Why bootstrap the residuals instead of assuming a distribution?

Because the spread I care about is the spread of mistakes the forecast actually makes, and
the tail isn't symmetric. A wet Wednesday in exam season is not the mirror image of a good
night. Fitting a normal to those residuals would smooth away exactly the part that decides
whether a booking is survivable.

They're **out-of-sample** residuals, from Phase 2's folds grouped by event, so no night is
ever scored by a model that had seen it. In-sample residuals would be too small, and too
small in the direction that loses money.

And that claim is pinned now, which it wasn't. A review swapped `rng.choice(residuals)` for
`rng.normal(0, residuals.std())` and the entire suite stayed green — the report said "no
normal distribution is fitted to anything" and nothing checked it. The test that fixes it is
one line of set arithmetic: an empirical bootstrap can only ever produce `exp(log_pred + r)`
for an `r` that's literally in the residual array, so I assert the distinct simulated ticket
counts are exactly that set. A parametric draw produces a continuum and fails immediately.

### 2b. How many Monte Carlo draws, and how do you know that's enough?

40,000 per event, and I know because the run measures it rather than because I picked a
round number. Every run simulates the whole programme twice on two bootstrap seeds and
prints how far the numbers moved: the worst event's 5th-percentile night shifts by about
£30, the median event by £0.

I had this wrong before. The report used to assert "4,000 draws gives a 5th percentile
stable to about a pound" and it wasn't true — at 4,000 the worst event moved £184 between
seeds. It was a checkable numeric claim that nobody had checked, in a project whose whole
thesis is not making those.

The more useful half of the answer is which uncertainty that *isn't*. Monte Carlo error
shrinks like 1/√draws, so it's the cheap one. The residual distribution is atomic — one
residual per event, about 120 of them — so a simulated 5th percentile can only land on one
of 120 values, and no draw count adds information about the tail. That's why I quote these
percentiles as an order of magnitude, and why the median move across seeds is £0 while the
worst event's is £30: between seeds it's usually the same atom and occasionally one step
away.

### 3. How do you decide guarantee versus split?

Take the guarantee when it pays more in expectation **and** the 5th-percentile night is a
loss the business can absorb. That's risk-adjusted, not expected-value, and "can absorb" is
a number written in the module rather than a feeling. Where the guarantee wins on the mean
but the bad night is too big, the recommendation flips to the split — I'm buying insurance,
and its price is the expected profit I hand over.

### 4. Your synthetic P(clear) is zero almost everywhere. What went wrong?

Nothing, and I'd flag it before you asked. The generator was built to make statistical
properties visible — a wrong-signed pooled regression, a recoverable elasticity — and its
cost stack was never calibrated against its ticket prices: the median break-even is about
1.1 times the median tickets sold, so almost every simulated night loses money on tickets
alone.

So on the fixture I read the shapes rather than the levels: how the three structures order
themselves, how the tail behaves, that the fee curve is monotone and lands where the
quantile says it should. The levels wait for the real cost stack, and the report says that
in a box rather than showing you a friendlier example.

### 5. Every distribution here excludes the cancelled events. How much does that cost you?

More than I can quantify, and that's the honest answer rather than a hedge. Every
probability in the phase is conditional on the night going ahead. Three events were pulled
or downscaled — the report counts them from the data rather than stating a number.

What I can't tell you is why each one was pulled, because their presale histories were
destroyed on the platform. The rows that would say how badly they were selling don't exist,
and no analysis recovers them. If poor sales are part of what gets a night pulled — and on
my own account they sometimes are — then the true left tail is worse than the one I've
drawn, and by an amount nobody can currently put a number on.

I had a stronger version of that sentence in the report, saying the pulled events were
pulled *because* they were selling badly. That's a claim about the real programme that the
dataset can't support, so it's gone. A known, unmeasured bias stated as unmeasured is worth
more than a confident sentence I can't back.

---

## Phase 6 — The counterfactual

### 1. What's your uplift number?

On the synthetic fixture, **+2.6% in contribution margin, projected**, from moving every
ladder by the +15% the business has already agreed it can live with. That is the headline
because it is the only uplift in the phase with no search in it: one price change, decided
in advance, evaluated once.

Two things I'd say in the same breath. It's modelled, not realised — nobody was charged
those prices. And there *is* a bigger number, +5.3%, from optimising each event over a
13 × 7 grid of level and spread multipliers; I don't lead with it because 100% of events
pick the top edge of that grid, so it's the grid bound talking, not the data. The report
prints it as "at least +5.3% within the ±30% range considered", underneath the headline
rather than as it.

### 2. Why is your optimum always at the boundary? Isn't that a broken optimiser?

No, it's arithmetic. Constant-elasticity revenue goes as `price^(1+beta)`, so with
`|beta| < 1` revenue rises with price without limit and the model stops only where I stop
it. The ±30% grid bound is my assumption about what a student night can be re-priced by
before it becomes a different product.

What would make the optimum interior is a model with a choke price, or a demand curve whose
elasticity rises with price, or an attendance-value term for bar spend and repeat custom.
None of those are supportable with ~120 events, so instead of inventing one I state the
bound and report the uplift at a dose the business already agreed to.

### 3. What happens to that number if the elasticity isn't identified?

It stops existing. Not "gets wider" — stops existing, because an uplift is an elasticity
times a price change. What I report instead is a curve of uplift against assumed elasticity,
with every row labelled with the assumption it rests on, and the point estimate marked
provisional.

And the curve is the argument for the experiment: it's U-shaped and touches zero at
beta = −1, so either side of that the model wants to move prices in *opposite* directions.
The double-digit uplift at the elastic end and the double-digit uplift at the inelastic end
aren't the same claim at different confidence — one is telling me to cut prices and the
other to raise them. They're contradictory advice wearing similar-looking numbers. That's
why the randomised ladder experiment matters more than any estimator choice.

I'll add the thing I got wrong there, because it's the kind of thing you'd find: those two
percentages used to be typed into the report by hand. They happened to match the table on
one seed and didn't on the next, so the report was contradicting itself. They're read off
the curve now, with a test that compares the sentence to the table on two seeds.

### 4. How do you know the counterfactual isn't nonsense?

Three checks. At the do-nothing grid point the counterfactual reproduces the actual
contribution exactly, so the machinery isn't adding anything. The optimum can never be worse
than doing nothing, because doing nothing is on the grid. And on the fixture I push the
chosen ladders back through the generator's **own** demand process — softmax reallocation
across tiers, the total moving with the average log price, the rounding and the sell-out cap
— and compare: **+5.3% model against +8.7% true process**, same sign, same order of
magnitude, pinned as a test across seeds.

The gap is explainable rather than mysterious: my model moves each tier by its own price
change, so it's implicitly quantity-weighted, while the generator moves the total by the
unweighted average. Flattening a ladder raises the cheap rungs, which carry most of the
units, so the quantity-weighted view predicts a bigger loss of sales. My model is the
conservative one, which is the direction to be wrong in.

### 5. Your model says to flatten the ladder. Would you actually do that?

No, and the report says not to. That's the model naming a question it can't answer rather
than giving advice. I fit one elasticity for every rung, so tiers have no separate
willingness to pay — and then the ladder's shape is decided entirely by whether that one
number is above or below −1. Inside unit elasticity the objective is concave in log price
and the model collapses the ladder to a single price; outside it, it's convex and the model
stretches the ladder as far as I let it. My own curve does both, at −0.8 and at −1.2. A
recommendation that flips on the sign of `1 + beta` is not a recommendation.

Early-bird buyers and door buyers are different people and this model can't see the
difference. Separating tier-level elasticities needs more within-event price variation than
exists — the same constraint that made the main estimate hard in the first place.

---

## SPEC.md §10 — the four questions that don't belong to one phase

Plus the survivorship question, which DECISIONS.md (2026-07-17) records as the
non-deferrable half of the cancelled-events decision: the reconstruction can wait, this
answer and the README paragraph cannot.

### 1. How do you know it's not overfit?

Because nothing is scored on data it was fitted to, and the splits are by **event**, never
by transaction. A random transaction split puts the same night in train and test, so I'd be
"predicting" an event I had already seen — SPEC §8.1, and the easiest way to fool yourself
on this dataset. So: grouped five-fold CV by event, plus a final-year holdout, and I report
both numbers. **43 tickets MAE cross-validated, 53 on the final-year holdout.** The holdout
being worse is what should happen; if it weren't, I'd go looking for a leak.

The second half of the answer is that the model is too small to hide in. 117 events, 13
parameters, plain OLS. A gradient booster would fit the noise beautifully and I'd have no
held-out set big enough to catch it — with ~120 rows, small n is the binding constraint on
the whole project, and being visibly honest about that *is* the signal (§8.5).

And it propagates: Phase 5's demand distribution is a bootstrap of Phase 2's **out-of-fold**
residuals, so every probability I quote is built out of mistakes the forecast actually made
on nights it had never seen. Not a fitted normal, and not in-sample error.

### 2. How did you avoid data-mining the calendar effects?

I wrote the list down before I opened the data. I've run ~150 of these events, so I don't
need to *discover* that freshers' week is enormous and revision week is dead — I lived it.
That list is `preregistration.md`, and it still says DRAFT, which is not an oversight: the
real-data path in this repo **refuses to load while it says DRAFT**. Every number I have
today came off a synthetic fixture, and the ordering — list first, data second — is the
thing the guard exists to protect (§5.3).

The rules travel in the file with the list, so they can't drift: test exactly those patterns
and nothing else; **Benjamini–Hochberg FDR across the whole list, corrected p-values
reported**, because 18 tests at p < 0.05 on pure noise gives me a ~60% chance of a
"finding" (§5.1); fit on the first four years and confirm on the fifth; report effect sizes
with uncertainty, not bare significance; and control the academic-year trend before
believing any calendar effect, because both brands grew and anything correlated with time
inherits that growth (§8.9).

Bonferroni would be the wrong tool here — ~18 correlated tests is exactly the case it is too
conservative for. And the discipline has teeth in the code: Phase 1 prints no p-value, no
correction and no significance claim anywhere, and there's a test that scans the report body
for that vocabulary. A pattern that isn't on the list is a hypothesis for next time.

### 3. Can you separate freshers' week from the loan drop?

**No — and nobody could with this data.** I'd say that before I said anything else.

Freshers' week, the start of semester, the student-loan instalment and the good weather
arrive together, within days of each other, every September, for five years running. They
are the same handful of dates repeated five times. With ~120 events there is no case in the
sample where one moves and the others don't, so it isn't a power problem I can estimate my
way out of — there is nothing in the data to estimate. Any coefficient I put on "freshers'
week" would be all four effects wearing one label.

So they are pre-registered as **one grouped "start of semester" effect**, reported
undecomposed, one line on the list rather than four (§5.6).

What would actually separate them, in order of how gettable it is:

- **A year where the instalment date moved** relative to term start. SLC/SAAS payment dates
  do shift, and one displaced cohort breaks the alignment for that year.
- **Cross-institution variation in term dates.** Two universities whose terms start a
  fortnight apart put freshers' week and the loan date on different weekends in the same
  season — that's independent variation without waiting for anything.
- **Weather is the separable one in principle**, because it varies within a week rather than
  across a season, but it needs an external daily series the repo doesn't have yet.

Naming that limit before an interviewer finds it is worth more than any coefficient I could
report instead.

### 4. What would you do with more time?

Randomise the tier prices — the one move nobody else applying can make, because I set the
prices (§9). It isn't hypothetical: it's designed, frozen and running
(`experiment/DESIGN.md`). Whole **events** get the HIGH or LOW ladder, never tiers within an
event, because otherwise two buyers see different prices for the same show and I have a
fairness problem and a contaminated experiment. Arms are ±15%, assignment is a pre-committed
cryptographic coin blocked on brand and venue, headliners are excluded, and the acceptable
downside per arm was written down before the first on-sale.

That is what pins beta. The observational estimate lives or dies on the 6.5% of price
variation that survives event dummies and the lead-time control; the experiment *makes*
price variation my own demand expectations did not cause. When it reads out, one row of
Phase 6's uplift curve stops being conditional and becomes the answer.

Three cheaper things behind it, in order: resolve `fee_treatment` — one real export row
against one order confirmation — because every pound in Phases 5 and 6 rests on that split
(§8.6); get the real per-head cost into the contribution, which moves every break-even; and
reconstruct the cancelled events (below).

### 5. Did you ever cancel or downscale an event that was selling badly? Where is it in the data?

Yes, a handful over five years — one to five, depending on whether a downscale counts. They
are the most informative rows I have and **they are not in the dataset**, so I state it
rather than wait to be caught: every model in this repo is **conditioned on the night having
gone ahead** (§8.4).

Two things follow, and they are different. The first is that my left tail is optimistic by an
unknown amount. If poor sales are part of what gets a night pulled — and on my own account
they sometimes are — then the worst nights are precisely the ones missing from the
distribution I bootstrap. The second is that I cannot check that story, because the presale
histories of the pulled events were **destroyed on-platform**: the rows that would say how
badly they were selling do not exist, and no analysis recovers them.

So what I do is bound it rather than fix it. The reports carry the count from the data, not
from memory, and say the bias is real and unquantified — a known, unmeasured bias stated as
unmeasured is worth more than a confident sentence I can't back. The README carries the same
paragraph, because a stranger reading the finding needs to know what the sample is
conditioned on. And ingest keeps a `cancelled`-events hook, so the day I reconstruct them
from bank records and the brand sheets they enter as flagged rows rather than as a rewrite.

If you want the number that would change my mind: if the pulled events were selling at half
the rate of the ones that ran, they'd sit in the bottom decile of my residual distribution,
and the 5th percentile I quote in Phase 5 would move out by roughly the width of that decile
per pulled event. That's arithmetic I can do; what I can't do is confirm the premise.
