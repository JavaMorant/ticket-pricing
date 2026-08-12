# Phase 4 — Price elasticity of demand

**SYNTHETIC DATA — nothing here is real (Brand A/B, venues V1-V4, FAKE ids).**

Source: `synthetic`, seed `20260811`
Regenerate: `uv run python -m pricing.phase4 --synthetic --seed 20260811`


**The question:** if I raise ticket prices 10%, how many fewer tickets do I sell?

**The problem:** I set the prices. SPEC.md section 4 is about nothing else.

## All four estimates, including the embarrassing one

| estimator | beta | SE (clustered by event) | 95% CI (t, G-1 df) | rows | clusters | params |
|---|---|---|---|---|---|---|
| 1. naive pooled OLS (SPEC 4.1) | **+0.698** | 0.099 | [+0.502, +0.893] | 402 | 117 | 2 |
| 2. event FE only (SPEC 4.2) | **-1.260** | 0.029 | [-1.317, -1.203] | 402 | 117 | 1 |
| 3. event FE + lead-time control (SPEC 4.3) | **-0.815** | 0.053 | [-0.920, -0.711] | 402 | 117 | 2 |
| 3b. the same, written with event dummies | **-0.815** | 0.063 | [-0.939, -0.691] | 402 | 117 | 119 |
| 4. IV — I. venue hire rate, trend controlled (PRE-SPECIFIED) | _not run_ | — | — | 117 | 117 | — |
| 4. IV — II. venue hire rate, NO trend control (diagnostic only) | **+0.366** | 0.199 | [-0.024, +0.755] | 117 | 117 | — |
| 4. IV — III. artist guarantee (INVALID by construction — shown deliberately) | **+1.008** | 0.084 | [+0.843, +1.173] | 117 | 117 | — |
| **TRUE value (synthetic only)** | **-0.800** | — | — | — | — | — |

## 1. Why the naive regression is wrong (SPEC.md 4.1)

Pooled across every tier of every event, log quantity on log price gives **+0.698 (SE 0.099, 95% CI [+0.502, +0.893])**.

Positive. Higher prices 'cause' more sales. That is not a bug and it is not fixed by dropping variables until the sign flips — doing that is p-hacking your own business (SPEC.md 8.3). It is what you should expect from prices an operator set using the same information that drove demand: big artist, good date, freshers' week, so we charged more **and** more people came. Price is correlated with the error term. Textbook simultaneity.

## 2. Event fixed effects (SPEC.md 4.2)

One fixed effect per event. Everything constant within an event — the artist, the date, the venue, the hype, the marketing spend, the weather — is absorbed, and beta is left to be identified off **within-event** price variation: the tier ladder.

That alone gives **-1.260**. Adding the lead-time control gives **-0.815**. The difference between those two numbers is section 3.

### How the fixed effect is fitted, and why it matters for the interval

The regression is run as the **within transform** — subtract each event's own mean from `log_q`, `log_p` and `lead_term`, then fit with no intercept — rather than by putting a dummy per event on the right-hand side. Frisch-Waugh-Lovell says those are the same regression, and on this panel they are: rows 3 and 3b of the table above give -0.815297490 and -0.815297490, a difference of 5.6e-16 — floating-point noise. There is a test that asserts it.

What is not the same is the parameter count, and with standard errors clustered by event that is not cosmetic. The dummy version estimates **119 parameters from 117 clusters**. A cluster-robust covariance matrix is a sum of one outer product per cluster, so its rank is at most G-1 = 116 — with 119 parameters it is singular, and each event dummy is being fitted off the 3.4 rows in its own cluster. The within version estimates **2 parameters from 117 clusters**, which is the footing cluster-robust inference actually assumes.

Both intervals are printed because the switch narrows the interval and that should be visible rather than discovered: within gives [-0.920, -0.711] (width 0.209), dummies give [-0.939, -0.691] (width 0.248). The raw cluster-robust variance is identical; the whole gap is statsmodels' finite-sample factor `(N-1)/(N-K) * G/(G-1)`, which counts all the absorbed dummies in K. Whether they belong there is genuinely unsettled — it is the same disagreement as Stata's `areg` versus `xtreg, fe` — so the verdict below is quoted on the within interval and the wider one is one row above it.

**The choice cannot have bought the verdict:** both widths (0.209 and 0.248) are inside the 1.00 CI-width criterion, and both intervals lie entirely below zero (-0.711 and -0.691). Every criterion lands the same way on either interval.

Both intervals use a **t distribution on G-1 = 116 degrees of freedom** — critical value 1.981, read straight off the fitted interval, against the normal's 1.960. Cluster-robust inference has as many independent pieces as there are clusters, not as there are rows.

## 3. The lead-time confound (SPEC.md 4.3)

Within an event, tier price is nearly the same variable as time-to-event. Early Bird is cheap **and** early; the door price is expensive **and** the night itself. Two stories are tangled: (a) Early Bird sells more because it is cheaper — the price effect we want; (b) Early Bird sells more because eager buyers buy early and it is the only thing on sale.

| diagnostic | value | reading |
|---|---|---|
| panel rows | 402 | one per (event x tier) |
| events with 2+ priced tiers | 117 | 3.44 tiers each |
| price variance explained by event alone | 0.743 | what the fixed effects take out |
| price variance explained by event + lead time | 0.935 | what is left is the estimator's entire diet |
| **residual price variation** | **0.065** | the identifying variation, and nothing else is |
| VIF(log price) | 15.4 | 1 / the row above, by definition |
| VIF(lead time) | 4.4 | the same collinearity from the other side |
| within-event corr(log price, lead term) | -0.865 | -1.0 would mean price IS timing |
| partial R2 of price | 0.558 | of the demand variation the event and timing leave behind, how much price explains |

## Verdict: **IDENTIFIED**

The criteria are not four of the same thing, so they are not printed as one table. Three of them are properties of the **design** — they ask whether this dataset can answer the question at all, and none of them looks at what the estimate came out as. The fourth asks about the **sign**, and that is a different kind of claim.

### Design criteria — can this data answer the question?

| criterion | value | threshold | pass |
|---|---|---|---|
| residual price variation | 0.065 | >= 0.020 | PASS |
| 95% CI width | 0.209 | <= 1.000 | PASS |
| events with 2+ priced tiers | 117 | >= 30 | PASS |

- *residual price variation* — share of log-price variance surviving event dummies + lead time — this is the only variation beta is estimated from
- *95% CI width* — an interval wider than this cannot tell a harmless price rise from a ruinous one
- *events with 2+ priced tiers* — the within estimator needs ladders, not events

### A sign check, stated as such

| criterion | value | threshold | pass |
|---|---|---|---|
| CI lies below zero | -0.711 | < 0.000 | PASS |

**This criterion conditions on the sign of the answer, and that has to be said out loud.** A within-event estimate of +0.30 with a tight interval and plenty of residual price variation would be labelled NOT-IDENTIFIED by this rule, even though the design criteria above all passed. Identification is a property of the design and the variation; the sign of the estimate is not.

The reason it is still a criterion is a specific one, and it is not 'we expected a negative number'. Demand curves slope down — that is the one thing about this problem nobody argues with, and the entire purpose of the fixed effects is to remove a confound (SPEC.md 4.1: the operator priced off the same demand signal that drove sales) whose known direction is to push the estimate UP. So a surviving positive estimate is not a discovery about demand; it is a measurement that the confound is still in there, and the honest label for a number contaminated by a confound you were trying to remove is 'not identified'.

Why that is not the same as p-hacking the sign: p-hacking is *searching* over specifications until the sign comes out right and then reporting only the winner. Everything searched here is printed — the pooled estimate with the wrong sign is the first row of the table, the FE-only estimate that is too elastic is the second, all three instrument specifications are shown including the invalid one, and the criteria were fixed in `IDENTIFICATION_CRITERIA` before the estimate was looked at. The test is whether a failed sign check would be *reported* as NOT-IDENTIFIED rather than quietly re-specified around, and SPEC.md 4.4 plus the `NOT-IDENTIFIED` branch of this report are what commit to that.

- *CI lies below zero* — a positive or zero-straddling interval means the confound is still in there; it does not mean demand slopes upward. THIS CHECK CONDITIONS ON THE SIGN and is reported separately for that reason

**Flags (reported, not fatal):**
- VIF(log price) = 15.4, above the textbook rule-of-thumb 10. Flagged, not failed: a VIF only says the standard error is inflated, and whether the resulting interval is usable is the CI-width criterion.

The criteria are in `IDENTIFICATION_CRITERIA` at the top of `src/pricing/phase4.py`, and they were written before the estimate was looked at. That ordering is the whole value of them.

### What that means

Within-event price variation identifies an elasticity of **-0.815 (SE 0.053, 95% CI [-0.920, -0.711])**: a 10% price rise sells about 8% fewer tickets, with a plausible range of 7% to 9%.

This is identified off the price movement that is neither the event nor the timing — ladders applied differently across events, tiers opened at different lead times, prices corrected mid-window. **It is still observational.** It rests on the assumption that whatever made those ladders differ was not itself a demand signal. The randomised experiment in `experiment/DESIGN.md` is what removes that assumption, and it is running now.

## 4. Cost-shifter instrument (SPEC.md 4.5)

The idea: find something that moves **price** but not **demand**. Demand-side hunches move both, which is the whole problem — but the cost stack only moves price. The venue put its hire rate up; that went into the ticket price; the buyer neither knows nor cares what the venue charged us.

This is a different regression on a different panel: one row per event, not one per tier. A cost shifter varies **across** events, so an event fixed effect would absorb it entirely. Fixed effects and instruments are two different answers to the same question, not two steps of one answer.

Relevance floor: first-stage **F >= 10** on the excluded instrument. Below it, no estimate is computed at all.

| spec | instrument | controls | first-stage F | dP/dZ | IV beta | OLS beta | balance p | trend share |
|---|---|---|---|---|---|---|---|---|
| I. venue hire rate, trend controlled (PRE-SPECIFIED) | `log_hire` | `C(venue) + C(academic_year)` | 2.8 | +0.331 | not run | +0.875 | 0.848 | 0.299 |
| II. venue hire rate, NO trend control (diagnostic only) | `log_hire` | `C(venue)` | 15.6 | +0.630 | **+0.366** | +0.883 | 0.248 | 0.299 |
| III. artist guarantee (INVALID by construction — shown deliberately) | `log_guarantee` | `C(venue) + C(academic_year)` | 96.5 | +0.214 | **+1.008** | +0.875 | 0.000 | 0.084 |

`dP/dZ` is the first stage: log price per log pound of hire rate. `balance p` tests whether the instrument, after its controls, still predicts demand-side observables (the artist guarantee as a proxy for how big the act was, and marketing spend). `trend share` is the share of the instrument — net of which venue it is — explained by the academic year.

### Reading the exclusion restriction honestly

**Exclusion cannot be tested with one instrument.** That is what makes it an assumption. A small balance p-value is evidence *against* exclusion; a large one is not evidence *for* it. What can be argued is the mechanism, and here it is:

- **Venue hire rate — passes on the mechanism.** A buyer sees a poster with a price on it. Nothing on that poster says what the room cost us. The hire rate moves our price and reaches demand through no other channel.
- **...but only conditional on the year.** SPEC.md 8.9: hire rates drift up year on year, and so does a growing brand's demand. An instrument with a time trend is an instrument correlated with the growth trend, and the growth trend is a demand shifter. Spec II is in the table to show exactly that — drop the year control and the first stage gets much stronger and the estimate goes the wrong way. That is not a better instrument, it is a broken one with more power.
- **Artist guarantee — fails outright.** A bigger guarantee means a bigger artist means more people come. It moves price *and* demand. Spec III has the strongest first stage in the table and is the least usable specification on it, which is the lesson: relevance is testable and validity is not, so the strong number is the one to distrust.

**Outcome: 2SLS is NOT reported.** The pre-specified instrument is excludable but weak — first-stage F = 2.8, below the floor of 10. Controlling the year trend is what makes it excludable and it is also what takes most of its strength away, and there is no version of this that gets both. The scaffold stays in the repo because the moment a real hire-rate shock lands — a venue that puts its rate card up 25% in one year — it becomes the best identification in the project.

## What this cannot identify, whatever the verdict says (SPEC.md 5.6)

- **Not the elasticity of a different price level.** Every estimate here is local to the ladders actually used. Extrapolating to a doubling of prices is not a modelling choice, it is a fabrication.
- **Not separate elasticities per tier.** Early-bird buyers and door buyers are different people with different willingness to pay, and a single beta averages them. Splitting it would need more within-event price variation than exists.
- **Not freshers' week versus the loan instalment versus the weather.** They arrive together every year, for five years. Nobody could separate them with ~120 events, so they are reported as one grouped start-of-semester effect and explicitly not decomposed.
- **Not cannibalisation.** Two events a week apart eat each other's audience (SPEC.md 8.10). Clustering by event does not fix that; the observations are not independent across events either, and the standard errors above quietly assume they are.
- **Not anything about buyers who never bought.** The panel only contains tiers that sold at least one ticket, at events that happened. Cancelled events are excluded and counted.

## Provisional while `fee_treatment` is UNKNOWN (SPEC.md 8.6)

Elasticity acts on **the price the buyer sees**, face value plus booking fee. Until the per-platform fee treatment is confirmed against a real order confirmation, `price` is a guess of the buyer's price, and every number in this report inherits that.

## Reproduce

```
uv run python -m pricing.phase4 --synthetic --seed 20260811
```

Companions: `docs/EXPLAINERS.md` (why these methods) and `docs/VIVA.md` (the questions this invites, with answers).
