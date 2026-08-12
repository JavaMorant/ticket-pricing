# Phase 6 — Counterfactual pricing backtest

**SYNTHETIC DATA — nothing here is real (Brand A/B, venues V1-V4, FAKE ids).**

Source: `synthetic`, seed `20260811`
Regenerate: `uv run python -m pricing.phase6 --synthetic --seed 20260811`


**The question:** what would an optimal tiered release schedule have earned across these events, versus what was actually charged?

## Headline: **+2.6% projected uplift in contribution margin** at the experiment's pre-committed +15% dose

At the Phase 4 elasticity of **-0.815** (verdict: IDENTIFIED), moving every rung of every ladder by +15% — the dose the business has already agreed it can live with (`experiment/DESIGN.md`) — would have changed contribution by +2.6% across 117 events.

**This is the headline because it is the only uplift here with no search in it.** One price change, decided in advance, evaluated once. The optimised number below is larger and means less: with an inelastic estimate the optimiser walks to whatever bound the grid sets, so it reports the bound.

**Projected, not realised.** Nobody was charged these prices. It is what a model says would have happened, and the difference between that and a realised number is the difference between a research finding and a fabrication (SPEC.md 6.4). The CV wording is *projected*, and it says so before anyone asks.

### Sensitivity: what an unconstrained search over the grid would have picked

Optimising each event over all 91 candidate ladders gives +5.3%.

**Corner solution — this is the grid bound, not a result.** 100% of events pick a level at the edge of the grid (modal level x1.30), which is what `price^(1+beta)` does when demand is inelastic: revenue rises with price without limit and the model stops only where the grid stops. Read it as *at least +5.3% within the x0.70–x1.30 ladder range considered*, and note that the range is a judgement about what a student night can be re-priced by, not something the data implies. Widening the grid moves this number and nothing else.

## The grid (there is no optimiser here)

- **level** — multiply every rung by the same factor: `0.70 .. 1.30` in steps of 0.05 (13 values)
- **spread** — multiply each rung's distance from its own ladder's average, in logs: `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` (1.00 leaves the ladder as it was, 0.00 collapses it to a single price)
- 91 candidate schedules per event, 117 events, 402 tier rows. Every candidate is a row in a table you can print.

At each candidate: quantity moves as `units * exp(beta * change in log price)`, the face value moves by the same proportion as the buyer price, and the event's total is capped at its licensed capacity with every tier scaled back proportionally. At (level 1.00, spread 1.00) every change is zero and the counterfactual reproduces what actually happened — which is the arithmetic check that the machinery is not quietly adding something.

## The uplift curve

| assumed beta | optimised uplift | median event uplift | at the experiment's +15% | modal level | what it wants | modal spread | at the grid edge |
|---|---|---|---|---|---|---|---|
| -0.4 | **+18.9%** | +18.7% | +8.7% | x1.30 | raise prices | x0.00 | 100% |
| -0.6 | **+12.0%** | +11.9% | +5.7% | x1.30 | raise prices | x0.00 | 100% |
| -0.8 | **+5.8%** | +5.7% | +2.8% | x1.30 | raise prices | x0.00 | 100% |
| -1.0 | **+0.0%** | +0.0% | +0.0% | x0.70 | cut prices | x0.25 | 30% |
| -1.2 | **+6.7%** | +7.5% | -2.8% | x0.70 | cut prices | x1.50 | 81% |
| -1.5 | **+17.1%** | +19.9% | -6.7% | x0.70 | cut prices | x1.50 | 71% |
| -2.0 | **+34.5%** | +42.2% | -13.0% | x0.70 | cut prices | x0.00 | 59% |

**Read the 'what it wants' column before the uplift column.** The curve is U-shaped, and it touches zero at beta = -1.0 for a reason that is pure arithmetic: at unit elasticity, revenue does not respond to the level of prices at all, so there is nothing to gain. Either side of that point the model wants to move prices in **opposite directions** — raise them if demand is inelastic, cut them if it is elastic. So the +34.5% uplift at beta = -2.0 (which gets there by telling you to cut prices) and the +18.9% uplift at beta = -0.4 (which gets there by telling you to raise prices) are not the same claim with different confidence: they are contradictory pieces of advice that happen to carry a similar-looking number. That is exactly why an unidentified beta cannot produce a headline number, and why the experiment matters more than any estimator choice.

## Two things in this table that are arithmetic, not discoveries

**1. The optimum sits on the edge of the grid.** At the Phase 4 beta, 100% of events pick a level at the grid boundary. That is not a discovery, it is `price^(1+beta)`: when demand is inelastic (`|beta| < 1`) revenue rises with price without limit, so the model wants infinity and stops only where the grid stops. **The +/-30% bound is an assumption about what a student night can be re-priced by without becoming a different product — it is not something the data implies.** The model has no mechanism for a brand becoming known as expensive, for a competitor undercutting, or for the bar spend of the people who stopped coming. That is why the '+15%' column exists: it is a dose the business has already agreed it can live with (`experiment/DESIGN.md`), so the uplift there is a number with a use rather than a corner solution.

**2. The ladder's shape is chosen by the same single number** — here the model wants to flatten the ladder (modal spread x0.00). Also arithmetic: with `|beta| < 1` the objective is concave in log price, so spreading prices apart can only lose money and the model collapses the ladder to one price; with `|beta| > 1` it is convex and the model stretches the ladder as far as the grid allows. Neither is advice. One beta for every rung means tiers have no separate willingness to pay, and early-bird buyers and door buyers are different people this model cannot tell apart. **Do not read the flat-ladder result as advice to abolish early-bird pricing.** Read it as the model naming the question it cannot answer — separating tier-level elasticities needs more within-event price variation than exists (SPEC.md 4.3, 5.6).

## What the schedule actually looks like at beta = -0.815

- total actual contribution: £338,254
- total counterfactual contribution: £356,198
- programme uplift at the grid optimum — a bound, see the headline: **+5.3%** (median event +5.2%)
- modal chosen ladder: level x1.30, spread x0.00 (100% of events at the grid edge)
- THE HEADLINE NUMBER — uplift from the experiment's +15% dose alone, no optimisation: **+2.6%**

## Ground-truth check (synthetic only)

The chosen ladders were pushed back through the generator's **own** demand process — softmax reallocation across tiers, event total responding to the average log price, the generator's floor and its sell-out cap — using the true elasticity of -0.80 rather than the estimated -0.815.

| | uplift |
|---|---|
| model-implied (constant elasticity, tier by tier) | +5.3% |
| the generator's true demand process | +8.7% |
| ratio | 0.61 |

The two agree in sign, and the gap between them has a specific cause worth knowing: the model moves each tier's quantity by that tier's own price change, so it is implicitly quantity-weighted, while the generator moves the event's total by the **unweighted average** change across the ladder. Flattening a ladder raises the cheap rungs (which carry most of the units) and cuts the expensive ones, so the quantity-weighted view sees a bigger average price rise than the unweighted one and predicts a bigger loss of sales. The model is the conservative one of the two here, which is the direction to be wrong in.

The ratio is what the model's simplification costs. There is a pinned test on both, with a wide tolerance on purpose: agreement to three decimal places would only prove that two nearly identical formulas are nearly identical.

**What this proves:** applying one elasticity tier by tier does not throw the answer away. **What it does not prove:** that the real world is a constant-elasticity world, that ladder shape has no effect, or that beta is what this fixture says. Nothing run on a fixture can prove any of those.

## What this counterfactual cannot see

- **Buyers moving between tiers.** Change the gaps and some Early Bird buyers become door buyers, or stop coming. That needs a tier-specific elasticity and the data cannot support one.
- **Time.** A ladder is a schedule, not just a set of prices, and the sales curve (Phase 3) is what says whether a tier had time to sell. This grid re-prices the rungs; it does not re-time them.
- **The competition, and the brand.** Nothing here knows that a rival promoter runs the same night, or that a night which gets a reputation for being expensive sells worse next term at any price.
- **Anything past the room.** Bar spend, repeat custom and the value of a full room to the next booking are all outside the objective, and all three argue for lower prices than this model wants.
- **The events that did not happen.** Cancelled events are excluded (SPEC.md 8.4), so this is an uplift conditional on the night going ahead.

## Provisional while `fee_treatment` is UNKNOWN (SPEC.md 8.6)

Demand responds to the price the buyer sees; the till receives the face value. The split between the two is a guess until one real export row is compared with one real order confirmation, and every pound in this report rests on it.

## Reproduce

```
uv run python -m pricing.phase6 --synthetic --seed 20260811
```
