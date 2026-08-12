# Phase 5 — Break-even, P(clear) and deal structure

**SYNTHETIC DATA — nothing here is real (Brand A/B, venues V1-V4, FAKE ids).**

Source: `synthetic`, seed `20260811`
Regenerate: `uv run python -m pricing.phase5 --synthetic --seed 20260811`


**The question:** an agent offers an artist for a fixed fee. What has to happen for that night to make money, how likely is it, and should the deal be a flat fee or a share of the door?

## Method, in four steps

1. **Contribution per ticket** = face value banked per paid ticket, less any per-head cost (£0.00 — see assumptions). Comps are excluded: a guestlist is not demand (SPEC.md 8.7).
2. **Break-even attendance** = fixed cost stack ÷ contribution per ticket. The fixed stack is venue + security + production + staffing + marketing + the artist guarantee.
3. **A demand distribution, not a point estimate — and it is Phase 2's, not a new one.** Phase 5 fits no model. It takes Phase 2's grouped-K-fold **out-of-fold** predictions (every event scored by a model that never saw it — SPEC.md 8.1) and resamples their residuals with replacement. No normal distribution is fitted to anything: the spread of simulated outcomes is the spread of mistakes the forecast actually made.
4. **40,000 draws per event**, each one priced through all three deal structures.

### How much of this is just the random number generator?

A simulated percentile is worth nothing without knowing how much it wobbles when you re-run it, so the whole thing is run twice — same events, same residuals, different bootstrap seed — and the gap is printed rather than described:

| number | median move across seeds | worst event |
|---|---|---|
| 5th-percentile night (fixed guarantee) | £0 | £31 |
| expected profit (fixed guarantee) | £2 | £22 |
| P(clear) | 0.000 | 0.004 |

The median move is small and the worst event is not, and the reason is worth knowing: the residual distribution is **atomic** — 117 residuals, one per event — so a simulated 5th percentile can only ever land on one of that many values. Between seeds it is usually the same value and occasionally one step away, and the size of that step is set by the data, not by the draw count.

**This is the smaller of the two uncertainties here, and the only one more draws can fix.** The one they cannot: that same atomic residual distribution is the entire evidence about the tail. The 5th percentile is being read off a handful of genuinely bad nights, and resampling them ten million times adds no information about how bad a night can get. Quote these percentiles as an order of magnitude.

## The demand distribution behind every probability here

- source: **Phase 2** (`pricing.phase2`), `117` out-of-fold predictions, one per event
- out-of-fold mean absolute error: 42.7 tickets (14% MAPE, 0.143 log points)
- the centre of each event's distribution is its **out-of-fold** prediction, so the bootstrap is centred on what a model that had never seen the night would have said, not on a fit that already knew the answer
- **price is deliberately not a feature of that forecast.** In a predictive model its coefficient is our own pricing rule, not a demand response (SPEC.md 4.1), and a model that learns 'higher prices sell more' is precisely the model that must never choose a price. Price enters this project once, as Phase 4's beta, applied on top of this forecast by Phase 6.
- two demand models in one repo means two answers to 'how many will come', and the day they disagree is the day the report stops being trustworthy. There is one.

## The biggest bookings

| event | capacity | guarantee | contribution/ticket | break-even | break-even sell-through | actually sold | **P(clear)** | most payable at 50% |
|---|---|---|---|---|---|---|---|---|
| FAKE-EV-020 | 1500 | £7,762 | £13.13 | 1239 | 83% | 537 | **0%** | **£0 — no fee clears** |
| FAKE-EV-072 | 800 | £7,299 | £18.85 | 688 | 86% | 726 | **20%** | £5,375 |
| FAKE-EV-112 | 800 | £7,268 | £17.14 | 795 | 99% | 493 | **3%** | £3,160 |
| FAKE-EV-027 | 800 | £6,755 | £13.36 | 886 | 111% | 442 | **0%** | £1,401 |
| FAKE-EV-024 | 800 | £6,175 | £8.40 | 1245 | 156% | 426 | **0%** | **£0 — no fee clears** |
| FAKE-EV-110 | 800 | £5,925 | £13.17 | 892 | 112% | 369 | **0%** | £1,915 |

Across all 117 events: median break-even is 586 tickets (113% of the room), median P(clear) is **0%**, and 117 events were more likely than not to lose money at the price they were sold at.

The last column is the one to take into the phone call: the largest fee that still leaves a 50% chance of clearing. It is the (1 − target) quantile of simulated `Net` — no solver, one line — **clamped at zero**, because the quantile goes negative on a night that is more likely than not to lose money before the artist is paid anything, and 'the most you can pay is minus a thousand pounds' is not a number you can take into a call. **£0 means no fee clears at this room**, not even a free booking. Median across the programme: £0 — no fee clears the 50% target on at least half the programme, against a median guarantee actually paid of £1,126.

> **Read the synthetic P(clear) numbers as machinery, not as a P&L.** The generator was built to make statistical properties visible (a wrong-signed pooled regression, a recoverable elasticity) and its cost stack was never calibrated against its ticket prices: the median break-even is 2.4x the median tickets actually sold, so almost every simulated night loses money and P(clear) sits at ~0. What is worth reading here is the *shape* — how the three structures order themselves, how the tail behaves, how the most-payable fee falls out of the draws. The levels wait for the real cost stack.

### What the fee is worth — FAKE-EV-072

The event with the most headroom in the programme — the one that could pay the largest fee and still be a 50% shot. Picked by that rule, not by which chart looked best.

The same simulation, run across a range of artist fees. This is the table to have open during the call: it turns "can you do £2,000?" into a probability instead of a feeling. `P(clear)` is one minus the empirical distribution of `Net`, so it can only fall as the fee rises — a property worth checking, and there is a test that does.

| artist fee | P(clear) | expected profit |
|---|---|---|
| £0 | 100% | £5,389 |
| £1,820 | 98% | £3,569 |
| £3,650 | 83% | £1,739 |
| £5,470 | 45% | £-81 |
| £7,300 | 20% | £-1,911 |
| £9,120 | 3% | £-3,731 |
| £10,950 | 0% | £-5,561 |
| £12,770 | 0% | £-7,381 |
| £14,600 | 0% | £-9,211 |

## The three deal structures (SPEC.md 6.3)

Writing `Net` for what the night makes before the artist is paid — ticket contribution minus every fixed cost except the artist:

| structure | your profit |
|---|---|
| fixed guarantee | `Net − F` |
| door split | `(1 − s)·Net`, s = 70% |
| guarantee OR s% of door, whichever is greater | `Net − max(F, s·Net)` |

The third line is the standard live-music deal and it is worth reading twice: it pays **the minimum of the other two, on every single night**. Below the crossover (`Net = F / s`) it behaves exactly like the flat fee; above it, exactly like the split. The artist takes the better of their two outcomes, so you take the worse of yours — the full downside of the guarantee deal *and* the capped upside of the split deal. The only compensation for that is a lower F; if the F is the same as a straight guarantee, you are being paid nothing for the difference.

| structure | who carries the risk | mean E[profit] | mean 5th-percentile night | share of events profitable in expectation |
|---|---|---|---|---|
| fixed guarantee | you, all of it | £-2,709 | £-3,456 | 0% |
| door split (70% to artist) | shared both ways | £-302 | £-526 | 17% |
| guarantee OR 70% of door, whichever is greater | you keep the worse of the two | £-2,712 | £-3,456 | 0% |

The fixed guarantee beats the split in expectation on **0 of 117** events, but after the tail test only **0** are recommended as guarantees.

### The decision rule

> Take the guarantee when `E[profit | guarantee] > E[profit | split]` **and** the left tail is survivable.

Survivable is a number, not a feeling: `MAX_SURVIVABLE_LOSS = £1,500` on one night. Where the guarantee wins on the mean but its 5th-percentile night is worse than that, the recommendation flips to the split — you are buying insurance, and its price is the expected profit you hand over.

## Assumptions, all of them

| assumption | value | where it comes from |
|---|---|---|
| variable cost per ticket | £0.00 | no per-head line exists in the cost stack yet. If one appears, it moves every break-even here |
| artist share of net door | 70% | **not in the data** — it is in the contracts. Used identically for every event so the structures compare like for like |
| survivable one-night loss | £1,500 | a business judgement, written down so it can be argued with |
| bootstrap draws | 40,000 | enough that re-running on a different RNG seed moves an event's 5th-percentile night by at most £31 (median £0) — measured, see below |
| bootstrap RNG seed | 12345 | a chosen number, like every other row here. `--seed` re-rolls the synthetic fixture, NOT the Monte Carlo; the second run at seed 13344 is what measures the row above |
| promoter keeps face value, platform keeps the booking fee | `fee_treatment` is **UNKNOWN** | SPEC.md 8.6. This is the one assumption every pound in this report rests on, and it is cheap to resolve: compare one real export row against one order confirmation |

## What this does not model

- **Bar spend.** For a promoter with a bar deal, tickets are not the whole margin, and a night that loses money on tickets can still be worth running. None of that is here.
- **Cancellation.** Cancelled events are excluded (SPEC.md 8.4), so every distribution here is conditional on the night going ahead. 3 events were pulled or downscaled. Why each one was pulled is **not something this dataset can check**: their presale histories were destroyed on-platform (DECISIONS.md, survivorship), so the rows that would say how badly they were selling do not exist. If poor sales are part of what gets a night pulled — and on the operator's own account they sometimes are — then the true left tail is worse than the one drawn here, and by an amount nobody can currently quantify. That is a known, unmeasured bias, not a rounding error.
- **Correlation between events.** Each night is bootstrapped independently. Two events a week apart eat each other's audience (SPEC.md 8.10), so a portfolio built by adding these distributions up would understate a bad month.
- **The residual distribution is the model's, not the world's.** It has as many draws as there are events (117). The 5th percentile is being read off a few dozen bad nights, and it should be quoted as an order of magnitude.

## Reproduce

```
uv run python -m pricing.phase5 --synthetic --seed 20260811
```

`--seed` re-rolls the fixture. The bootstrap is separate and fixed at `BOOTSTRAP_SEED = 12345`; to see how much of any number here is the RNG, change it in `phase5.py` — or read the two-seed table above, which is that experiment already run.
