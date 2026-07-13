# Ticket Pricing & Demand Model — Handover Spec

**Written:** 13 July 2026
**For:** future me, picking this up cold after the August diet
**Repo name:** suggest `ticket-pricing` (not `events-analytics` — see §2)

---

## 0. Read this first (60 seconds)

You are building **a research finding**, not a piece of software.

The deliverable is a repo with a notebook and a README that has **numbers in it**. Not a dashboard. Not a platform. The GUI comes at the very end, it takes a weekend, and it is a footnote.

You will want to start with the interface. Everyone does. **Don't.** The reason is in §2 and it is a business reason, not a CV reason.

**The one-line goal:**

> What would an optimal tiered release schedule have earned across my events, versus what I actually charged — and what is the probability a given artist guarantee gets cleared?

If you can answer those two questions with defensible numbers, the project is done.

---

## 1. Why this project exists

**Two goals, both real:**

| | Goal | Deliverable |
|---|---|---|
| **CV** | Prove I can produce a research finding and defend it | Repo + README + two CV bullets |
| **Business** | Price Brand A/Brand B events better; stop guessing on guarantees | A tool I actually use |

The CV goal is the binding one, because it's the one with a deadline (interviews, Oct–Dec 2026) and it's the one that's harder.

**Why this beats every other project on the list:** every quant applicant has a pairs backtest and an 8-K sentiment model. Nobody else has five years of proprietary ticketing data from two businesses they personally own, priced with their own money. It is the only thing on the CV that cannot be copied.

**But the data isn't why it's good.** It's good because it forces a genuinely hard identification problem (§4) that has no clean answer. Wrestling with that honestly is the thing that gets you hired.

---

## 2. Build order — and why the GUI is last

**A pricing tool is a wrapper around a demand model.**

If you build the wrapper first, you will end up pricing real events — with real artist guarantees, real venue deposits, real money — using a model whose elasticity estimate is confounded and which you never validated out of sample.

You would be making **worse decisions with more confidence**, because now there's an interface showing you a number.

That is the actual risk. Not the CV. The £150K.

```
Phase 0  Data assembly        ← the tedious part. It is also the moat.
Phase 1  Descriptives         ← look at it before you model it
Phase 2  Demand forecast      ← predictive. Useful immediately.
Phase 3  Sales curve          ← how tickets accumulate over time
Phase 4  Elasticity           ← THE HARD ONE. This is the CV bullet.
Phase 5  Risk / Monte Carlo   ← break-even, P(clear guarantee)
Phase 6  Counterfactual       ← "what should I have charged?"
Phase 7  The tool             ← one weekend. Streamlit. One file.
```

Phases 0–6 are the project. Phase 7 earns you exactly one clause on the CV:
*"packaged as an internal tool now used to price live events."*

That's a good clause. It says the work went into production. **It is not a bullet.**

---

## 3. The data

### 3.1 What you export

**Fixr** is the primary source (Youni/TBC secondary — decide early whether to include or note the exclusion).

Transaction-level export. Expect roughly:

| field | notes |
|---|---|
| `order_id` | |
| `event_id` / event name | join key — normalise these, they will be inconsistent |
| `ticket_type` / tier name | "Early Bird", "Tier 1", "Release 2"… **normalise to an ordinal** |
| `price_paid` | **face value or face+fee? Find out. It matters — see §8.6** |
| `booking_fee` | |
| `purchased_at` | timestamp. This is the whole sales curve. Guard it. |
| `quantity` | |
| `promo_code` | if present — **these are gold, see §4.3** |

### 3.2 What Fixr does NOT have — and what makes this project possible

**The cost side.** None of it is in the ticketing platform. You have to reconstruct it from your own records — artist contracts, venue invoices, bank statements, security and production quotes.

Per event you need:

- `capacity` (licensed, not aspirational)
- `artist_guarantee` (£ paid, fixed)
- `deal_structure` — **guarantee / door-split / "guarantee OR % of door, whichever is greater"** (see §6.3)
- `venue_cost`, `security_cost`, `production_cost`, `staffing_cost`
- `marketing_spend`
- `date`, `venue`, `city`, `brand` (Brand A / Brand B)
- `cancelled` flag — **including events you pulled or scaled down (see §8.4)**

> **This reconstruction is the single most tedious part of the project and it is also the entire moat. Without the cost side there is no contribution margin, no break-even, and no P(clear). Do it first, do it properly, and never do it again.**

### 3.3 Derived tables

- **`events`** — one row per event, all metadata + costs above.
- **`transactions`** — one row per ticket, joined to `events`.
- **`event_tier`** — one row per (event × tier): price, units sold, window open/close. **This is the panel that Phase 4 runs on.**

---

## 4. The core problem: you set the prices

This is the intellectual heart of the project. Understand it before you write any modelling code.

### 4.1 Why the obvious thing fails

You want elasticity. The obvious move:

```
log(Q_e) = α + β·log(P_e) + ε_e
```

You expect β < 0 — higher price, fewer tickets.

**You will very likely get β > 0.** Higher prices "cause" more sales.

That is not a bug. It's because **you set the prices, using the same information that drives demand.** Big artist, good date, freshers' week → you charged more *and* more people came. Price is correlated with the error term. This is textbook simultaneity bias, and it is exactly why you cannot estimate demand from observational market data.

**When you see a positive β, do not "fix" it by dropping variables until it turns negative.** That is p-hacking your own business. Write it down as a finding and move on to §4.2.

### 4.2 The workhorse: event fixed effects

You have **multiple tiers within each event**. Early Bird → Tier 1 → Tier 2 → Door. Different prices, same event.

```
log(Q_et) = α_e + β·log(P_et) + γ·X_et + ε_et
```

`α_e` absorbs everything constant within the event — the artist, the date, the venue, the hype, the marketing spend. β is now identified off **within-event** price variation.

### 4.3 The confound that will bite you, and it is fatal if you miss it

**Within an event, tier price is nearly collinear with time-to-event.**

Early Bird is cheap **and** early. Door price is expensive **and** last-minute.

So two stories are tangled together:

- **(a)** Early Bird sells more because it's cheaper. ← the price effect you want
- **(b)** Early Bird sells more because eager buyers buy early and it's the only thing on sale. ← selection/timing

**These are not separable from tier structure alone.** You must do one of:

1. **Control for lead time explicitly** and check whether any residual price variation survives. It probably won't — they're near-collinear. Test it (VIF, or just look).
2. **Exploit variation in tier *timing* across events.** Did you ever release Tier 2 earlier for one event and later for another? That gives you price variation *at a given lead time*. This is your best shot.
3. **Find natural experiments.** Did you ever mis-price and correct it? Run a promo code for reasons unrelated to demand? Announce a tier at the wrong price? **Any price variation that wasn't driven by your demand expectations is gold.** Grep for it.
4. **Admit it isn't identified.** ← completely legitimate, see below.

### 4.4 If you can't identify it — say so, and say what would fix it

With ~100–150 events you may simply not have the variation. **Do not force it.**

The honest bullet is not the weaker bullet:

> *"Documented why price elasticity is not cleanly identifiable from prices set by the operator, and specified the release-tier randomisation required to recover it."*

That is the same move as the deflated Sharpe. A candidate who says *"the naive regression said higher prices cause higher sales, which is obviously backwards, here's why, and here's the experiment that would settle it"* is a research hire. A candidate who reports a confident elasticity from confounded data is not.

### 4.5 The upgrade almost nobody thinks of: cost shifters as instruments

You need something that moves **price** but does **not** move demand directly. Your demand-side hunches move both, which is the whole problem. But your **cost stack** only moves price.

- The venue put its hire rate up. You passed it into the ticket price. **The buyer does not know or care what the venue charged you.**
- Security or production costs jumped for one event. Same logic.
- An artist's fee moved for reasons unrelated to their pull (agent, schedule, exchange rate).

That is a **cost-shifter instrument** — exactly what industrial-organisation economists use to estimate demand when price is endogenous. Two-stage least squares: regress price on the cost shifter, then quantity on the *fitted* price.

**Test the exclusion restriction honestly.** A bigger artist guarantee is *not* a valid instrument, because a bigger guarantee means a bigger artist means more demand — it fails exclusion outright. Venue hire rate is much cleaner. Say which of yours pass and which don't, and why.

This is a harder, better answer than event fixed effects, and if you can pull it off it is the most sophisticated thing on your CV.

---

## 5. Problem 2: you will find patterns that aren't there

> *"I want it to find patterns and trends."*

You will find them. That is the problem. With ~150 events and a rich feature set, **the data will hand you patterns whether or not they exist.**

This is the same disease as the stat arb backtest, and you already know the cure — it's on your CV. *"Treated backtest overfitting as the primary risk... a deflated Sharpe ratio correcting for the number of trials."* Same discipline. Different dataset.

### 5.1 The arithmetic

Test 20 candidate patterns at p < 0.05 on pure noise:

```
P(at least one false positive) = 1 − 0.95^20 = 0.64
```

**A 64% chance of a spurious "finding."** And you will test more than 20 — every calendar effect, every day-of-week, every interaction. Left unchecked, you are guaranteed to discover something that isn't real, and it will be the thing you put on your CV.

### 5.2 The cardinal sin

**Never use the same data to both generate and confirm a hypothesis.**

You look at the data. You notice Tuesdays sell well. You then "test" whether Tuesdays sell well — on the same data. Of course it's significant. You picked it *because* it looked significant. That is a circle, not a finding.

### 5.3 The move: pre-specify from domain knowledge

Here is the thing nobody else applying to these firms can do.

**You have run ~150 of these events. You already know the patterns.** You don't need to discover that freshers' week is enormous and revision week is dead — you lived it.

So: **write the list down from memory, before you open the data.**

That converts a fishing expedition into a **pre-registered study**. It is not a workaround; it is a stronger design than data-mining could ever be, and it is powered by the exact thing that makes you unique — you are the operator, not an analyst who was handed a CSV.

Then test *exactly those* and nothing else. If a pattern wasn't on the list, it's a hypothesis for next time, not a result.

### 5.4 Your candidate list (start here, edit from memory)

Academic calendar:
- Freshers' week
- Semester start / mid / end
- Revision and exam periods
- Reading week
- End-of-term release
- Vacation vs. term time
- Ball season

Money:
- **Student loan instalment dates** — the student-finance body pays in instalments. That is a real, near-exogenous spending shock and it is probably the single best feature nobody would think of. Get the actual dates.

Timing:
- Day of week (Wed/Thu/Fri/Sat are *not* interchangeable for student events)
- Lead time / days-to-event
- Days since your last event → **cannibalisation, see §8.9**

Event characteristics:
- Artist billing tier
- Venue, city, capacity
- Brand (Brand A vs. Brand B)
- Competing promoters' nights — you know these; nobody else could reconstruct them

Environment:
- Weather (matters for Event C and anything outdoor)
- Academic-year cohort effects

That's ~18. Which is precisely §5.1's problem.

### 5.5 What to do about it

1. **Correct for multiple testing.** Benjamini-Hochberg FDR is the right tool here — Bonferroni is too conservative for ~18 correlated tests. Report *corrected* p-values.
2. **Hold out events.** Fit on the first four years, confirm on the fifth. A pattern that doesn't survive out-of-sample is not a pattern.
3. **Report effect sizes, not just significance.** "Freshers' week adds 180 tickets ± 60" is a finding. "p = 0.03" is not.

### 5.6 The limitation you must state out loud

**Freshers' week ≈ semester start ≈ the loan drop ≈ good weather.** These arrive together, every year, for five years.

**You cannot separate them with 150 events.** Nobody could. Don't pretend otherwise — group them into one "start of semester" effect and *say* that you can't decompose it further, and what data would let you.

Naming your own identification limits before an interviewer finds them is worth more than any coefficient you could report.

---

## 6. The models

Keep these **separate in your head**. They are different tasks with different standards of evidence.

### 6.1 Demand forecast (predictive — Phase 2)

Given what you knew *at the time of pricing*, predict final tickets sold.

**Features:** artist billing/tier, day-of-week, **academic-calendar structure** (freshers' week, term boundaries, exam periods, reading week), venue capacity, city, brand, lead time, marketing spend, historical brand performance.

**Metric:** MAE / MAPE on **held-out events**.

**No causal identification needed.** It just has to predict. This is also the piece the business needs most.

### 6.2 Sales curve (Phase 3)

How do tickets accumulate over time? Usually an S-curve. Fit cumulative sales as a function of lead time (Bass diffusion, or something simpler that works).

**Why it matters more than it looks:**
- Forecast final sales from partial presale → **this is "presale velocity" and it's already on your CV**
- Decide *when* to drop the next tier
- Decide when to panic and spend on marketing

Easier than elasticity, and arguably the most operationally useful thing here.

### 6.3 Risk / Monte Carlo (Phase 5) — the quant centrepiece

```
break_even_attendance = fixed_cost_stack / contribution_per_ticket
P(clear) = P(Q > break_even_attendance)
```

Fit a demand *distribution* (not a point estimate), then Monte Carlo it against the cost stack. Report per booking:

- break-even attendance
- P(clearing the guarantee)
- the full profit distribution — mean, and the left tail

**Then the decision that makes this a trading project:**

| structure | your payoff | who holds the risk |
|---|---|---|
| **Fixed guarantee** | `R(Q) − F − Costs` | you hold all demand risk |
| **Door split** | `(1−s)·(R(Q) − Costs)` | shared; capped downside, capped upside |
| **"Guarantee OR x% of door, whichever is greater"** | `R(Q) − max(F, s·Net) − Costs` | **you are short an option to the artist** |

That third row is the standard live-music deal. If any of your contracts are structured that way, **that is literally a short call and you should price it as one.** Do not write the words "short option" on your CV — a quant reader will spot the structure instantly and enjoy spotting it. Just model it correctly and let them.

**The decision rule:** take the guarantee when `E[profit | guarantee] > E[profit | split]` *and* the left tail is survivable. That's a risk-adjusted choice, not an expected-value one. Say so.

### 6.4 Counterfactual pricing backtest (Phase 6)

The headline number.

Given the fitted demand model and the cost stack, optimise the tier schedule (price × timing × allocation) under the capacity constraint. Compare to what you actually charged.

**Output: `X% uplift in contribution margin`.**

Be careful: this is a *model-implied* uplift, not a realised one. **Say "projected" on the CV.** Claiming a realised uplift you never ran is the same sin as a fabricated Sharpe.

---

## 7. Definition of done

The CV bullets **are** the acceptance criteria. Every `[FILL]` is a task. If you can't fill it, that piece isn't finished.

- [ ] `[N]` events, `[M]`k transactions — Phase 0
- [ ] Cost stack reconstructed for every event — Phase 0
- [ ] Demand forecast MAE on held-out events — Phase 2
- [ ] Elasticity β **with an identification strategy you can defend** — Phase 4 *(or a written statement of why it isn't identified)*
- [ ] Break-even attendance + P(clear) per booking — Phase 5
- [ ] `[X]%` projected uplift in contribution margin — Phase 6
- [ ] README that a stranger can read in 5 minutes and understand the finding

**Then and only then:** Phase 7, the Streamlit tool. Inputs: date, venue capacity, artist guarantee, cost stack. Outputs: demand distribution, break-even, P(clear), recommended tier schedule. One file. One weekend.

---

## 8. Traps — read before Phase 2

1. **Split by event, never by transaction.** A random transaction split puts the same event in train *and* test. You'd be "predicting" an event you've already seen. **`GroupKFold(groups=event_id)`.** This is the single easiest way to fool yourself here.

2. **Look-ahead.** Only use information available *at the moment you'd make the decision*. Final sell-through cannot be a feature for predicting sell-through. Obvious when stated; easy to do by accident.

3. **Don't p-hack the sign.** See §4.1.

4. **Survivorship.** Did you ever cancel or downscale an event that was selling badly? Those are your failures, and they're missing from the data. If you only model events that happened, you're conditioning on success. **Dig them out. They're the most informative rows you have.**

5. **Small n.** ~30 events/year × 5 years ≈ 100–150 events. That is *small*. Do not fit a 40-feature gradient booster on 120 rows. Report confidence intervals. Be visibly honest about what the sample can and cannot support — that honesty *is* the signal.

6. **Face value vs. booking fee.** Elasticity acts on **the price the buyer sees**, not your take. If Fixr adds a fee on top, the buyer's price is face + fee. Get this right or every number downstream is wrong.

7. **Comps and guestlist are not demand.** Strip them, or model them separately.

8. **Normalise tier names.** "Early Bird", "EARLY BIRD", "Tier 0", "First Release" are the same thing. This will eat an afternoon. Budget for it.

9. **The growth trend will masquerade as a pattern.** Your brand got bigger over five years. So *anything* correlated with time picks up that growth. If you started running Thursday events more recently, "Thursday" will look brilliant — because it is secretly proxying for "recent." **Control for year / brand maturity, or detrend, before you believe a single calendar effect.**

10. **Cannibalisation.** Two events a week apart eat each other — same audience, same finite student budget. Your events are *not* independent observations. Add `days_since_last_event` and `events_in_trailing_14d` as features, and be aware this violates the independence your standard errors assume.

---

## 9. The stretch goal that would make this exceptional

**You control the prices. So randomise them.**

Randomise release-tier pricing across the autumn programme and recover a **clean, causally identified elasticity from an actual experiment** — on your own business, with your own money at stake.

No graduate applicant does this. Even if you haven't run it by interview time, saying:

> *"The next step is randomising tier prices across the autumn events, which I can do because I set them"*

is an answer that stops a trader mid-question.

**Design note:** randomise at the *event* level (whole events get the high or low tier ladder), not within event — otherwise buyers see different prices for the same show and you have a fairness problem and a contaminated experiment. Block on artist tier and term-week so the arms are balanced.

---

## 10. Interview prep — the questions you are inviting

Write bullets you *want* to be asked about. These are the questions those bullets invite. Have the answers cold.

| Question | Where the answer lives |
|---|---|
| "Why not just regress quantity on price?" | §4.1 — because I set the prices |
| "So how *did* you identify elasticity?" | §4.2–4.3 — event FE, and the lead-time confound |
| "What if it isn't identified?" | §4.4 — say so, propose the experiment |
| "What was your break-even on Event C?" | Have the number. Fixed stack ÷ contribution per ticket. |
| "How do you know it's not overfit?" | §8.1 — grouped CV by event, held-out events, small-n honesty |
| "Is that uplift real or modelled?" | **Modelled.** Say it before they ask. |
| "How did you avoid data-mining the calendar effects?" | §5.3 — pre-specified from operating experience, FDR-corrected, confirmed out-of-sample |
| "Can you separate freshers' week from the loan drop?" | §5.6 — **no**, and here's what would |
| "What would you do with more time?" | §9 — randomise the autumn tier prices |

---

## 11. If you only do one thing

Pull the Fixr export and reconstruct the cost stack for every event.

Everything else is downstream of that, and it's the part nobody else on earth can do for you.
