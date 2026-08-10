# Pre-registered design — autumn 2026 randomised tier-pricing experiment

**Frozen:** 2026-08-10, before the first experimental on-sale (w/e 2026-08-15/16).
**Status:** FROZEN. No edits after this commit except the Amendments section, which may only
add dated entries — never rewrite history.
**Purpose:** recover a causally identified price elasticity of demand for club-night tickets by
randomising release-tier price ladders at the event level, on the operator's own programme.
This executes §9 of `../SPEC.md`. Identification rationale: §4 of `../SPEC.md` (operator-set
prices are endogenous; only variation the operator's demand expectations did not cause can
identify elasticity).

Entities are pseudonymised per repo policy (`../DECISIONS.md`): brands → A/B, venues → V1–V4.
The real-name key is local-only (`key.local.md`, gitignored). Ticket prices are public
information once on sale and appear in clear.

## 1. Experimental pool

Resident club nights, autumn 2026 programme, both brands, two cities. Four blocks:

| Block | Brand | Venue | Expected n | Baseline ladder (adv. releases + door) |
|---|---|---|---|---|
| B1 | A | V1 (home city) | 4–5 | £6 / £8 / £10 · door £11 (pre-committed) |
| B2 | B | V1 (home city) | 5 | £5 / £7 / £9 · door £10 (pre-committed) |
| B3 | A | V2 (second city) | 4 | varies — recorded per event before reveal |
| B4 | A | V3 (second city) | 2 | varies — recorded per event before reveal |

**Excluded:** the 2–3 headline shows carrying artist guarantees (V4 + possibly one second-city
show). They are priced normally and are not experimental units. Exclusion decided before any
assignment, on risk grounds (guarantee money must not ride on a coin) and power grounds
(2–3 heterogeneous units add ~nothing).

**Unit eligibility:** the operator (or a committed branch manager) controls the ladder; sells
on-platform; prices not yet committed anywhere at assignment time. Events already priced or
announced before this freeze are ineligible.

## 2. Treatment

Two arms, **HIGH = ×1.15** and **LOW = ×0.85**, applied to every rung of the event's baseline
ladder **including the door price**.

- **Baseline** = the ladder that would have been set absent the experiment. B1/B2 baselines are
  pre-committed above. B3/B4 baselines are written down per event **before** the assignment is
  revealed (see §4 SOP) — this ordering is load-bearing: a baseline chosen after reveal lets
  the operator compensate and destroys the design.
- **Rounding:** to the nearest £0.50; exact ties round **away from the baseline rung** (so the
  tie amplifies rather than erodes the dose). Implemented in `assign.py` in exact integer
  arithmetic; the realised (post-rounding) ladder is what buyers see and what analysis uses.
- Realised dose varies slightly by rung after rounding; analysis uses realised log-price, not
  the nominal ±15%.

Worked ladders for the pre-committed blocks:

| Block | Arm | Ladder |
|---|---|---|
| B1 | HIGH | £7.00 / £9.00 / £11.50 · door £12.50 |
| B1 | LOW | £5.00 / £7.00 / £8.50 · door £9.50 |
| B2 | HIGH | £6.00 / £8.00 / £10.50 · door £11.50 |
| B2 | LOW | £4.00 / £6.00 / £7.50 · door £8.50 |

## 3. Assignment mechanism

**Permuted blocks of size 2, in announcement order, within each design block.**

- Within a block, events are indexed 1, 2, 3, … in the order they are confirmed for
  announcement (the programme announces rolling, 10–20 days before each event, so a full
  date-ordered calendar does not exist at freeze; announcement order is the deterministic
  substitute and is recorded in `assignments.md` as it happens).
- Events (1,2) form pair 1, (3,4) pair 2, etc. The first event of each pair gets the arm from
  a deterministic coin; its pair-mate gets the opposite arm. An odd final event keeps its own
  coin. This guarantees within-block balance to ±1.
- **The coin is cryptographic and pre-committed:** arm = parity of the first byte of
  SHA-256(`"ticket-pricing-autumn-2026-v1:<block>:<pair>"`). No randomness at run time; anyone
  can recompute every assignment from this document alone. The seed string is frozen in this
  commit and in `assign.py`.
- Assignments for all pairs are therefore already determined by this freeze; they are merely
  *revealed* per event by running `assign.py` at announcement time, after the baseline is
  recorded.

## 4. Standard operating procedure (per event)

1. Event confirmed for announcement → give it the next index in its block.
2. **Record the baseline ladder first** — for B3/B4, append it to `assignments.local.md` and
   commit (the commit timestamp is the proof of ordering). B1/B2 baselines are pre-committed.
3. Run: `python3 experiment/assign.py --block B2 --index 3 --baseline 5,7,9,10`
4. Build the poster/listing with the realised ladder it prints. Announce as normal.
5. Log the row in `assignments.md` (pseudonymised) and `assignments.local.md` (real names).

## 5. Conduct rules

1. **No mid-flight price changes**, by anyone, for any reason. A violated event is flagged in
   `assignments.md` and excluded-with-reason from the primary analysis. (The business may
   always choose to violate — the experiment just loses that unit, honestly.)
2. **Marketing rescues are allowed** — this is a live business — but every discretionary promo
   push is logged per event (date, channel, approx. spend) and reported; asymmetric rescue of
   LOW-pacing nights is a real confound and will be examined, not hidden.
3. **Sellouts close sales normally.** A sold-out event is recorded as censored at capacity.
4. Door staff charge the assigned door price.
5. The second-city branch manager is briefed that a pricing trial is running and commits to
   running assigned ladders exactly; the no-changes rule binds them too.
6. **No peeking, no early stopping:** the experiment runs the full autumn programme; no interim
   elasticity estimates inform any pricing decision inside the pool.

## 6. Outcomes and analysis plan (pre-registered)

- **Primary outcome:** log total presale tickets per event (platform export; comps/guestlist
  stripped per `../SPEC.md` §8.7).
- **Secondary outcomes:** total attendance (presale + door reconciliation), gross ticket
  revenue, revenue per unit capacity.
- **Primary estimand:** demand elasticity β from `log(Q_e) = α_block + β·Δlog(P_e) + ε_e`,
  where Δlog(P_e) is the event's realised mean log-price deviation from its baseline ladder
  (quantity-weighting specified at analysis as tier-level robustness, not primary).
- **Inference:** randomisation inference — permutation of arm labels within blocks (all
  within-block permutations; exact p). Asymptotic SEs are reported but not trusted at n≈15.
- **Robustness (all pre-specified):** (i) excluding censored/sold-out events; (ii) excluding
  violated events (by definition); (iii) tier-level regression with event FE absorbed —
  noting that within-event tier variation remains observational (the experiment randomises
  the ladder level, not relative rung spacing); (iv) marketing-push covariate check.
- **Reporting:** effect size with uncertainty, in plain units (e.g. "a 15% price rise changed
  presales by X% [CI]"). "Insufficient signal" is an acceptable, publishable result and will
  be stated as such — no smoothing, no sign-hunting (`../SPEC.md` §4.1).

## 7. Risk position (the downside-cap decision required by ../DECISIONS.md)

Symmetric paired arms mean portfolio exposure largely cancels; per-event worst case ≈ ±£300 on
a ~£2k night if demand is fully inelastic (HIGH gains offset) or fully elastic (LOW gains
offset). **Accepted explicitly at ±15% with no mid-flight abort rule** — an abort rule requires
peeking, which costs more (the whole experiment) than it can save. Guarantee-backed shows are
outside the pool, so no fixed obligation rides on any assignment.

## 8. Amendments

- 2026-08-10 (at freeze, before any assignment): assignment switched from
  date-order-pairing-at-freeze (July sketch) to permuted blocks of 2 in announcement order,
  because the programme announces rolling and no full calendar exists at freeze. Decided
  before unit #1's assignment was computed; equivalent balancing properties.
