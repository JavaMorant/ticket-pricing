# Pre-registered observational pattern list — ticket-pricing

**Status: FROZEN 2026-08-12 — by Awande's instruction "freeze as-is": the §5.4 candidate list
below, unedited, is the pre-registered list. This commit precedes first data contact.**
No further edits except dated entries under Amendments (same rule as `experiment/DESIGN.md`).
Per `SPEC.md` §5.3, this list is written **before first data contact**: only patterns named here
get tested; anything discovered later in the data is a hypothesis for next time, not a result.

## Binding analysis rules (from SPEC §5, restated so the list can't drift)
1. Test **exactly** the patterns below, nothing else.
2. Benjamini–Hochberg FDR correction across the full list; report corrected p-values.
3. Fit on the first four years, confirm on the fifth (held-out events, split by event).
4. Report effect sizes with uncertainty, not bare significance.
5. Control for year/brand-growth trend before believing any calendar effect (SPEC §8.9).
6. Freshers' week ≈ semester start ≈ loan drop ≈ weather arrive together: reported as ONE
   grouped "start of semester" effect, explicitly not decomposed (SPEC §5.6).

## Candidate patterns (EDIT FROM MEMORY — strike what you don't believe, add what you lived)

### Academic calendar
- [ ] Freshers' week (grouped per rule 6)
- [ ] Semester start / mid / end
- [ ] Revision and exam periods (dead weeks)
- [ ] Reading week
- [ ] End-of-term release
- [ ] Vacation vs term time
- [ ] Ball season

### Money
- [ ] Student loan instalment dates (get the real SAAS/SLC dates per year)

### Timing
- [ ] Day of week (Wed/Thu/Fri/Sat not interchangeable)
- [ ] Lead time / days-to-event
- [ ] Days since our last event (cannibalisation, SPEC §8.10)
- [ ] Events in trailing 14 days (both brands pooled)

### Event characteristics
- [ ] Artist billing tier
- [ ] Venue / city / capacity
- [ ] Brand (A vs B)
- [ ] Competing promoters' nights (from memory — nobody else can reconstruct these)

### Environment
- [ ] Weather (outdoor-sensitive events)
- [ ] Academic-year cohort effects

### Awande's additions (from operating memory — fill in)
- [ ]
- [ ]
- [ ]

## Amendments
*(dated entries only, after freeze)*
