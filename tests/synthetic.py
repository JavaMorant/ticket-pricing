"""Synthetic fixtures. Obviously fake by construction — no real buyer or event data.

Every person-shaped value is prefixed "FAKE". Brands are Brand A / Brand B and venues
are V1..V4, matching the pseudonyms used everywhere that could be committed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def draft_preregistration(tmp_path: Path) -> Path:
    """A `preregistration.md` that is still DRAFT, for exercising the real-data guard.

    The repo's own `preregistration.md` has been FROZEN since commit `1c7196d`, so any test
    that wants to see the DRAFT refusal has to bring its own file. Reading the ambient one
    is what went silently stale at the freeze: ten guard tests were asserting a fact about
    the repo rather than a property of the guard, and they all flipped in one commit.
    """
    path = Path(tmp_path) / "preregistration.md"
    path.write_text(
        "# Pre-registered observational pattern list — fixture\n\n"
        "**Status: DRAFT — not frozen; first data contact has not happened.**\n",
        encoding="utf-8",
    )
    return path


# (event_key, tier_name, unit_price, booking_fee, n_tickets, first_purchase_date)
_TICKET_SPEC = [
    ("E1", "Early Bird", 8.00, 1.00, 4, "2026-01-10"),
    ("E1", "Tier 1", 10.00, 1.20, 3, "2026-01-25"),
    ("E1", "Door", 12.00, 0.00, 2, "2026-03-01"),
    ("E1", "Guestlist", 0.00, 0.00, 1, "2026-02-28"),
    ("E2", "EARLY BIRD", 9.00, 1.00, 3, "2026-02-14"),
    ("E2", "Release 2", 11.00, 1.30, 2, "2026-03-20"),
    ("E2", "Door", 14.00, 0.00, 1, "2026-04-15"),
    ("E2", "Comp", 0.00, 0.00, 1, "2026-04-10"),
]


def fake_transactions() -> pd.DataFrame:
    """Canonical-column ticket rows, WITH fake PII columns attached.

    The PII columns are here on purpose: the tests assert they do not survive ingest.
    """
    rows = []
    n = 0
    for event_key, tier, price, fee, count, first_day in _TICKET_SPEC:
        start = pd.Timestamp(first_day)
        for i in range(count):
            n += 1
            rows.append(
                {
                    "order_id": f"FAKE-ORD-{n:03d}",
                    "event_key": event_key,
                    "tier_name": tier,
                    "price_paid": price,
                    "booking_fee": fee,
                    "purchased_at": start + pd.Timedelta(hours=6 * i),
                    "quantity": 1,
                    "promo_code": "FAKEPROMO" if i == 0 else None,
                    "platform": "fixr",
                    # --- fake buyer PII, must be stripped at ingest ---
                    "buyer_name": f"FAKE Buyer {n:03d}",
                    "buyer_email": f"fake.buyer.{n:03d}@example.invalid",
                    "phone": "+44 0000 000000",
                    "address": "1 FAKE Street",
                    "postcode": "FA1 1KE",
                }
            )
    return pd.DataFrame(rows)


def fake_costs() -> pd.DataFrame:
    """Canonical-column per-event cost rows for two fake events."""
    return pd.DataFrame(
        [
            {
                "event_key": "E1",
                "event_name": "FAKE Event One",
                "event_date": "2026-03-01",
                "venue": "V1",
                "city": "City A",
                "brand": "Brand A",
                "capacity": 300,
                "artist_guarantee": 1500.0,
                "deal_structure": "guarantee",
                "venue_cost": 800.0,
                "security_cost": 300.0,
                "production_cost": 200.0,
                "staffing_cost": 150.0,
                "marketing_spend": 400.0,
                "cancelled": False,
                "finance_contact_email": "fake.finance@example.invalid",
            },
            {
                "event_key": "E2",
                "event_name": "FAKE Event Two",
                "event_date": "2026-04-15",
                "venue": "V2",
                "city": "City B",
                "brand": "Brand B",
                "capacity": 500,
                "artist_guarantee": 2500.0,
                "deal_structure": "guarantee_or_split",
                "venue_cost": 1000.0,
                "security_cost": 400.0,
                "production_cost": 250.0,
                "staffing_cost": 200.0,
                "marketing_spend": 600.0,
                "cancelled": False,
                "finance_contact_email": "fake.finance@example.invalid",
            },
        ]
    )
