"""FounderForge Business Logic — Core Mathematical Models for Startup Growth Simulation.

This module contains deterministic, well-documented functions that drive the
financial engine of the FounderForge environment. Each function models a
real-world startup dynamic: operational burn, user acquisition, and venture
capital fundraising.
"""

from typing import Tuple


def calculate_burn_rate(team: dict, marketing_spend: float) -> float:
    """Calculate monthly cash burn including base operations, team salaries,
    and any marketing expenditure.

    Args:
        team: Dictionary with keys 'engineers' and 'sales' mapping to headcounts.
        marketing_spend: One-time marketing expenditure this month.

    Returns:
        Total cash consumed this month.

    Formula:
        burn = base_ops ($10k) + engineers * $12k + sales * $8k + marketing_spend
    """
    base_ops = 10_000.0
    engineer_salary = 12_000.0
    sales_salary = 8_000.0

    salaries = (team.get("engineers", 0) * engineer_salary
                + team.get("sales", 0) * sales_salary)

    return base_ops + salaries + marketing_spend


def calculate_traction(marketing_spend: float, product_quality: float) -> float:
    """Calculate new user acquisition from a marketing campaign.

    Uses a square-root model so that ROI diminishes at higher spend levels,
    rewarding agents that spread spend across rounds rather than dumping it
    all at once.

    Args:
        marketing_spend: Cash allocated to marketing (after any modifiers).
        product_quality: Multiplicative quality factor (starts at 1.0).

    Returns:
        Number of new users acquired this month.

    Formula:
        traction = sqrt(marketing_spend) * product_quality * 10
    """
    if marketing_spend <= 0:
        return 0.0
    return (marketing_spend ** 0.5) * product_quality * 10.0


def attempt_funding_round(round_name: str, users: float, revenue: float) -> Tuple[bool, float]:
    """Attempt to close a venture capital funding round.

    Each round has a hard user-count gate. If the startup meets the
    threshold the round succeeds and capital is injected.

    Args:
        round_name: One of 'Pre-Seed', 'Seed', 'Series A', 'Series B', 'IPO'.
        users: Current total user base.
        revenue: Current monthly revenue (reserved for future use).

    Returns:
        Tuple of (success: bool, cash_raised: float).
    """
    funding_table = {
        "Pre-Seed": {"users": 500,     "raise": 250_000.0},
        "Seed":     {"users": 5_000,   "raise": 1_000_000.0},
        "Series A": {"users": 50_000,  "raise": 5_000_000.0},
        "Series B": {"users": 200_000, "raise": 20_000_000.0},
        "IPO":      {"users": 1_000_000, "raise": 100_000_000.0},
    }

    reqs = funding_table.get(round_name)
    if not reqs:
        return False, 0.0

    if users >= reqs["users"]:
        return True, reqs["raise"]

    return False, 0.0
