"""FounderForge — Programmatic Graders for Each Task.

Each grader takes the final observation and returns a deterministic score
in [0.01, 0.99] based on how well the agent achieved the task objective.

Grader Design Principles (from rubric):
    - Scores are strictly between 0.0 and 1.0 (we use [0.01, 0.99]).
    - Graders are deterministic and reproducible for the same trajectory.
    - Each task has a distinct grading function with clear success/failure criteria.
"""

from typing import Dict, Any


def _clamp(value: float, lo: float = 0.01, hi: float = 0.99) -> float:
    """Clamp a value to the scoring range."""
    return min(max(value, lo), hi)


def grade_bootstrap_survival(obs: Dict[str, Any]) -> float:
    """Grade the bootstrap_survival task (Easy).

    Criteria:
        - 60% weight: Did you reach the 5,000 user target?
        - 20% weight: How much cash do you have remaining (runway efficiency)?
        - 20% weight: Did you avoid bankruptcy?

    Returns:
        Score in [0.01, 0.99].
    """
    target_users = 5_000
    initial_cash = 250_000.0

    user_ratio = min(obs.get("users", 0) / target_users, 1.0)
    cash_ratio = min(obs.get("cash", 0) / initial_cash, 1.0) if obs.get("cash", 0) > 0 else 0.0
    alive_bonus = 0.0 if obs.get("cash", 0) <= 0 else 1.0

    score = (user_ratio * 0.6) + (cash_ratio * 0.2) + (alive_bonus * 0.2)
    return _clamp(score)


def grade_growth_stage(obs: Dict[str, Any]) -> float:
    """Grade the growth_stage task (Medium).

    Criteria:
        - 50% weight: User acquisition toward 50,000 target.
        - 25% weight: Did you successfully raise at least Seed funding?
        - 15% weight: Cash runway remaining.
        - 10% weight: Team size (scaling execution).

    Returns:
        Score in [0.01, 0.99].
    """
    target_users = 50_000
    funding_progression = {"Pre-Seed": 0.0, "Seed": 0.4, "Series A": 0.8, "Series B": 1.0, "IPO": 1.0}

    user_ratio = min(obs.get("users", 0) / target_users, 1.0)
    funding_score = funding_progression.get(obs.get("current_round", "Pre-Seed"), 0.0)
    cash_ratio = min(obs.get("cash", 0) / 1_000_000, 1.0) if obs.get("cash", 0) > 0 else 0.0
    team = obs.get("team", {})
    team_score = min((team.get("engineers", 0) + team.get("sales", 0)) / 10, 1.0)

    score = (user_ratio * 0.5) + (funding_score * 0.25) + (cash_ratio * 0.15) + (team_score * 0.1)
    return _clamp(score)


def grade_unicorn_ipo(obs: Dict[str, Any]) -> float:
    """Grade the unicorn_ipo task (Hard).

    Criteria:
        - 40% weight: User acquisition toward 1,000,000 target.
        - 30% weight: Funding round progression (Series B or IPO required for high scores).
        - 15% weight: Cash remaining after surviving black-swan events.
        - 15% weight: Product quality (sustained engineering investment).

    Returns:
        Score in [0.01, 0.99].
    """
    target_users = 1_000_000
    funding_progression = {"Pre-Seed": 0.0, "Seed": 0.2, "Series A": 0.5, "Series B": 0.8, "IPO": 1.0}

    user_ratio = min(obs.get("users", 0) / target_users, 1.0)
    funding_score = funding_progression.get(obs.get("current_round", "Pre-Seed"), 0.0)
    cash_ratio = min(obs.get("cash", 0) / 5_000_000, 1.0) if obs.get("cash", 0) > 0 else 0.0
    pq = min(obs.get("product_quality", 1.0) / 5.0, 1.0)

    score = (user_ratio * 0.4) + (funding_score * 0.3) + (cash_ratio * 0.15) + (pq * 0.15)
    return _clamp(score)


# Registry for the environment to look up graders by task name
GRADERS = {
    "bootstrap_survival": grade_bootstrap_survival,
    "growth_stage": grade_growth_stage,
    "unicorn_ipo": grade_unicorn_ipo,
}
