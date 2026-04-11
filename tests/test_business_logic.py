"""Tests for FounderForge business logic functions."""

import pytest
from founderforge_env.business import calculate_burn_rate, calculate_traction, attempt_funding_round


class TestBurnRate:
    def test_base_burn_with_empty_team(self):
        assert calculate_burn_rate({"engineers": 0, "sales": 0}, 0) == 10_000.0

    def test_burn_with_team_and_marketing(self):
        team = {"engineers": 2, "sales": 1}  # 24k + 8k = 32k
        assert calculate_burn_rate(team, 5000) == 47_000.0

    def test_burn_with_only_marketing(self):
        assert calculate_burn_rate({"engineers": 0, "sales": 0}, 100_000) == 110_000.0


class TestTraction:
    def test_zero_spend_gives_zero_users(self):
        assert calculate_traction(0, 1.0) == 0.0

    def test_spend_10k_quality_1(self):
        assert calculate_traction(10_000, 1.0) == 1_000.0

    def test_higher_quality_more_users(self):
        assert calculate_traction(10_000, 2.0) == 2_000.0

    def test_diminishing_returns(self):
        """Doubling spend should NOT double users (sqrt model)."""
        t1 = calculate_traction(10_000, 1.0)
        t2 = calculate_traction(20_000, 1.0)
        assert t2 < t1 * 2


class TestFundingRounds:
    def test_preseed_success(self):
        success, raised = attempt_funding_round("Pre-Seed", 600, 0)
        assert success is True
        assert raised == 250_000.0

    def test_preseed_failure(self):
        success, raised = attempt_funding_round("Pre-Seed", 100, 0)
        assert success is False
        assert raised == 0.0

    def test_seed_success(self):
        success, raised = attempt_funding_round("Seed", 6_000, 0)
        assert success is True
        assert raised == 1_000_000.0

    def test_seed_failure(self):
        success, raised = attempt_funding_round("Seed", 1_000, 0)
        assert success is False

    def test_invalid_round(self):
        success, raised = attempt_funding_round("Series Z", 999_999, 0)
        assert success is False
        assert raised == 0.0

    def test_ipo_requires_million_users(self):
        success, raised = attempt_funding_round("IPO", 1_000_000, 0)
        assert success is True
        assert raised == 100_000_000.0
