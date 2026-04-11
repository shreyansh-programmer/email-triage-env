"""Tests for FounderForge programmatic graders."""

import pytest
from founderforge_env.evaluation import (
    grade_bootstrap_survival,
    grade_growth_stage,
    grade_unicorn_ipo,
    GRADERS,
)


class TestGraderRegistry:
    def test_all_three_tasks_have_graders(self):
        assert "bootstrap_survival" in GRADERS
        assert "growth_stage" in GRADERS
        assert "unicorn_ipo" in GRADERS

    def test_graders_are_callable(self):
        for name, fn in GRADERS.items():
            assert callable(fn)


class TestBootstrapGrader:
    def test_perfect_run(self):
        obs = {"users": 5_000, "cash": 250_000}
        score = grade_bootstrap_survival(obs)
        assert 0.90 <= score <= 0.99

    def test_bankrupt_low_score(self):
        obs = {"users": 0, "cash": 0}
        score = grade_bootstrap_survival(obs)
        assert score == 0.01

    def test_half_users(self):
        obs = {"users": 2_500, "cash": 100_000}
        score = grade_bootstrap_survival(obs)
        assert 0.3 < score < 0.8

    def test_score_always_bounded(self):
        for users in [0, 100, 5_000, 100_000]:
            for cash in [0, 50_000, 250_000]:
                score = grade_bootstrap_survival({"users": users, "cash": cash})
                assert 0.01 <= score <= 0.99


class TestGrowthGrader:
    def test_perfect_run(self):
        obs = {"users": 50_000, "cash": 1_000_000, "current_round": "Series A", "team": {"engineers": 5, "sales": 5}}
        score = grade_growth_stage(obs)
        assert score > 0.8

    def test_no_progress(self):
        obs = {"users": 0, "cash": 0, "current_round": "Pre-Seed", "team": {"engineers": 0, "sales": 0}}
        score = grade_growth_stage(obs)
        assert score == 0.01


class TestUnicornGrader:
    def test_ipo_achieved(self):
        obs = {"users": 1_000_000, "cash": 5_000_000, "current_round": "IPO", "product_quality": 5.0}
        score = grade_unicorn_ipo(obs)
        assert score > 0.9

    def test_partial_progress(self):
        obs = {"users": 100_000, "cash": 1_000_000, "current_round": "Seed", "product_quality": 2.0}
        score = grade_unicorn_ipo(obs)
        assert 0.1 < score < 0.6

    def test_deterministic(self):
        obs = {"users": 500_000, "cash": 2_000_000, "current_round": "Series A", "product_quality": 3.0}
        s1 = grade_unicorn_ipo(obs)
        s2 = grade_unicorn_ipo(obs)
        assert s1 == s2  # graders must be deterministic
