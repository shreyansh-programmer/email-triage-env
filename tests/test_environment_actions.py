"""Tests for FounderForge environment actions and state transitions."""

import pytest
from founderforge_env.server.environment import FounderForgeEnvironment
from founderforge_env.models import FounderForgeAction


class TestEnvironmentInitialization:
    """Verify reset() produces clean, correct initial state."""

    def test_default_task_is_bootstrap(self):
        env = FounderForgeEnvironment()
        obs = env.reset()
        assert obs.task_name == "bootstrap_survival"
        assert obs.cash == 250_000.0
        assert obs.users == 0.0
        assert obs.product_quality == 1.0
        assert obs.team == {"engineers": 0, "sales": 0}

    def test_tools_list_has_five_tools(self):
        env = FounderForgeEnvironment()
        obs = env.reset()
        assert len(obs.tools_list) == 5
        tool_names = {t["name"] for t in obs.tools_list}
        assert tool_names == {
            "hire_personnel", "layoff_staff", "pivot_strategy",
            "launch_marketing_campaign", "attempt_fundraise",
        }

    def test_growth_stage_config(self):
        env = FounderForgeEnvironment()
        obs = env.reset(task_name="growth_stage")
        assert obs.cash == 1_000_000.0
        assert obs.task_name == "growth_stage"

    def test_unicorn_ipo_config(self):
        env = FounderForgeEnvironment()
        obs = env.reset(task_name="unicorn_ipo")
        assert obs.cash == 5_000_000.0
        assert obs.task_name == "unicorn_ipo"

    def test_reset_clears_previous_state(self):
        env = FounderForgeEnvironment()
        env.reset(task_name="unicorn_ipo")
        obs = env.reset(task_name="bootstrap_survival")
        assert obs.cash == 250_000.0
        assert obs.users == 0.0
        assert obs.done is False

    def test_observation_has_strategy_field(self):
        env = FounderForgeEnvironment()
        obs = env.reset()
        assert obs.strategy == "product_led"

    def test_initial_reward_is_zero(self):
        env = FounderForgeEnvironment()
        obs = env.reset()
        assert obs.reward == 0.0


class TestToolExecution:
    """Verify each tool executes correctly."""

    def test_hire_engineer(self):
        env = FounderForgeEnvironment()
        env.reset()
        action = FounderForgeAction(
            action_type="ToolCallAction",
            tool_name="hire_personnel",
            arguments={"role": "engineer"},
        )
        obs = env.step(action)
        assert obs.team["engineers"] == 1
        assert obs.product_quality == 1.5  # +0.5

    def test_hire_sales(self):
        env = FounderForgeEnvironment()
        env.reset()
        action = FounderForgeAction(
            action_type="ToolCallAction",
            tool_name="hire_personnel",
            arguments={"role": "sales"},
        )
        obs = env.step(action)
        assert obs.team["sales"] == 1

    def test_layoff_reduces_headcount(self):
        env = FounderForgeEnvironment()
        env.reset()
        # Hire first
        env.step(FounderForgeAction(action_type="ToolCallAction", tool_name="hire_personnel", arguments={"role": "engineer"}))
        # Then layoff
        obs = env.step(FounderForgeAction(action_type="ToolCallAction", tool_name="layoff_staff", arguments={"role": "engineer"}))
        assert obs.team["engineers"] == 0

    def test_layoff_empty_team(self):
        env = FounderForgeEnvironment()
        env.reset()
        obs = env.step(FounderForgeAction(action_type="ToolCallAction", tool_name="layoff_staff", arguments={"role": "engineer"}))
        assert obs.team["engineers"] == 0
        assert "No" in obs.tool_result

    def test_pivot_strategy(self):
        env = FounderForgeEnvironment()
        env.reset()
        obs = env.step(FounderForgeAction(action_type="ToolCallAction", tool_name="pivot_strategy", arguments={"focus": "survival_mode"}))
        assert obs.strategy == "survival_mode"

    def test_marketing_gains_users(self):
        env = FounderForgeEnvironment()
        env.reset()
        obs = env.step(FounderForgeAction(action_type="ToolCallAction", tool_name="launch_marketing_campaign", arguments={"spend_amount": 10000}))
        assert obs.users > 0

    def test_fundraise_pre_seed(self):
        env = FounderForgeEnvironment()
        env.reset()
        # First get enough users
        env._users = 600
        obs = env.step(FounderForgeAction(action_type="ToolCallAction", tool_name="attempt_fundraise", arguments={"round_name": "Pre-Seed"}))
        assert "Raised" in obs.tool_result


class TestBankruptcyAndTermination:
    """Verify episode boundaries are sensible."""

    def test_bankruptcy_ends_episode(self):
        env = FounderForgeEnvironment()
        env.reset()
        env._cash = 0
        obs = env.step(FounderForgeAction(action_type="skip"))
        assert obs.done is True
        assert "BANKRUPT" in obs.last_action_result.upper() or "bankrupt" in obs.last_action_result.lower()

    def test_finish_action_ends_episode(self):
        env = FounderForgeEnvironment()
        env.reset()
        obs = env.step(FounderForgeAction(action_type="finish"))
        assert obs.done is True

    def test_max_steps_ends_episode(self):
        env = FounderForgeEnvironment()
        env.reset()
        for _ in range(13):  # bootstrap has 12 max steps
            obs = env.step(FounderForgeAction(action_type="skip"))
        assert obs.done is True

    def test_reward_bounded_after_bankruptcy(self):
        env = FounderForgeEnvironment()
        env.reset()
        env._cash = 0
        obs = env.step(FounderForgeAction(action_type="skip"))
        assert 0.0 < obs.reward < 1.0


class TestRewardShaping:
    """Verify reward signal is dense and meaningful."""

    def test_reward_increases_with_users(self):
        env = FounderForgeEnvironment()
        env.reset()
        obs1 = env.step(FounderForgeAction(action_type="ToolCallAction", tool_name="launch_marketing_campaign", arguments={"spend_amount": 50000}))
        r1 = obs1.reward
        obs2 = env.step(FounderForgeAction(action_type="ToolCallAction", tool_name="launch_marketing_campaign", arguments={"spend_amount": 50000}))
        r2 = obs2.reward
        assert r2 >= r1  # more users → higher reward

    def test_reward_always_in_bounds(self):
        env = FounderForgeEnvironment()
        env.reset()
        for _ in range(12):
            obs = env.step(FounderForgeAction(action_type="ToolCallAction", tool_name="launch_marketing_campaign", arguments={"spend_amount": 20000}))
            assert 0.0 < obs.reward < 1.0
