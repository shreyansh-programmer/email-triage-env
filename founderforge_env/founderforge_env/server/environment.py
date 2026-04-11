"""FounderForge CEO Simulator — Core Environment.

This module implements the stateful simulation loop for the FounderForge
startup management benchmark. It supports three difficulty tiers, dynamic
market events, dense reward shaping, and five agent tools.

Design Principles:
    - Deterministic grading at fixed seed (env uses seeded RNG per episode).
    - Dense, continuous reward signal — not sparse binary.
    - Semantic market events that require reading comprehension.
    - Clean reset() → fresh state every time.
"""

import random
from uuid import uuid4
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..models import FounderForgeAction, FounderForgeObservation
from ..business import calculate_burn_rate, calculate_traction, attempt_funding_round


# ── Market Event Corpus ─────────────────────────────────────────────────────
# Each event is a dictionary with:
#   type          — event category identifier
#   msg           — unstructured prose the agent must read and interpret
#   solution_focus — the correct pivot_strategy focus to resolve the event
#   solution_role  — the correct hire/layoff role to address the event

MODERATE_EVENTS: List[Dict[str, str]] = [
    {
        "type": "ai_threat",
        "msg": (
            "MARKET REPORT: A major competitor just launched a viral AI feature "
            "that is pulling users away from your platform. If you don't pivot to "
            "'product_led' and hire engineers quickly, user churn will accelerate."
        ),
        "solution_focus": "product_led",
        "solution_role": "engineer",
    },
    {
        "type": "ad_cost_spike",
        "msg": (
            "MARKET REPORT: Apple's latest iOS update removed third-party ad "
            "tracking. Customer acquisition cost has tripled overnight. Pivot to "
            "'sales_led' and hire sales reps so you can close B2B deals directly."
        ),
        "solution_focus": "sales_led",
        "solution_role": "sales",
    },
    {
        "type": "talent_war",
        "msg": (
            "MARKET REPORT: A wave of big-tech layoffs flooded the talent market. "
            "Senior engineers are available at a discount. This is the perfect time "
            "to hire engineers and boost product quality before the window closes."
        ),
        "solution_focus": "product_led",
        "solution_role": "engineer",
    },
    {
        "type": "viral_moment",
        "msg": (
            "MARKET REPORT: An influencer posted about a product similar to yours "
            "and the category is trending. Launch a marketing campaign now to ride "
            "the wave while attention is high."
        ),
        "solution_focus": "sales_led",
        "solution_role": "sales",
    },
]

EXTREME_EVENTS: List[Dict[str, str]] = [
    {
        "type": "bank_run",
        "msg": (
            "CRITICAL MARKET REPORT: Silicon Valley Bank is collapsing and VCs "
            "are freezing all capital deployment. Pivot to 'survival_mode' and "
            "layoff staff immediately to slash burn rate, or face near-certain "
            "bankruptcy within two months."
        ),
        "solution_focus": "survival_mode",
        "solution_role": "layoff",
    },
    {
        "type": "recession",
        "msg": (
            "CRITICAL MARKET REPORT: The Federal Reserve hiked interest rates to "
            "8%. Consumer spending is contracting. Your burn rate is unsustainable. "
            "Switch to 'survival_mode' and reduce headcount immediately."
        ),
        "solution_focus": "survival_mode",
        "solution_role": "layoff",
    },
]

CALM_EVENT: Dict[str, str] = {
    "type": "calm",
    "msg": "The market is stable this month. Focus on steady execution.",
}


class FounderForgeEnvironment(Environment):
    """Stateful startup simulation environment.

    Implements the OpenEnv interface: reset() → Observation, step(Action) → Observation.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ── Tool Definitions (OpenAI function-calling schema) ────────────────
    TOOLS: List[Dict[str, Any]] = [
        {
            "name": "hire_personnel",
            "description": (
                "Hire a team member. Engineers (+$12k/mo burn, +0.5 product quality) "
                "or sales reps (+$8k/mo burn, improve direct-sales channel)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["engineer", "sales"],
                        "description": "The role to hire: 'engineer' or 'sales'.",
                    }
                },
                "required": ["role"],
            },
        },
        {
            "name": "layoff_staff",
            "description": (
                "Fire a team member to cut monthly burn rate. "
                "Reduces product quality by 0.2 as a morale penalty."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["engineer", "sales"],
                        "description": "The role to layoff: 'engineer' or 'sales'.",
                    }
                },
                "required": ["role"],
            },
        },
        {
            "name": "pivot_strategy",
            "description": (
                "Change the company's strategic focus. "
                "'product_led' boosts engineering impact. "
                "'sales_led' boosts marketing ROI by 1.5x. "
                "'survival_mode' halves burn rate but bleeds 5% users/month."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "enum": ["product_led", "sales_led", "survival_mode"],
                        "description": "Strategic focus to adopt.",
                    }
                },
                "required": ["focus"],
            },
        },
        {
            "name": "launch_marketing_campaign",
            "description": (
                "Spend cash on user acquisition marketing. "
                "Traction = sqrt(spend) * product_quality * 10. "
                "ROI is affected by current strategy and market conditions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "spend_amount": {
                        "type": "number",
                        "description": "Cash to allocate to the campaign.",
                    }
                },
                "required": ["spend_amount"],
            },
        },
        {
            "name": "attempt_fundraise",
            "description": (
                "Pitch VCs for a funding round. Rounds available: "
                "Pre-Seed (500 users, $250k), Seed (5k users, $1M), "
                "Series A (50k users, $5M), Series B (200k users, $20M), "
                "IPO (1M users, $100M). Fails if user threshold not met."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "round_name": {
                        "type": "string",
                        "enum": ["Pre-Seed", "Seed", "Series A", "Series B", "IPO"],
                        "description": "The funding round to attempt.",
                    }
                },
                "required": ["round_name"],
            },
        },
    ]

    # ── Task Configuration ───────────────────────────────────────────────
    TASK_CONFIG = {
        "bootstrap_survival": {
            "cash": 250_000.0,
            "max_steps": 12,
            "target_users": 5_000,
            "event_severity": "none",
            "description": "Survive 12 months on $250k seed money and reach 5,000 users.",
        },
        "growth_stage": {
            "cash": 1_000_000.0,
            "max_steps": 24,
            "target_users": 50_000,
            "event_severity": "moderate",
            "description": "Scale to 50,000 users in 24 months while navigating market shifts.",
        },
        "unicorn_ipo": {
            "cash": 5_000_000.0,
            "max_steps": 36,
            "target_users": 1_000_000,
            "event_severity": "extreme",
            "description": "Reach 1M users and IPO in 36 months surviving black-swan crises.",
        },
    }

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()  # per-episode RNG for deterministic replay
        self._init_defaults()

    def _init_defaults(self) -> None:
        """Set all mutable state to safe defaults."""
        self._cash = 0.0
        self._users = 0.0
        self._product_quality = 1.0
        self._team: Dict[str, int] = {"engineers": 0, "sales": 0}
        self._current_round = "Pre-Seed"
        self._strategy = "product_led"
        self._task_name = "bootstrap_survival"
        self._max_steps = 12
        self._target_users = 5_000
        self._event_severity = "none"
        self._active_event: Optional[Dict[str, str]] = None
        self._dense_reward_buffer = 0.0
        self._done = False
        self._last_result = ""
        self._last_tool_result: Optional[str] = None

    # ── OpenEnv Interface ────────────────────────────────────────────────

    def reset(self, **kwargs: Any) -> FounderForgeObservation:
        """Reset the environment to a clean initial state for a given task."""
        seed = kwargs.get("seed", None)
        self._rng = random.Random(seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name = kwargs.get("task_name", "bootstrap_survival")

        cfg = self.TASK_CONFIG.get(self._task_name, self.TASK_CONFIG["bootstrap_survival"])
        self._cash = cfg["cash"]
        self._max_steps = cfg["max_steps"]
        self._target_users = cfg["target_users"]
        self._event_severity = cfg["event_severity"]

        self._users = 0.0
        self._product_quality = 1.0
        self._team = {"engineers": 0, "sales": 0}
        self._current_round = "Pre-Seed"
        self._strategy = "product_led"
        self._active_event = None
        self._dense_reward_buffer = 0.0
        self._done = False
        self._last_tool_result = None
        self._last_result = (
            f"Welcome to FounderForge — Task: {self._task_name}. "
            f"You have ${self._cash:,.0f} and {self._max_steps} months. "
            f"Target: {self._target_users:,} users. Read Market Reports carefully!"
        )

        return self._make_obs(reward=0.0)

    def step(self, action: FounderForgeAction) -> FounderForgeObservation:
        """Execute one month of simulation."""
        self._state.step_count += 1

        if self._done:
            return self._make_obs(reward=self._calculate_reward())

        if self._state.step_count > self._max_steps:
            self._done = True
            self._last_result = f"Time's up — reached month {self._max_steps}."
            return self._make_obs(reward=self._calculate_reward())

        marketing_spend = 0.0
        self._last_tool_result = None
        step_bonus = 0.0
        step_penalty = 0.0

        # ── Phase 1: Evaluate semantic alignment with active event ────
        if self._active_event and self._active_event["type"] != "calm":
            step_bonus, step_penalty = self._evaluate_event_response(action)

        # ── Phase 2: Execute Action ───────────────────────────────────
        if action.action_type == "ToolCallAction":
            marketing_spend = self._execute_tool(action)
        elif action.action_type == "skip":
            self._last_result = "Month passed with no action taken."
        elif action.action_type == "finish":
            self._done = True
            self._last_result = "CEO ended the simulation."

        # ── Phase 3: Monthly Burn ─────────────────────────────────────
        burn = calculate_burn_rate(self._team, marketing_spend)
        if self._strategy == "survival_mode":
            burn *= 0.5
            self._users = max(0.0, self._users * 0.95)  # bleed 5% users
        self._cash -= burn

        # ── Phase 4: Bankruptcy Check ─────────────────────────────────
        if self._cash <= 0:
            self._cash = 0.0
            self._done = True
            step_penalty -= 0.3
            self._last_result = f"BANKRUPT — burned ${burn:,.0f} with no cash remaining."

        # ── Phase 5: Roll next market event ───────────────────────────
        if self._active_event is None or self._state.step_count % 3 == 0:
            self._active_event = self._generate_market_event()

        self._dense_reward_buffer += step_bonus + step_penalty
        return self._make_obs(reward=self._calculate_reward())

    # ── Private Helpers ──────────────────────────────────────────────────

    def _evaluate_event_response(self, action: FounderForgeAction):
        """Score how well the agent's action addresses the active market event."""
        bonus = 0.0
        penalty = 0.0
        evt = self._active_event

        if action.action_type != "ToolCallAction":
            penalty -= 0.08  # ignoring a crisis entirely
            return bonus, penalty

        t_name = action.tool_name or ""
        args = action.arguments or {}

        # Correct pivot resolves the event
        if t_name == "pivot_strategy" and args.get("focus") == evt.get("solution_focus"):
            bonus += 0.12
            self._active_event = CALM_EVENT  # resolved
        # Correct hire addresses the event
        elif t_name == "hire_personnel" and args.get("role") == evt.get("solution_role"):
            bonus += 0.08
        # Correct layoff during a crisis
        elif t_name == "layoff_staff" and evt.get("solution_role") == "layoff":
            bonus += 0.12
        else:
            penalty -= 0.04  # wrong action during a crisis

        return bonus, penalty

    def _execute_tool(self, action: FounderForgeAction) -> float:
        """Execute a tool call and return the marketing_spend consumed."""
        tool_name = action.tool_name or ""
        args = action.arguments or {}
        marketing_spend = 0.0

        if tool_name == "hire_personnel":
            role = str(args.get("role", "")).lower()
            if role in ("engineer", "sales"):
                key = "engineers" if role == "engineer" else "sales"
                self._team[key] += 1
                self._last_tool_result = f"Hired 1 {key}. Team now: {dict(self._team)}."
                if key == "engineers":
                    self._product_quality += 0.5
            else:
                self._last_tool_result = f"Invalid role '{role}'. Use 'engineer' or 'sales'."

        elif tool_name == "layoff_staff":
            role = str(args.get("role", "")).lower()
            key = "engineers" if role == "engineer" else "sales"
            if self._team.get(key, 0) > 0:
                self._team[key] -= 1
                self._product_quality = max(0.5, self._product_quality - 0.2)
                self._last_tool_result = f"Laid off 1 {key}. Team now: {dict(self._team)}."
            else:
                self._last_tool_result = f"No {key} to lay off."

        elif tool_name == "pivot_strategy":
            focus = str(args.get("focus", "")).lower()
            if focus in ("product_led", "sales_led", "survival_mode"):
                self._strategy = focus
                self._last_tool_result = f"Strategy pivoted to '{focus}'."
            else:
                self._last_tool_result = f"Invalid focus '{focus}'."

        elif tool_name == "launch_marketing_campaign":
            try:
                raw_spend = float(args.get("spend_amount", 0.0))
            except (ValueError, TypeError):
                raw_spend = 0.0
                self._last_tool_result = f"Invalid spend_amount received. Defaulting to 0."
                
            marketing_spend = min(max(raw_spend, 0.0), self._cash)  # can't spend more than you have, or negative

            # Strategy and event modifiers
            modifier = 1.0
            if self._strategy == "sales_led":
                modifier = 1.5
            if self._active_event and self._active_event.get("type") == "ad_cost_spike":
                modifier *= 0.3

            traction = calculate_traction(marketing_spend * modifier, self._product_quality)
            self._users += traction
            self._last_tool_result = (
                f"Spent ${marketing_spend:,.0f} on marketing (modifier={modifier:.1f}x). "
                f"Acquired {traction:,.0f} new users."
            )

        elif tool_name == "attempt_fundraise":
            round_name = str(args.get("round_name", ""))
            success, raised = attempt_funding_round(round_name, self._users, 0.0)
            if success:
                self._cash += raised
                self._current_round = round_name
                self._last_tool_result = f"Raised {round_name}! +${raised:,.0f} cash."
            else:
                self._last_tool_result = f"Failed {round_name} — insufficient traction."

        else:
            self._last_tool_result = f"Unknown tool: '{tool_name}'."

        self._last_result = f"Executed tool: {tool_name}"
        return marketing_spend

    def _generate_market_event(self) -> Dict[str, str]:
        """Roll a market event based on the current difficulty severity."""
        if self._event_severity == "none":
            return CALM_EVENT

        pool: List[Dict[str, str]] = []
        if self._event_severity in ("moderate", "extreme"):
            pool.extend(MODERATE_EVENTS)
        if self._event_severity == "extreme":
            pool.extend(EXTREME_EVENTS)

        # 40% chance of an event firing each cycle
        if self._rng.random() > 0.6:
            return self._rng.choice(pool)
        return CALM_EVENT

    def _calculate_reward(self) -> float:
        """Compute a bounded [0.01, 0.99] reward from task progress + dense shaping."""
        task_progress = self._users / max(self._target_users, 1)
        shaped = task_progress + self._dense_reward_buffer

        if self._done and self._cash <= 0:
            shaped *= 0.5  # bankruptcy multiplier

        return min(max(shaped, 0.01), 0.99)

    def _make_obs(self, reward: float) -> FounderForgeObservation:
        """Construct the observation payload for the agent."""
        event_msg = (self._active_event or CALM_EVENT).get("msg", "All is well.")
        market_report = f"[MARKET UPDATE] {event_msg}"

        return FounderForgeObservation(
            done=self._done,
            reward=reward,
            cash=self._cash,
            users=self._users,
            product_quality=self._product_quality,
            team=self._team.copy(),
            current_round=self._current_round,
            strategy=self._strategy,
            last_action_result=f"{market_report}\n{self._last_result}",
            task_name=self._task_name,
            task_description=self.TASK_CONFIG[self._task_name]["description"],
            tools_list=self.TOOLS,
            tool_result=self._last_tool_result,
            metadata={
                "step": self._state.step_count,
                "strategy": self._strategy,
                "episode_id": self._state.episode_id,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self) -> None:
        """Clean up any resources (no-op for this in-process environment)."""
        pass
