from uuid import uuid4
from typing import Dict, Any, List
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from ..models import FounderForgeAction, FounderForgeObservation
from ..business import calculate_burn_rate, calculate_traction, attempt_funding_round

class FounderForgeEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        self.tools_list = [
            {
                "name": "hire_personnel",
                "description": "Hire an engineer or sales person. Engineers cost 12k/mo, Sales 8k/mo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "description": "'engineer' or 'sales'"}
                    },
                    "required": ["role"]
                }
            },
            {
                "name": "launch_marketing_campaign",
                "description": "Spend cash on marketing to gain user traction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spend_amount": {"type": "number", "description": "Cash to spend."}
                    },
                    "required": ["spend_amount"]
                }
            },
            {
                "name": "attempt_fundraise",
                "description": "Attempt to raise VC money based on traction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "round_name": {"type": "string", "description": "e.g., 'Series A'"}
                    },
                    "required": ["round_name"]
                }
            }
        ]
        
        self.reset()
        
    def reset(self, **kwargs) -> FounderForgeObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        self._task_name = kwargs.get("task_name", "bootstrap_survival")
        
        if self._task_name == "bootstrap_survival":
            self._cash = 250000.0
            self._max_steps = 12
            self._target_users = 5000
        elif self._task_name == "growth_stage":
            self._cash = 1000000.0
            self._max_steps = 24
            self._target_users = 50000
        else: # unicorn_ipo
            self._cash = 5000000.0
            self._max_steps = 36
            self._target_users = 1000000

        self._users = 0.0
        self._product_quality = 1.0
        self._team = {"engineers": 0, "sales": 0}
        self._current_round = "Pre-Seed"
        self._done = False
        self._last_result = f"Welcome to FounderForge. Task: {self._task_name}"
        self._last_tool_result = None
        return self._make_obs(reward=0.0)

    def step(self, action: FounderForgeAction) -> FounderForgeObservation:
        self._state.step_count += 1
        
        if self._done:
            return self._make_obs(0.0)
            
        if self._state.step_count > self._max_steps:
            self._done = True
            self._last_result = f"Time up. Reached max steps: {self._max_steps}."
            return self._make_obs(reward=self._calculate_reward())

        action_type = action.action_type
        marketing_spend = 0.0
        self._last_tool_result = None
        
        if action_type == "ToolCallAction":
            tool_name = action.tool_name
            args = action.arguments or {}
            
            if tool_name == "hire_personnel":
                role = str(args.get("role", "")).lower()
                if "engineer" in role or "sales" in role:
                    key = "engineers" if "engineer" in role else "sales"
                    self._team[key] = self._team.get(key, 0) + 1
                    self._last_tool_result = f"Hired 1 {key}."
                    if key == "engineers":
                        self._product_quality += 0.5
                else:
                    self._last_tool_result = f"Invalid role: {role}"
                    
            elif tool_name == "launch_marketing_campaign":
                marketing_spend = float(args.get("spend_amount", 0.0))
                if marketing_spend > self._cash:
                    marketing_spend = self._cash
                traction = calculate_traction(marketing_spend, self._product_quality)
                self._users += traction
                self._last_tool_result = f"Spent {marketing_spend} on marketing. Gained {traction} users."
                
            elif tool_name == "attempt_fundraise":
                target = str(args.get("round_name", ""))
                success, raised = attempt_funding_round(target, self._users, 0.0)
                if success:
                    self._cash += raised
                    self._current_round = target
                    self._last_tool_result = f"Successfully raised {target}! Got {raised} cash."
                else:
                    self._last_tool_result = f"Failed to raise {target}. Not enough traction."
            else:
                self._last_tool_result = f"Tool {tool_name} not found."
            
            self._last_result = f"Tool Execution: {tool_name}"
            
        elif action_type == "skip":
            self._last_result = "Month passed with no direct action."
            
        elif action_type == "finish":
            self._done = True
            self._last_result = "Game ended by CEO."

        # Apply monthly burn
        burn = calculate_burn_rate(self._team, marketing_spend)
        self._cash -= burn
        
        # Check bankruptcy
        if self._cash <= 0:
            self._done = True
            self._last_result = f"Bankrupt! Burned {burn} this month with no cash left."
            
        return self._make_obs(reward=self._calculate_reward())

    def _calculate_reward(self):
        if self._cash <= 0:
            return 0.0
        score = self._users / self._target_users
        return min(max(score, 0.0), 1.0)

    def _make_obs(self, reward: float) -> FounderForgeObservation:
        return FounderForgeObservation(
            done=self._done,
            reward=reward,
            cash=self._cash,
            users=self._users,
            product_quality=self._product_quality,
            team=self._team.copy(),
            current_round=self._current_round,
            last_action_result=self._last_result,
            task_name=self._task_name,
            task_description=f"Reach {self._target_users} users in {self._max_steps} months.",
            tools_list=self.tools_list,
            tool_result=self._last_tool_result,
            metadata={"step": self._state.step_count}
        )

    @property
    def state(self) -> State:
        return self._state
