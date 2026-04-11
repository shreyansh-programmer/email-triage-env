import random
from uuid import uuid4
from typing import Dict, Any, List, Tuple
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from ..models import FounderForgeAction, FounderForgeObservation
from ..business import calculate_burn_rate, calculate_traction, attempt_funding_round

class FounderForgeEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.tools_list = self._get_base_tools()
        self.reset()
        
    def _get_base_tools(self) -> List[Dict]:
        return [
            {
                "name": "hire_personnel",
                "description": "Hire an engineer or sales person. Engineers cost 12k/mo, Sales 8k/mo.",
                "parameters": {
                    "type": "object",
                    "properties": {"role": {"type": "string", "description": "'engineer' or 'sales'"}},
                    "required": ["role"]
                }
            },
            {
                "name": "layoff_staff",
                "description": "Fire an engineer or sales person to instantly cut burn rate. Reduces morale.",
                "parameters": {
                    "type": "object",
                    "properties": {"role": {"type": "string", "description": "'engineer' or 'sales'"}},
                    "required": ["role"]
                }
            },
            {
                "name": "pivot_strategy",
                "description": "Change the company's core focus to adapt to market conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {"focus": {"type": "string", "description": "'product_led', 'sales_led', or 'survival_mode'"}},
                    "required": ["focus"]
                }
            },
            {
                "name": "launch_marketing_campaign",
                "description": "Spend cash on marketing to gain user traction.",
                "parameters": {
                    "type": "object",
                    "properties": {"spend_amount": {"type": "number", "description": "Cash to spend."}},
                    "required": ["spend_amount"]
                }
            },
            {
                "name": "attempt_fundraise",
                "description": "Attempt to raise VC money based on traction.",
                "parameters": {
                    "type": "object",
                    "properties": {"round_name": {"type": "string", "description": "e.g., 'Series A'"}},
                    "required": ["round_name"]
                }
            }
        ]

    def reset(self, **kwargs) -> FounderForgeObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name = kwargs.get("task_name", "bootstrap_survival")
        
        # Difficulty Configuration
        if self._task_name == "bootstrap_survival":
            self._cash = 250000.0
            self._max_steps = 12
            self._target_users = 5000
            self._event_severity = "none"
        elif self._task_name == "growth_stage":
            self._cash = 1000000.0
            self._max_steps = 24
            self._target_users = 50000
            self._event_severity = "moderate"
        else: # unicorn_ipo
            self._cash = 5000000.0
            self._max_steps = 36
            self._target_users = 1000000
            self._event_severity = "extreme"

        self._users = 0.0
        self._product_quality = 1.0
        self._team = {"engineers": 0, "sales": 0}
        self._current_round = "Pre-Seed"
        self._strategy = "product_led"
        self._active_event = None
        self._dense_reward_buffer = 0.0
        self._done = False
        
        self._last_result = f"Welcome to FounderForge. Task: {self._task_name}. Pay close attention to Market Reports!"
        self._last_tool_result = None
        
        return self._make_obs(reward=0.0)

    def _generate_market_event(self) -> Dict:
        """Generates semantic market events for the LLM to read and reason about."""
        if self._event_severity == "none":
            return {"type": "calm", "msg": "The market is calm. Focus on steady growth based on fundamentals."}
            
        events = []
        if self._event_severity in ["moderate", "extreme"]:
            events.append({"type": "ai_threat", "msg": "MARKET REPORT: A major competitor launched a viral AI feature. If you don't pivot to 'product_led' and hire engineers quickly, you'll bleed users.", "solution_focus": "product_led", "solution_role": "engineer"})
            events.append({"type": "ad_cost_spike", "msg": "MARKET REPORT: Apple changed ad-tracking privacy rules. Marketing ROI is terrible right now! You need to pivot to 'sales_led' and hire sales reps.", "solution_focus": "sales_led", "solution_role": "sales"})
            
        if self._event_severity == "extreme":
            events.append({"type": "bank_run", "msg": "CRITICAL MARKET REPORT: Silicon Valley Bank is collapsing! VCs are freezing capital. Pivot to 'survival_mode' and layoff staff immediately to slash burn, or you will likely go bankrupt.", "solution_focus": "survival_mode", "solution_role": "layoff"})

        return random.choice(events) if random.random() > 0.6 else {"type": "calm", "msg": "The market is stable this month."}

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
        step_penalty = 0.0
        step_bonus = 0.0
        
        # 1. Validate if they are solving the active event correctly (Semantic Reasoning)
        if self._active_event and self._active_event["type"] != "calm":
            if action_type == "ToolCallAction":
                t_name = action.tool_name
                args = action.arguments or {}
                
                # Check pivoting
                if t_name == "pivot_strategy" and args.get("focus") == self._active_event.get("solution_focus"):
                    step_bonus += 0.15
                    self._active_event = None # Solved!
                # Check hiring correctly
                elif t_name == "hire_personnel" and args.get("role") == self._active_event.get("solution_role"):
                    step_bonus += 0.10
                # Check layoffs correctly
                elif t_name == "layoff_staff" and self._active_event.get("solution_role") == "layoff":
                    step_bonus += 0.15
                else:
                    step_penalty -= 0.05 # Ignoring critical strategy
            else:
                step_penalty -= 0.10 # Completely ignoring event
        
        # 2. Execute Action
        if action_type == "ToolCallAction":
            tool_name = action.tool_name
            args = action.arguments or {}
            
            if tool_name == "hire_personnel":
                role = str(args.get("role", "")).lower()
                if "engineer" in role or "sales" in role:
                    key = "engineers" if "engineer" in role else "sales"
                    if self._cash < 25000:
                        step_penalty -= 0.1 # Very bad financial decision
                    self._team[key] = self._team.get(key, 0) + 1
                    self._last_tool_result = f"Hired 1 {key}."
                    if key == "engineers":
                        self._product_quality += 0.5
                else:
                    self._last_tool_result = f"Invalid role: {role}"
                    
            elif tool_name == "layoff_staff":
                role = str(args.get("role", "")).lower()
                key = "engineers" if "engineer" in role else "sales"
                if self._team.get(key, 0) > 0:
                    self._team[key] -= 1
                    self._last_tool_result = f"Laid off 1 {key}. Burn rate reduced."
                    self._product_quality = max(0.5, self._product_quality - 0.2)
                else:
                    self._last_tool_result = f"Cannot layoff {key}, none exist."

            elif tool_name == "pivot_strategy":
                focus = str(args.get("focus", "")).lower()
                if focus in ["product_led", "sales_led", "survival_mode"]:
                    self._strategy = focus
                    self._last_tool_result = f"Successfully pivoted to {focus}."
                else:
                    self._last_tool_result = f"Invalid focus constraint: {focus}"

            elif tool_name == "launch_marketing_campaign":
                marketing_spend = float(args.get("spend_amount", 0.0))
                if marketing_spend > self._cash:
                    marketing_spend = self._cash
                    step_penalty -= 0.1 # Bad budgeting logic
                
                # Contextual modifiers
                mod = 1.0
                if self._strategy == "sales_led": mod = 1.5
                if self._active_event and self._active_event["type"] == "ad_cost_spike": mod = 0.3
                
                traction = calculate_traction(marketing_spend * mod, self._product_quality)
                self._users += traction
                self._last_tool_result = f"Spent {marketing_spend} on marketing. Gained {traction} users."
                
            elif tool_name == "attempt_fundraise":
                target = str(args.get("round_name", ""))
                success, raised = attempt_funding_round(target, self._users, 0.0)
                if success:
                    self._cash += raised
                    self._current_round = target
                    self._last_tool_result = f"Successfully raised {target}! Got {raised} cash."
                    step_bonus += 0.2
                else:
                    self._last_tool_result = f"Failed to raise {target}. Not enough traction."
                    step_penalty -= 0.05
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
        if self._strategy == "survival_mode":
            burn *= 0.5 # Survival mode slashes base ops overhead
            self._users = max(0, self._users - (self._users * 0.05)) # But bleeds users
            
        self._cash -= burn
        
        # Check bankruptcy
        if self._cash <= 0:
            self._done = True
            step_penalty -= 0.5
            self._last_result = f"Bankrupt! Burned {burn} this month with no cash left."
            
        # Add a new market event if the old one was solved or timed out
        if not self._active_event or self._state.step_count % 3 == 0:
            self._active_event = self._generate_market_event()
            
        self._dense_reward_buffer += (step_bonus + step_penalty)
            
        return self._make_obs(reward=self._calculate_reward())

    def _calculate_reward(self):
        # Base task progress
        task_score = self._users / self._target_users
        
        # Smooth with dense logic shaping
        final_score = task_score + self._dense_reward_buffer
        
        if self._done and self._cash <= 0:
            # Major penalty for bankruptcy end state but doesn't wipe dense rewards entirely
            final_score *= 0.5 
            
        return min(max(final_score, 0.01), 1.0) # Always bound strictly between 0 and 1

    def _make_obs(self, reward: float) -> FounderForgeObservation:
        event_str = self._active_event.get("msg", "All is well.") if self._active_event else "All is well."
        monthly_report = f"[MONTHLY MARKET UPDATE] {event_str} -> Current Strategy: {self._strategy}"
        
        return FounderForgeObservation(
            done=self._done,
            reward=reward,
            cash=self._cash,
            users=self._users,
            product_quality=self._product_quality,
            team=self._team.copy(),
            current_round=self._current_round,
            last_action_result=monthly_report + "\n" + self._last_result,
            task_name=self._task_name,
            task_description=f"Task: {self._task_name}. Reach {self._target_users} users in {self._max_steps} months without bankruptcy.",
            tools_list=self.tools_list,
            tool_result=self._last_tool_result,
            metadata={"step": self._state.step_count, "strategy": self._strategy}
        )

    @property
    def state(self) -> State:
        return self._state
