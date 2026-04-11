"""FounderForge — Pydantic Models for Actions and Observations.

These typed models are the contract between the agent and the environment.
They satisfy the OpenEnv spec requirement for typed Action and Observation
interfaces with full documentation.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.interfaces import Action, Observation


class FounderForgeAction(Action):
    """An action the agent sends to the environment each step.

    Attributes:
        action_type: One of 'ToolCallAction', 'skip', or 'finish'.
        tool_name: The tool to invoke (required when action_type is 'ToolCallAction').
        arguments: Keyword arguments for the tool call.
    """
    action_type: str
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None


class FounderForgeObservation(Observation):
    """The observation returned by the environment after each step.

    Attributes:
        cash: Current cash balance.
        users: Total accumulated user base.
        product_quality: Current product quality multiplier.
        team: Headcount breakdown {'engineers': int, 'sales': int}.
        current_round: Last successfully closed funding round.
        strategy: Current strategic focus of the company.
        last_action_result: Human-readable summary of what happened.
        task_name: Active task identifier.
        task_description: Human-readable task objective.
        tools_list: Available tool definitions in OpenAI function-calling format.
        tool_result: Result string from the last tool execution, if any.
    """
    cash: float
    users: float
    product_quality: float
    team: Dict[str, int]
    current_round: str
    strategy: str
    last_action_result: str
    company_name: str
    task_name: str
    task_description: str
    tools_list: List[Dict[str, Any]]
    tool_result: Optional[str] = None
