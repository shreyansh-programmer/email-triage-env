from typing import Dict, Any, Optional, List
from openenv.core.env_server.interfaces import Action, Observation

class FounderForgeAction(Action):
    action_type: str  # e.g., "ToolCallAction", "skip", "finish"
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None

class FounderForgeObservation(Observation):
    cash: float
    users: float
    product_quality: float
    team: dict
    current_round: str
    last_action_result: str
    task_name: str
    task_description: str
    tools_list: List[Dict[str, Any]]
    tool_result: Optional[Any] = None
