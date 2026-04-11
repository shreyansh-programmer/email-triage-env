import pytest
from founderforge_env.server.environment import FounderForgeEnvironment
from founderforge_env.models import FounderForgeAction

def test_environment_initialization():
    env = FounderForgeEnvironment()
    obs = env.reset()
    assert obs.cash == 250000.0
    assert len(obs.tools_list) == 3

def test_environment_hire_action():
    env = FounderForgeEnvironment()
    env.reset()
    
    # ToolCallAction for hiring
    action = FounderForgeAction(
        action_type="ToolCallAction", 
        tool_name="hire_personnel", 
        arguments={"role": "engineer"}
    )
    obs = env.step(action)
    
    assert obs.team["engineers"] == 1
    assert "Hired 1 engineers." == obs.tool_result

def test_bankruptcy_condition():
    env = FounderForgeEnvironment()
    env.reset()
    
    env._cash = 0
    
    action = FounderForgeAction(action_type="skip")
    obs = env.step(action)
    
    assert obs.done is True
    assert "bankrupt" in obs.last_action_result.lower()
