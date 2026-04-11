import pytest
from founderforge_env.business import calculate_burn_rate, calculate_traction, attempt_funding_round

def test_calculate_burn_rate():
    # Base burn should be $10k/month. Engineers cost $12k/month. Marketing is additive.
    base_team = {"engineers": 0, "sales": 0}
    assert calculate_burn_rate(base_team, 0) == 10000.0

    team = {"engineers": 2, "sales": 1} # 24k + 8k = 32k
    assert calculate_burn_rate(team, 5000) == 47000.0 # 10k(base) + 32k + 5k

def test_calculate_traction():
    # Traction formula: (marketing_spend ^ 0.5) * product_quality * 10
    assert calculate_traction(10000, 1.0) == 1000.0
    assert calculate_traction(10000, 2.0) == 2000.0
    assert calculate_traction(0, 1.0) == 0.0

def test_attempt_funding_round():
    # Attempting to raise Seed round (requires 5k users, gives 1M cash)
    success, raised = attempt_funding_round("Seed", users=6000, revenue=0)
    assert success is True
    assert raised == 1000000.0

    success, raised = attempt_funding_round("Seed", users=1000, revenue=0)
    assert success is False
    assert raised == 0.0
