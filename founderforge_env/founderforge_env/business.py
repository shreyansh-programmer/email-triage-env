"""FounderForge Business Logic for Evaluation"""

def calculate_burn_rate(team: dict, marketing_spend: float) -> float:
    """Calculate monthly burn rate including base ops, team salaries, and marketing."""
    base_ops = 10000.0  # $10k base
    
    # Salaries: Engineers cost $12k, Sales cost $8k
    salaries = team.get("engineers", 0) * 12000.0 + team.get("sales", 0) * 8000.0
    
    return base_ops + salaries + marketing_spend

def calculate_traction(marketing_spend: float, product_quality: float) -> float:
    """Calculate user traction based on marketing and product quality."""
    if marketing_spend <= 0:
        return 0.0
    return (marketing_spend ** 0.5) * product_quality * 10.0

def attempt_funding_round(round_name: str, users: float, revenue: float) -> tuple[bool, float]:
    """Attempt a funding round based on current metrics."""
    metrics_required = {
        "Seed": {"users": 5000, "raise": 1000000.0},
        "Series A": {"users": 50000, "raise": 5000000.0},
        "Series B": {"users": 200000, "raise": 20000000.0},
        "IPO": {"users": 1000000, "raise": 100000000.0}
    }
    
    reqs = metrics_required.get(round_name)
    if not reqs:
        return False, 0.0
        
    if users >= reqs["users"]:
        return True, reqs["raise"]
    
    return False, 0.0
