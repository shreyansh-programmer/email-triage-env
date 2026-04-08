"""
Evaluation and grading functions for the Email Triage Environment.

Provides deterministic scoring for three task levels:
- priority_classification: accuracy on HIGH/MEDIUM/LOW labels
- route_and_classify: combined priority + department routing accuracy
- full_triage: priority + routing + response type quality

All scores are clamped to [0.01, 0.99] to avoid boundary issues.
"""

from typing import Any, Dict, List, Optional, Tuple


# Valid values
VALID_PRIORITIES = {"HIGH", "MEDIUM", "LOW"}
VALID_DEPARTMENTS = {"Engineering", "Sales", "Legal", "HR", "Support", "Executive"}
VALID_RESPONSE_TYPES = {"acknowledge", "escalate", "delegate", "decline", "info_request"}

# Priority ordering for off-by-one scoring
PRIORITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# Related departments (for partial credit on routing)
RELATED_DEPARTMENTS = {
    "Engineering": {"Support"},
    "Support": {"Engineering"},
    "Sales": {"Executive"},
    "Executive": {"Sales", "Legal"},
    "Legal": {"Executive", "HR"},
    "HR": {"Legal"},
}


def clamp_score(score: float) -> float:
    """Clamp score to [0.01, 0.99] to prevent boundary issues."""
    return max(0.01, min(0.99, score))


def score_priority(predicted: Optional[str], ground_truth: str) -> float:
    """Score a single priority prediction.

    - Exact match: 1.0
    - Off by one level: 0.5
    - Wrong or missing: 0.0
    """
    if not predicted or predicted.upper() not in VALID_PRIORITIES:
        return 0.0

    predicted = predicted.upper()
    if predicted == ground_truth:
        return 1.0

    # Off-by-one partial credit
    diff = abs(PRIORITY_ORDER.get(predicted, -1) - PRIORITY_ORDER.get(ground_truth, -1))
    if diff == 1:
        return 0.5

    return 0.0


def score_routing(predicted: Optional[str], ground_truth: str) -> float:
    """Score a single department routing prediction.

    - Exact match: 1.0
    - Related department: 0.3
    - Wrong or missing: 0.0
    """
    if not predicted:
        return 0.0

    # Normalize: capitalize first letter
    predicted_norm = predicted.strip().title()

    if predicted_norm == ground_truth:
        return 1.0

    # Partial credit for related departments
    related = RELATED_DEPARTMENTS.get(ground_truth, set())
    if predicted_norm in related:
        return 0.3

    return 0.0


def score_response_type(predicted: Optional[str], ground_truth: str) -> float:
    """Score a response type prediction.

    - Exact match: 1.0
    - Acknowledge when escalate expected: 0.2 (safe but not ideal)
    - Other mismatch: 0.0
    """
    if not predicted:
        return 0.0

    predicted_norm = predicted.strip().lower()

    if predicted_norm == ground_truth:
        return 1.0

    # Partial credit: acknowledge is always a "safe" response
    if predicted_norm == "acknowledge" and ground_truth != "acknowledge":
        return 0.2

    # Partial credit: escalate when delegate is expected (being cautious is okay)
    if predicted_norm == "escalate" and ground_truth == "delegate":
        return 0.3

    return 0.0


def compute_efficiency_bonus(steps_used: int, max_steps: int, emails_count: int) -> float:
    """Compute efficiency bonus based on steps used.

    Optimal: ~2 steps per email (read + act)
    Returns value between 0.5 and 1.0
    """
    if max_steps <= 0:
        return 0.5
    optimal_steps = emails_count * 2  # read + classify per email
    if steps_used <= optimal_steps:
        return 1.0
    ratio = steps_used / max_steps
    return max(0.5, 1.0 - ratio * 0.5)


def grade_task(
    task_name: str,
    actions_log: List[Dict[str, Any]],
    ground_truths: Dict[str, Dict[str, str]],
    steps_used: int,
    max_steps: int,
    total_emails: int,
) -> Tuple[float, Dict[str, Any]]:
    """Grade a completed task episode.

    Args:
        task_name: Task identifier
        actions_log: List of action records with email_id and predictions
        ground_truths: Dict of email_id -> {priority, department, response_type}
        steps_used: Total steps taken
        max_steps: Maximum allowed steps
        total_emails: Total emails in the inbox

    Returns:
        Tuple of (score, details_dict)
    """
    if task_name == "priority_classification":
        return _grade_priority(actions_log, ground_truths, steps_used, max_steps, total_emails)
    elif task_name == "route_and_classify":
        return _grade_route_and_classify(actions_log, ground_truths, steps_used, max_steps, total_emails)
    elif task_name == "full_triage":
        return _grade_full_triage(actions_log, ground_truths, steps_used, max_steps, total_emails)
    else:
        return clamp_score(0.0), {"error": f"Unknown task: {task_name}"}


def _grade_priority(
    actions_log: List[Dict],
    ground_truths: Dict[str, Dict[str, str]],
    steps_used: int,
    max_steps: int,
    total_emails: int,
) -> Tuple[float, Dict]:
    """Grade priority classification task."""
    priority_scores = []
    classified_emails = set()

    for action_record in actions_log:
        if action_record.get("action_type") == "classify":
            email_id = action_record.get("email_id")
            if email_id and email_id in ground_truths and email_id not in classified_emails:
                gt = ground_truths[email_id]["priority"]
                pred = action_record.get("priority_label")
                s = score_priority(pred, gt)
                priority_scores.append(s)
                classified_emails.add(email_id)

    if not priority_scores:
        return clamp_score(0.0), {"error": "No classifications made"}

    accuracy = sum(priority_scores) / total_emails  # penalize unclassified emails
    efficiency = compute_efficiency_bonus(steps_used, max_steps, total_emails)

    # 85% accuracy, 15% efficiency
    final = accuracy * 0.85 + efficiency * 0.15
    return clamp_score(final), {
        "accuracy": accuracy,
        "efficiency": efficiency,
        "classified": len(classified_emails),
        "total": total_emails,
        "per_email_scores": priority_scores,
    }


def _grade_route_and_classify(
    actions_log: List[Dict],
    ground_truths: Dict[str, Dict[str, str]],
    steps_used: int,
    max_steps: int,
    total_emails: int,
) -> Tuple[float, Dict]:
    """Grade route and classify task."""
    priority_scores = []
    routing_scores = []
    processed_classify = set()
    processed_route = set()

    for action_record in actions_log:
        email_id = action_record.get("email_id")
        if not email_id or email_id not in ground_truths:
            continue

        if action_record.get("action_type") == "classify" and email_id not in processed_classify:
            gt = ground_truths[email_id]["priority"]
            pred = action_record.get("priority_label")
            priority_scores.append(score_priority(pred, gt))
            processed_classify.add(email_id)

        if action_record.get("action_type") == "route" and email_id not in processed_route:
            gt = ground_truths[email_id]["department"]
            pred = action_record.get("department")
            routing_scores.append(score_routing(pred, gt))
            processed_route.add(email_id)

    if not priority_scores and not routing_scores:
        return clamp_score(0.0), {"error": "No actions taken"}

    priority_acc = sum(priority_scores) / total_emails if priority_scores else 0.0
    routing_acc = sum(routing_scores) / total_emails if routing_scores else 0.0
    efficiency = compute_efficiency_bonus(steps_used, max_steps, total_emails)

    # 40% priority + 40% routing + 20% efficiency
    final = priority_acc * 0.40 + routing_acc * 0.40 + efficiency * 0.20
    return clamp_score(final), {
        "priority_accuracy": priority_acc,
        "routing_accuracy": routing_acc,
        "efficiency": efficiency,
        "classified": len(processed_classify),
        "routed": len(processed_route),
        "total": total_emails,
    }


def _grade_full_triage(
    actions_log: List[Dict],
    ground_truths: Dict[str, Dict[str, str]],
    steps_used: int,
    max_steps: int,
    total_emails: int,
) -> Tuple[float, Dict]:
    """Grade full triage task."""
    priority_scores = []
    routing_scores = []
    response_scores = []
    processed_classify = set()
    processed_route = set()
    processed_respond = set()

    for action_record in actions_log:
        email_id = action_record.get("email_id")
        if not email_id or email_id not in ground_truths:
            continue

        if action_record.get("action_type") == "classify" and email_id not in processed_classify:
            gt = ground_truths[email_id]["priority"]
            pred = action_record.get("priority_label")
            priority_scores.append(score_priority(pred, gt))
            processed_classify.add(email_id)

        if action_record.get("action_type") == "route" and email_id not in processed_route:
            gt = ground_truths[email_id]["department"]
            pred = action_record.get("department")
            routing_scores.append(score_routing(pred, gt))
            processed_route.add(email_id)

        if action_record.get("action_type") == "respond" and email_id not in processed_respond:
            gt = ground_truths[email_id]["response_type"]
            pred = action_record.get("response_type")
            response_scores.append(score_response_type(pred, gt))
            processed_respond.add(email_id)

    if not priority_scores and not routing_scores and not response_scores:
        return clamp_score(0.0), {"error": "No actions taken"}

    priority_acc = sum(priority_scores) / total_emails if priority_scores else 0.0
    routing_acc = sum(routing_scores) / total_emails if routing_scores else 0.0
    response_acc = sum(response_scores) / total_emails if response_scores else 0.0
    efficiency = compute_efficiency_bonus(steps_used, max_steps, total_emails)

    # 30% priority + 25% routing + 30% response + 15% efficiency
    final = priority_acc * 0.30 + routing_acc * 0.25 + response_acc * 0.30 + efficiency * 0.15
    return clamp_score(final), {
        "priority_accuracy": priority_acc,
        "routing_accuracy": routing_acc,
        "response_accuracy": response_acc,
        "efficiency": efficiency,
        "classified": len(processed_classify),
        "routed": len(processed_route),
        "responded": len(processed_respond),
        "total": total_emails,
    }


def compute_step_reward(
    action_type: str,
    email_id: Optional[str],
    action_result: Dict[str, Any],
    ground_truths: Dict[str, Dict[str, str]],
    already_read: set,
    already_classified: set,
    task_name: str,
) -> float:
    """Compute per-step reward for immediate feedback.

    Returns a reward value (can be negative for penalties).
    """
    reward = 0.0

    if action_type == "read":
        # Small positive reward for reading an email
        reward = 0.05

    elif action_type == "classify":
        if email_id and email_id in ground_truths:
            if email_id not in already_read:
                # Penalty: classifying without reading
                reward = -0.3
            elif email_id in already_classified:
                # Penalty: duplicate classification
                reward = -0.1
            else:
                gt = ground_truths[email_id]["priority"]
                pred = action_result.get("priority_label")
                reward = score_priority(pred, gt) * 0.3
        else:
            reward = -0.05  # invalid email

    elif action_type == "route":
        if email_id and email_id in ground_truths:
            if email_id not in already_read:
                reward = -0.3
            else:
                gt = ground_truths[email_id]["department"]
                pred = action_result.get("department")
                reward = score_routing(pred, gt) * 0.25
        else:
            reward = -0.05

    elif action_type == "respond":
        if email_id and email_id in ground_truths:
            if email_id not in already_read:
                reward = -0.3
            else:
                gt = ground_truths[email_id]["response_type"]
                pred = action_result.get("response_type")
                reward = score_response_type(pred, gt) * 0.2
        else:
            reward = -0.05

    elif action_type == "skip":
        # Penalize skipping HIGH priority emails
        if email_id and email_id in ground_truths:
            if ground_truths[email_id]["priority"] == "HIGH":
                reward = -0.4
            else:
                reward = 0.02  # small positive for efficient skipping of low prio
        else:
            reward = 0.0

    elif action_type == "finish":
        reward = 0.0  # neutral; final score computed at end

    else:
        reward = -0.05  # invalid action type

    return round(reward, 4)
