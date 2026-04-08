# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Email Triage Environment Implementation.

Simulates an executive assistant managing an overflowing inbox.
The agent must read, classify, route, and respond to emails
under time/step pressure with partial information.

Supports three tasks:
  - priority_classification (easy): classify 10 emails as HIGH/MEDIUM/LOW
  - route_and_classify (medium): classify + route 15 emails to departments
  - full_triage (hard): classify + route + respond for 20 emails with threads
"""

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import EmailTriageAction, EmailTriageObservation
    from ..email_data import Email, generate_email_corpus
    from ..evaluation import (
        VALID_PRIORITIES,
        VALID_DEPARTMENTS,
        VALID_RESPONSE_TYPES,
        compute_step_reward,
        grade_task,
        clamp_score,
    )
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation
    from email_data import Email, generate_email_corpus
    from evaluation import (
        VALID_PRIORITIES,
        VALID_DEPARTMENTS,
        VALID_RESPONSE_TYPES,
        compute_step_reward,
        grade_task,
        clamp_score,
    )


# Task configurations
TASK_CONFIGS = {
    "priority_classification": {
        "description": (
            "Classify each email in the inbox as HIGH, MEDIUM, or LOW priority. "
            "Read each email carefully, then use the 'classify' action with a priority_label. "
            "You get partial credit for off-by-one errors (e.g., HIGH vs MEDIUM). "
            "Process all emails efficiently to maximize your score."
        ),
        "max_steps": 35,
        "email_count": 10,
        "required_actions": ["read", "classify", "skip", "finish"],
        "seed": 42,
    },
    "route_and_classify": {
        "description": (
            "For each email: (1) classify priority as HIGH/MEDIUM/LOW, and "
            "(2) route to the correct department (Engineering, Sales, Legal, HR, Support, or Executive). "
            "Read each email first, then classify and route. You get partial credit for "
            "related departments. Efficiency matters — minimize wasted steps."
        ),
        "max_steps": 55,
        "email_count": 15,
        "required_actions": ["read", "classify", "route", "skip", "finish"],
        "seed": 123,
    },
    "full_triage": {
        "description": (
            "Complete inbox triage: for each email, (1) classify priority, (2) route to department, "
            "and (3) choose the right response type (acknowledge, escalate, delegate, decline, info_request). "
            "Some emails are part of threads — read related messages for context. "
            "HIGH priority emails from VIP senders should be escalated, not just acknowledged. "
            "Balance thoroughness with efficiency."
        ),
        "max_steps": 80,
        "email_count": 20,
        "required_actions": ["read", "classify", "route", "respond", "skip", "finish"],
        "seed": 777,
    },
}


class EmailTriageEnvironment(Environment):
    """
    Email Triage Environment.

    Simulates inbox management where an agent must read, classify, route,
    and respond to emails. Supports three difficulty levels as separate tasks.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name: str = "priority_classification"
        self._emails: List[Email] = []
        self._ground_truths: Dict[str, Dict[str, str]] = {}
        self._current_email_idx: int = -1
        self._current_email: Optional[Email] = None
        self._actions_log: List[Dict[str, Any]] = []
        self._already_read: set = set()
        self._already_classified: set = set()
        self._already_routed: set = set()
        self._already_responded: set = set()
        self._max_steps: int = 35
        self._done: bool = False
        self._total_reward: float = 0.0
        self._episode_rewards: List[float] = []

    def reset(self, **kwargs) -> EmailTriageObservation:
        """Reset the environment for a new episode.

        Accepts optional task_name in kwargs to select difficulty level.
        """
        # Determine task
        task_name = kwargs.get("task_name") or os.environ.get(
            "EMAIL_TRIAGE_TASK", "priority_classification"
        )
        if task_name not in TASK_CONFIGS:
            task_name = "priority_classification"

        self._task_name = task_name
        config = TASK_CONFIGS[task_name]

        # Generate email corpus
        self._emails = generate_email_corpus(task_name, seed=config["seed"])
        self._ground_truths = {
            e.id: {
                "priority": e.gt_priority,
                "department": e.gt_department,
                "response_type": e.gt_response_type,
            }
            for e in self._emails
        }

        # Reset state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_email_idx = -1
        self._current_email = None
        self._actions_log = []
        self._already_read = set()
        self._already_classified = set()
        self._already_routed = set()
        self._already_responded = set()
        self._max_steps = config["max_steps"]
        self._done = False
        self._total_reward = 0.0
        self._episode_rewards = []

        return EmailTriageObservation(
            done=False,
            reward=0.0,
            current_email_id=None,
            email_from=None,
            email_subject=None,
            email_body=None,
            email_timestamp=None,
            email_thread_id=None,
            total_emails=len(self._emails),
            emails_processed=0,
            emails_remaining=len(self._emails),
            steps_used=0,
            max_steps=self._max_steps,
            last_action_result="Environment reset. Use 'read' to view the first email.",
            last_action_error=None,
            task_name=self._task_name,
            task_description=config["description"],
            available_actions=config["required_actions"],
            metadata={
                "task": self._task_name,
                "total_emails": len(self._emails),
                "max_steps": self._max_steps,
            },
        )

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:  # type: ignore[override]
        """Execute one step in the environment."""
        self._state.step_count += 1

        # Check if already done
        if self._done:
            return self._make_observation(
                reward=0.0,
                result="Episode already finished.",
                error=None,
            )

        # Check step budget
        if self._state.step_count > self._max_steps:
            self._done = True
            final_score, details = grade_task(
                self._task_name,
                self._actions_log,
                self._ground_truths,
                self._state.step_count,
                self._max_steps,
                len(self._emails),
            )
            return self._make_observation(
                reward=final_score,
                result=f"Step budget exhausted. Final score: {final_score:.3f}",
                error=None,
                done=True,
                metadata={"final_score": final_score, "grading_details": details},
            )

        action_type = action.action_type.lower().strip()

        # Dispatch action
        if action_type == "read":
            return self._handle_read()
        elif action_type == "classify":
            return self._handle_classify(action)
        elif action_type == "route":
            return self._handle_route(action)
        elif action_type == "respond":
            return self._handle_respond(action)
        elif action_type == "skip":
            return self._handle_skip()
        elif action_type == "finish":
            return self._handle_finish()
        else:
            # Invalid action
            reward = -0.05
            self._episode_rewards.append(reward)
            self._total_reward += reward
            return self._make_observation(
                reward=reward,
                result="",
                error=f"Invalid action_type '{action_type}'. Valid: read, classify, route, respond, skip, finish",
            )

    def _handle_read(self) -> EmailTriageObservation:
        """Handle 'read' action: advance to next email."""
        self._current_email_idx += 1

        if self._current_email_idx >= len(self._emails):
            # All emails have been presented
            self._done = True
            final_score, details = grade_task(
                self._task_name,
                self._actions_log,
                self._ground_truths,
                self._state.step_count,
                self._max_steps,
                len(self._emails),
            )
            return self._make_observation(
                reward=final_score,
                result=f"All emails processed. Final score: {final_score:.3f}",
                error=None,
                done=True,
                metadata={"final_score": final_score, "grading_details": details},
            )

        email = self._emails[self._current_email_idx]
        self._current_email = email
        self._already_read.add(email.id)

        reward = compute_step_reward(
            "read", email.id, {}, self._ground_truths,
            self._already_read, self._already_classified, self._task_name,
        )
        self._episode_rewards.append(reward)
        self._total_reward += reward

        return EmailTriageObservation(
            done=False,
            reward=reward,
            current_email_id=email.id,
            email_from=f"{email.sender_name}, {email.sender_title} <{email.sender_email}>",
            email_subject=email.subject,
            email_body=email.body,
            email_timestamp=email.timestamp,
            email_thread_id=email.thread_id,
            total_emails=len(self._emails),
            emails_processed=self._current_email_idx,
            emails_remaining=len(self._emails) - self._current_email_idx - 1,
            steps_used=self._state.step_count,
            max_steps=self._max_steps,
            last_action_result=f"Reading email {email.id}: '{email.subject}'",
            last_action_error=None,
            task_name=self._task_name,
            task_description=TASK_CONFIGS[self._task_name]["description"],
            available_actions=TASK_CONFIGS[self._task_name]["required_actions"],
            metadata={"email_id": email.id, "step": self._state.step_count},
        )

    def _handle_classify(self, action: EmailTriageAction) -> EmailTriageObservation:
        """Handle 'classify' action: assign priority label."""
        if self._current_email is None:
            reward = -0.05
            self._episode_rewards.append(reward)
            self._total_reward += reward
            return self._make_observation(
                reward=reward,
                result="",
                error="No email currently loaded. Use 'read' first.",
            )

        email = self._current_email
        priority = (action.priority_label or "").upper().strip()

        if priority not in VALID_PRIORITIES:
            reward = -0.05
            self._episode_rewards.append(reward)
            self._total_reward += reward
            return self._make_observation(
                reward=reward,
                result="",
                error=f"Invalid priority '{priority}'. Must be HIGH, MEDIUM, or LOW.",
            )

        action_record = {
            "action_type": "classify",
            "email_id": email.id,
            "priority_label": priority,
        }
        self._actions_log.append(action_record)

        reward = compute_step_reward(
            "classify", email.id, {"priority_label": priority},
            self._ground_truths, self._already_read,
            self._already_classified, self._task_name,
        )
        self._already_classified.add(email.id)
        self._episode_rewards.append(reward)
        self._total_reward += reward

        return self._make_observation(
            reward=reward,
            result=f"Classified email {email.id} as {priority} priority.",
            error=None,
        )

    def _handle_route(self, action: EmailTriageAction) -> EmailTriageObservation:
        """Handle 'route' action: assign to department."""
        if self._current_email is None:
            reward = -0.05
            self._episode_rewards.append(reward)
            self._total_reward += reward
            return self._make_observation(
                reward=reward,
                result="",
                error="No email currently loaded. Use 'read' first.",
            )

        email = self._current_email
        department = (action.department or "").strip().title()

        if department not in VALID_DEPARTMENTS:
            reward = -0.05
            self._episode_rewards.append(reward)
            self._total_reward += reward
            return self._make_observation(
                reward=reward,
                result="",
                error=f"Invalid department '{department}'. Must be one of: {', '.join(sorted(VALID_DEPARTMENTS))}",
            )

        action_record = {
            "action_type": "route",
            "email_id": email.id,
            "department": department,
        }
        self._actions_log.append(action_record)

        reward = compute_step_reward(
            "route", email.id, {"department": department},
            self._ground_truths, self._already_read,
            self._already_classified, self._task_name,
        )
        self._already_routed.add(email.id)
        self._episode_rewards.append(reward)
        self._total_reward += reward

        return self._make_observation(
            reward=reward,
            result=f"Routed email {email.id} to {department} department.",
            error=None,
        )

    def _handle_respond(self, action: EmailTriageAction) -> EmailTriageObservation:
        """Handle 'respond' action: choose response type."""
        if self._current_email is None:
            reward = -0.05
            self._episode_rewards.append(reward)
            self._total_reward += reward
            return self._make_observation(
                reward=reward,
                result="",
                error="No email currently loaded. Use 'read' first.",
            )

        email = self._current_email
        response_type = (action.response_type or "").strip().lower()

        if response_type not in VALID_RESPONSE_TYPES:
            reward = -0.05
            self._episode_rewards.append(reward)
            self._total_reward += reward
            return self._make_observation(
                reward=reward,
                result="",
                error=f"Invalid response_type '{response_type}'. Must be one of: {', '.join(sorted(VALID_RESPONSE_TYPES))}",
            )

        action_record = {
            "action_type": "respond",
            "email_id": email.id,
            "response_type": response_type,
            "response_text": action.response_text or "",
        }
        self._actions_log.append(action_record)

        reward = compute_step_reward(
            "respond", email.id, {"response_type": response_type},
            self._ground_truths, self._already_read,
            self._already_classified, self._task_name,
        )
        self._already_responded.add(email.id)
        self._episode_rewards.append(reward)
        self._total_reward += reward

        return self._make_observation(
            reward=reward,
            result=f"Responded to email {email.id} with '{response_type}'.",
            error=None,
        )

    def _handle_skip(self) -> EmailTriageObservation:
        """Handle 'skip' action: skip current email."""
        email = self._current_email
        email_id = email.id if email else None

        reward = compute_step_reward(
            "skip", email_id, {},
            self._ground_truths, self._already_read,
            self._already_classified, self._task_name,
        )
        self._episode_rewards.append(reward)
        self._total_reward += reward

        result = f"Skipped email {email_id}." if email_id else "Nothing to skip."
        return self._make_observation(reward=reward, result=result, error=None)

    def _handle_finish(self) -> EmailTriageObservation:
        """Handle 'finish' action: end episode and compute final score."""
        self._done = True

        final_score, details = grade_task(
            self._task_name,
            self._actions_log,
            self._ground_truths,
            self._state.step_count,
            self._max_steps,
            len(self._emails),
        )

        return self._make_observation(
            reward=final_score,
            result=f"Episode finished. Final score: {final_score:.3f}",
            error=None,
            done=True,
            metadata={"final_score": final_score, "grading_details": details},
        )

    def _make_observation(
        self,
        reward: float,
        result: str,
        error: Optional[str],
        done: bool = False,
        metadata: Optional[Dict] = None,
    ) -> EmailTriageObservation:
        """Helper to construct an observation."""
        email = self._current_email
        config = TASK_CONFIGS[self._task_name]

        obs = EmailTriageObservation(
            done=done or self._done,
            reward=reward,
            current_email_id=email.id if email else None,
            email_from=f"{email.sender_name}, {email.sender_title} <{email.sender_email}>" if email else None,
            email_subject=email.subject if email else None,
            email_body=email.body if email else None,
            email_timestamp=email.timestamp if email else None,
            email_thread_id=email.thread_id if email else None,
            total_emails=len(self._emails),
            emails_processed=max(0, self._current_email_idx + 1) if self._current_email_idx >= 0 else 0,
            emails_remaining=max(0, len(self._emails) - self._current_email_idx - 1) if self._current_email_idx >= 0 else len(self._emails),
            steps_used=self._state.step_count,
            max_steps=self._max_steps,
            last_action_result=result,
            last_action_error=error,
            task_name=self._task_name,
            task_description=config["description"],
            available_actions=config["required_actions"],
            metadata=metadata or {"step": self._state.step_count},
        )
        return obs

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
