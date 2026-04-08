# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Email Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import EmailTriageAction, EmailTriageObservation


class EmailTriageEnv(
    EnvClient[EmailTriageAction, EmailTriageObservation, State]
):
    """
    Client for the Email Triage Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> async with EmailTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = await client.reset(task_name="priority_classification")
        ...     print(result.observation.task_name)
        ...
        ...     result = await client.step(EmailTriageAction(action_type="read"))
        ...     print(result.observation.email_subject)

    Example with Docker:
        >>> client = await EmailTriageEnv.from_docker_image("email_triage_env:latest")
        >>> try:
        ...     result = await client.reset()
        ...     result = await client.step(EmailTriageAction(action_type="read"))
        ... finally:
        ...     await client.close()
    """

    def _step_payload(self, action: EmailTriageAction) -> Dict:
        """Convert EmailTriageAction to JSON payload for step message."""
        payload = {"action_type": action.action_type}

        if action.priority_label is not None:
            payload["priority_label"] = action.priority_label
        if action.department is not None:
            payload["department"] = action.department
        if action.response_type is not None:
            payload["response_type"] = action.response_type
        if action.response_text is not None:
            payload["response_text"] = action.response_text

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[EmailTriageObservation]:
        """Parse server response into StepResult[EmailTriageObservation]."""
        obs_data = payload.get("observation", {})

        observation = EmailTriageObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            current_email_id=obs_data.get("current_email_id"),
            email_from=obs_data.get("email_from"),
            email_subject=obs_data.get("email_subject"),
            email_body=obs_data.get("email_body"),
            email_timestamp=obs_data.get("email_timestamp"),
            email_thread_id=obs_data.get("email_thread_id"),
            total_emails=obs_data.get("total_emails", 0),
            emails_processed=obs_data.get("emails_processed", 0),
            emails_remaining=obs_data.get("emails_remaining", 0),
            steps_used=obs_data.get("steps_used", 0),
            max_steps=obs_data.get("max_steps", 50),
            last_action_result=obs_data.get("last_action_result", ""),
            last_action_error=obs_data.get("last_action_error"),
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            available_actions=obs_data.get("available_actions", []),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
