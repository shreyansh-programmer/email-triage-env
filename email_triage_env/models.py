# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Email Triage Environment.

Defines typed Action and Observation models for email triage:
- Classifying priority (HIGH / MEDIUM / LOW)
- Routing to departments (Engineering, Sales, Legal, HR, Support, Executive)
- Composing responses (acknowledge, escalate, delegate, decline, info_request)
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class EmailTriageAction(Action):
    """Action the agent takes on the current email.

    action_type determines what the agent is doing:
      - "read"       : Read the next email (no other fields required)
      - "classify"   : Classify the current email (requires priority_label)
      - "route"      : Route the current email (requires department)
      - "respond"    : Compose a response (requires response_type, optionally response_text)
      - "skip"       : Skip the current email (no other fields required)
      - "finish"     : End the episode early
    """

    action_type: str = Field(
        ...,
        description="Action type: read, classify, route, respond, skip, or finish",
    )
    priority_label: Optional[str] = Field(
        default=None,
        description="Priority label: HIGH, MEDIUM, or LOW (used with classify)",
    )
    department: Optional[str] = Field(
        default=None,
        description="Department to route to: Engineering, Sales, Legal, HR, Support, Executive (used with route)",
    )
    response_type: Optional[str] = Field(
        default=None,
        description="Response type: acknowledge, escalate, delegate, decline, info_request (used with respond)",
    )
    response_text: Optional[str] = Field(
        default=None,
        description="Optional free-text response content (used with respond)",
    )


class EmailTriageObservation(Observation):
    """Observation returned to the agent after each step.

    Contains the current email being viewed, inbox status,
    and feedback from the last action.
    """

    # Current email details (populated after a 'read' action)
    current_email_id: Optional[str] = Field(default=None, description="ID of the current email")
    email_from: Optional[str] = Field(default=None, description="Sender name and title")
    email_subject: Optional[str] = Field(default=None, description="Email subject line")
    email_body: Optional[str] = Field(default=None, description="Email body text")
    email_timestamp: Optional[str] = Field(default=None, description="When the email was received")
    email_thread_id: Optional[str] = Field(default=None, description="Thread ID for related emails")

    # Inbox status
    total_emails: int = Field(default=0, description="Total emails in inbox")
    emails_processed: int = Field(default=0, description="Number of emails processed so far")
    emails_remaining: int = Field(default=0, description="Number of unprocessed emails")

    # Step budget
    steps_used: int = Field(default=0, description="Steps used so far")
    max_steps: int = Field(default=50, description="Maximum allowed steps")

    # Feedback from last action
    last_action_result: str = Field(default="", description="Result of the last action taken")
    last_action_error: Optional[str] = Field(default=None, description="Error from last action, if any")

    # Task info
    task_name: str = Field(default="", description="Current task name")
    task_description: str = Field(default="", description="Current task instructions")

    # Available actions hint
    available_actions: List[str] = Field(
        default_factory=lambda: ["read", "classify", "route", "respond", "skip", "finish"],
        description="Actions available in the current state",
    )
