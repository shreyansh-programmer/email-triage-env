# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Email Triage Environment — OpenEnv.

An AI agent environment that simulates inbox management.
Agents must read, classify, route, and respond to emails
under time pressure with partial information.
"""

from .client import EmailTriageEnv
from .models import EmailTriageAction, EmailTriageObservation

__all__ = [
    "EmailTriageAction",
    "EmailTriageObservation",
    "EmailTriageEnv",
]
