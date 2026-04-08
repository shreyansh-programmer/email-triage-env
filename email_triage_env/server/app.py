# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Email Triage RL Environment (OpenEnv).

This is a Reinforcement Learning environment built on the OpenEnv framework.
Agents interact with a simulated email inbox through step()/reset()/state() APIs,
receiving rewards for correct triage decisions.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /web: Interactive web interface for human exploration

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os

# Enable the built-in OpenEnv web interface for interactive exploration
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import EmailTriageAction, EmailTriageObservation
    from .email_triage_env_environment import EmailTriageEnvironment
except (ImportError, SystemError):
    from models import EmailTriageAction, EmailTriageObservation
    from server.email_triage_env_environment import EmailTriageEnvironment


# Create the app with web interface and README integration
app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage_env",
    max_concurrent_envs=3,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution.

    Enables running the server via:
        uv run --project . server
        python -m email_triage_env.server.app
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
