"""FounderForge — FastAPI HTTP Server with Web Interface.

This module wraps the FounderForgeEnvironment with:
1. The standard OpenEnv HTTP interface (/reset, /step, /state, /health)
2. A premium interactive web dashboard at the root (/)
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from openenv.core.env_server.app import create_app
from .environment import FounderForgeEnvironment
from ..models import FounderForgeAction

# Create the base OpenEnv app
app = create_app(
    environment_cls=FounderForgeEnvironment,
)

# ── Persistent environment instance for the web UI ───────────────────────
_web_env = FounderForgeEnvironment()


@app.get("/")
async def serve_web_ui():
    """Serve the interactive CEO simulator dashboard."""
    static_dir = Path(__file__).parent.parent / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse({"error": "Web UI not found"}, status_code=404)


@app.post("/reset")
async def web_reset(body: dict = {}):
    """Reset the environment for interactive play."""
    task_name = body.get("task_name", "bootstrap_survival")
    obs = _web_env.reset(task_name=task_name)
    return _obs_to_dict(obs)


@app.post("/step")
async def web_step(body: dict = {}):
    """Execute one step for interactive play."""
    action = FounderForgeAction(
        action_type=body.get("action_type", "skip"),
        tool_name=body.get("tool_name"),
        arguments=body.get("arguments"),
    )
    obs = _web_env.step(action)
    return _obs_to_dict(obs)


def _obs_to_dict(obs) -> dict:
    """Convert observation to a JSON-serializable dict."""
    return {
        "done": obs.done,
        "reward": obs.reward,
        "cash": obs.cash,
        "users": obs.users,
        "product_quality": obs.product_quality,
        "team": obs.team,
        "current_round": obs.current_round,
        "strategy": obs.strategy,
        "last_action_result": obs.last_action_result,
        "task_name": obs.task_name,
        "task_description": obs.task_description,
        "tools_list": obs.tools_list,
        "tool_result": obs.tool_result,
        "metadata": obs.metadata,
    }


def main():
    import uvicorn
    uvicorn.run(
        "founderforge_env.server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
