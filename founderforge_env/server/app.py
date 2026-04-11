"""FounderForge — Top-level FastAPI Server (Docker entrypoint).

The Dockerfile CMD runs: cd /app/env && uvicorn server.app:app
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from openenv.core.env_server import create_fastapi_app
from founderforge_env.server.environment import FounderForgeEnvironment
from founderforge_env.models import FounderForgeAction

app = create_fastapi_app(
    environment_cls=FounderForgeEnvironment,
)

_web_env = FounderForgeEnvironment()


@app.get("/")
async def serve_web_ui():
    static_dir = Path(__file__).parent / "founderforge_env" / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse({"status": "FounderForge API running", "endpoints": ["/reset", "/step", "/health"]})


@app.post("/reset")
async def web_reset(body: dict = {}):
    task_name = body.get("task_name", "bootstrap_survival")
    obs = _web_env.reset(task_name=task_name)
    return _obs_to_dict(obs)


@app.post("/step")
async def web_step(body: dict = {}):
    action = FounderForgeAction(
        action_type=body.get("action_type", "skip"),
        tool_name=body.get("tool_name"),
        arguments=body.get("arguments"),
    )
    obs = _web_env.step(action)
    return _obs_to_dict(obs)


def _obs_to_dict(obs) -> dict:
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
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
