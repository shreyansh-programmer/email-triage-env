"""FounderForge — Top-level FastAPI Server (Docker entrypoint).

The Dockerfile CMD runs: cd /app/env && uvicorn server.app:app
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from openenv.core.env_server import create_fastapi_app
from founderforge_env.server.environment import FounderForgeEnvironment
from founderforge_env.models import FounderForgeAction, FounderForgeObservation

app = create_fastapi_app(
    env=FounderForgeEnvironment,
    action_cls=FounderForgeAction,
    observation_cls=FounderForgeObservation,
)

_web_env = FounderForgeEnvironment()


@app.get("/")
async def serve_web_ui():
    static_dir = Path(__file__).parent.parent / "founderforge_env" / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse({"status": "FounderForge API running", "endpoints": ["/reset", "/step", "/health"]})





def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
