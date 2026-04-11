"""FounderForge — FastAPI HTTP Server with Web Interface.

Wraps the FounderForgeEnvironment with the standard OpenEnv HTTP interface
(/reset, /step, /state, /health) plus an interactive web dashboard at /.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from openenv.core.env_server import create_fastapi_app
from .environment import FounderForgeEnvironment
from ..models import FounderForgeAction, FounderForgeObservation

# Create the base OpenEnv app with standard endpoints
app = create_fastapi_app(
    env=FounderForgeEnvironment,
    action_cls=FounderForgeAction,
    observation_cls=FounderForgeObservation,
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
    return JSONResponse({"status": "FounderForge API running", "endpoints": ["/reset", "/step", "/health"]})




def main():
    import uvicorn
    uvicorn.run("founderforge_env.server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
