"""FounderForge — FastAPI HTTP Server (OpenEnv-compatible).

This module is the entrypoint used by the Dockerfile CMD:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

It wraps the FounderForgeEnvironment with the standard OpenEnv HTTP
interface so the platform can call /reset, /step, /state, and /health.
"""

from openenv.core.env_server.app import create_app
from founderforge_env.server.environment import FounderForgeEnvironment

app = create_app(
    environment_cls=FounderForgeEnvironment,
)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
