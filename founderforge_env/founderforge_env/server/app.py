from typing import Dict, Any

from openenv.core.env_server.app import create_app
from .environment import FounderForgeEnvironment

app = create_app(
    environment_cls=FounderForgeEnvironment,
)

def main():
    import uvicorn
    uvicorn.run("founderforge_env.server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
