FROM python:3.10-slim

WORKDIR /app

# Install curl for the validator checks and network debug
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and the source directory
COPY pyproject.toml .
COPY founderforge_env/ ./founderforge_env/

# Install the application and its dependencies
RUN pip install -e ".[dev]"

# Expose standard openenv port
EXPOSE 8000

# Start the uvicorn server via the defined app file
CMD ["uvicorn", "founderforge_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
