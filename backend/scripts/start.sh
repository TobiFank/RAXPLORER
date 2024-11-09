#!/bin/bash
set -e

# Wait for Milvus to be ready
echo "Waiting for Milvus..."
timeout 30s bash -c 'until nc -z $MILVUS_HOST 19530; do sleep 1; done'

# Apply database migrations
echo "Applying database migrations..."
poetry run alembic upgrade head

# Start the application
echo "Starting FastAPI application..."
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000