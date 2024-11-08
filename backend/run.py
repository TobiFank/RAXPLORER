# run.py
import uvicorn
import asyncio
from app.main import app
from app.core.database import init_db
from app.core.config import settings

async def startup():
    """Perform startup initialization"""
    # Initialize database tables
    init_db()

    # Additional startup tasks can be added here
    print(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"Database URL: {settings.DATABASE_URL}")
    print(f"API Version: {settings.API_V1_STR}")

if __name__ == "__main__":
    # Run startup tasks
    asyncio.run(startup())

    # Start the FastAPI application
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )