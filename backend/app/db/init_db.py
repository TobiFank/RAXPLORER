# app/db/init_db.py
import asyncio
import logging

from sqlalchemy.ext.asyncio import create_async_engine
from .session import Base, Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_db() -> None:
    settings = Settings()

    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=True
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()

    logger.info("Database tables created successfully")

if __name__ == "__main__":
    asyncio.run(init_db())