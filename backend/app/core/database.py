# app/core/database.py
import logging
from typing import Generator

from app.core.config import settings
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
from tenacity import retry, stop_after_attempt, wait_exponential

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency function to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Initialize database tables
def init_db() -> None:
    """Initialize all database tables"""
    Base.metadata.create_all(bind=engine)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def verify_db_connection() -> bool:
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))  # Add text() wrapper
        db.close()
        return True
    except SQLAlchemyError as e:
        logging.error(f"Database connection failed: {e}")
        raise


def init_db(retries: int = 3) -> None:
    """Initialize database with retry logic"""
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logging.error(f"Failed to initialize database: {e}")
        raise
