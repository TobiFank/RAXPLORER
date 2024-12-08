# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/dbname"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Full document Storage
    DOCUMENT_STORAGE_PATH: str = "storage/documents"

    # API
    API_V1_PREFIX: str = "/api/v1"

    # Ollama
    OLLAMA_HOST: str = "http://ollama:11434"
