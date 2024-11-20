# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/dbname"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    # API
    API_V1_PREFIX: str = "/api/v1"

    # Reraanking
    RERANKING_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Ollama
    OLLAMA_HOST: str = "http://ollama:11434"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    class Config:
        env_file = ".env"