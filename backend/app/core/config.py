# app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Chat RAG Backend"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database
    DATABASE_URL: str = "sqlite:///./test.db"

    # Vector DB
    VECTOR_DB_URL: Optional[str] = None
    VECTOR_DB_API_KEY: Optional[str] = None

    # AI Models
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    OLLAMA_HOST: Optional[str] = "http://localhost:11434"

    # File Upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    UPLOAD_DIR: str = "uploads"

    # Security
    RATE_LIMIT_PER_MINUTE: int = 100

    # Milvus Configuration
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "document_chunks"
    MILVUS_VECTOR_DIM: int = 4096  # Matches Llama2 embedding dimension

    # RAG settings
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 200

    # Vector search configuration
    VECTOR_SIMILARITY_METRIC: str = "COSINE"
    VECTOR_TOP_K: int = 3

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()