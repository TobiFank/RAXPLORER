# app/api/v1/deps.py
from typing import Generator
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.services.model_config import ModelConfigService
from app.utils.vector_store import MilvusVectorStore
from app.core.config import settings
from fastapi import Depends
from app.services.file.processor import FileProcessor
from app.services.rag.processor import RAGProcessor
from app.services.llm.ollama import OllamaService

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_model_config_service(db: Session = Depends(get_db)) -> ModelConfigService:
    return ModelConfigService(db)

async def get_vector_store() -> MilvusVectorStore:
    vector_store = MilvusVectorStore(
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
        collection_name=settings.MILVUS_COLLECTION,
        dim=settings.MILVUS_VECTOR_DIM
    )
    await vector_store.initialize()
    return vector_store

async def get_rag_processor():
    """Create and initialize RAG processor"""
    vector_store = MilvusVectorStore(
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
        collection_name=settings.MILVUS_COLLECTION,
        dim=settings.MILVUS_VECTOR_DIM
    )
    await vector_store.initialize()

    llm_service = OllamaService(base_url=settings.OLLAMA_HOST)
    await llm_service.initialize()

    return RAGProcessor(llm_service, vector_store)

async def get_file_processor(rag_processor: RAGProcessor = Depends(get_rag_processor)):
    """Create file processor"""
    return FileProcessor(rag_processor)