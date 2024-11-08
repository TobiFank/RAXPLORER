# app/api/deps.py
from typing import Generator
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.services.model_config import ModelConfigService
from app.utils.vector_store import MilvusVectorStore
from app.core.config import settings
from fastapi import Depends

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