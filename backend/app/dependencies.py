# app/depedencies.py
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from .db.session import get_db
from .services.chat import ChatService
from .services.model_config import ModelConfigService
from .services.storage import StorageService


async def get_chat_service(
        db: AsyncSession = Depends(get_db),
        llm_service=None,
        rag_service=None
) -> ChatService:
    from .main import app  # Import here to avoid circular import
    return ChatService(db, app.state.llm_service, app.state.rag_service)


async def get_storage_service(
        db: AsyncSession = Depends(get_db),
        rag_service=None
) -> StorageService:
    from .main import app  # Import here to avoid circular import
    return StorageService(db, app.state.rag_service)

async def get_model_config_service(
        db: AsyncSession = Depends(get_db)
) -> ModelConfigService:
    return ModelConfigService(db)