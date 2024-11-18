# app/main.py
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager

from .api import chat, files, models
from .core.config import Settings
from .db.session import get_db
from .services.llm import LLMService
from .services.rag import RAGService
from .services.chat import ChatService
from .services.storage import StorageService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup Services
    settings = Settings()
    llm_service = LLMService()
    rag_service = RAGService(llm_service)

    # Add to app state
    app.state.settings = settings
    app.state.llm_service = llm_service
    app.state.rag_service = rag_service

    yield

app = FastAPI(lifespan=lifespan)

# Dependencies
async def get_chat_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    return ChatService(db, app.state.llm_service, app.state.rag_service)

async def get_storage_service(db: AsyncSession = Depends(get_db)) -> StorageService:
    return StorageService(db, app.state.rag_service)

# Include routers
app.include_router(chat.router, prefix="/api/v1", dependencies=[Depends(get_chat_service)])
app.include_router(files.router, prefix="/api/v1", dependencies=[Depends(get_storage_service)])
app.include_router(models.router, prefix="/api/v1")