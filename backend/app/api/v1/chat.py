# app/api/v1/chat.py
from typing import List, AsyncGenerator

from app.core.config import settings
from app.core.database import SessionLocal
from app.schemas.chat import (
    ChatCreate,
    ChatResponse,
    ChatListResponse,
    ChatUpdate,
    MessageCreate
)
from app.services.chat import ChatService
from app.services.llm.factory import create_llm_service
from app.services.model_config import ModelConfigService
from app.services.rag.processor import RAGProcessor
from app.utils.vector_store import MilvusVectorStore
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

router = APIRouter()


# Dependency for database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Dependency for vector store
async def get_vector_store():
    vector_store = MilvusVectorStore(
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
        collection_name=settings.MILVUS_COLLECTION
    )
    await vector_store.initialize()
    return vector_store


# Dependency for RAG processor that uses the same LLM service as chat
async def get_rag_processor(
        db: Session = Depends(get_db),
        vector_store: MilvusVectorStore = Depends(get_vector_store),
        model_config: dict = None  # Will be passed from the chat endpoint
):
    if not model_config:
        # If no model config provided, use a default one
        model_config_service = ModelConfigService(db)
        # Get the first available config as default
        default_config = model_config_service.get_default_config()
        if not default_config:
            raise HTTPException(
                status_code=500,
                detail="No model configuration available"
            )
        model_config = {
            "provider": default_config.provider,
            "model": default_config.model,
            "temperature": 0.0  # Always use 0 temperature for embeddings
        }

    # Create LLM service using the same provider and model as chat
    llm_service = await create_llm_service(
        provider=model_config["provider"],
        api_key=None,  # Will be fetched from saved config
        model=model_config["model"]
    )

    return RAGProcessor(
        llm_service=llm_service,
        vector_store=vector_store,
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP
    )


# Dependency for chat service
async def get_chat_service(
        db: Session = Depends(get_db),
        model_config: dict = None
) -> ChatService:
    vector_store = await get_vector_store()
    rag_processor = await get_rag_processor(db, vector_store, model_config)
    return ChatService(db, rag_processor)


@router.post("/chats/{chat_id}/messages")
async def create_message(
        chat_id: str,
        message: MessageCreate,
        db: Session = Depends(get_db)
) -> StreamingResponse:
    """Create a new message and stream the response"""

    async def message_generator() -> AsyncGenerator[str, None]:
        try:
            model_config = {
                "provider": message.modelConfig.provider,
                "model": message.modelConfig.model,
                "temperature": message.modelConfig.temperature,
            }

            # Create chat service with the current model config
            chat_service = await get_chat_service(db, model_config)

            async for chunk in chat_service.create_message(
                    chat_id=chat_id,
                    content=message.content,
                    model_config=model_config
            ):
                yield chunk
        except Exception as e:
            print(f"Error in message generation: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        message_generator(),
        media_type='text/event-stream'
    )


@router.post("/chats", response_model=ChatResponse)
async def create_chat(
        db: Session = Depends(get_db)
):
    """Create a new chat session"""
    chat_service = await get_chat_service(db)
    chat_create = ChatCreate(title="New Chat")
    return chat_service.create_chat(chat_create)


@router.get("/chats", response_model=List[ChatListResponse])
async def list_chats(
        db: Session = Depends(get_db)
):
    chat_service = await get_chat_service(db)
    return chat_service.list_chats()


@router.get("/chats/{chat_id}", response_model=ChatResponse)
async def get_chat(
        chat_id: str,
        db: Session = Depends(get_db)
):
    chat_service = await get_chat_service(db)
    return chat_service.get_chat(chat_id)


@router.patch("/chats/{chat_id}", response_model=ChatResponse)
async def update_chat(
        chat_id: str,
        chat_update: ChatUpdate,
        db: Session = Depends(get_db)
):
    chat_service = await get_chat_service(db)
    return chat_service.update_chat(chat_id, chat_update)


@router.delete("/chats/{chat_id}")
async def delete_chat(
        chat_id: str,
        db: Session = Depends(get_db)
):
    chat_service = await get_chat_service(db)
    chat_service.delete_chat(chat_id)
    return {"status": "success"}
