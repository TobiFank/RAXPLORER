# app/api/v1/chat.py
from typing import List, AsyncGenerator

from app.core.database import SessionLocal
from app.schemas.chat import (
    ChatCreate,
    ChatResponse,
    ChatListResponse,
    ChatUpdate,
    MessageCreate
)
from app.services.chat import ChatService
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

router = APIRouter()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/chats", response_model=ChatResponse)
async def create_chat(
        chat_create: ChatCreate,
        db: Session = Depends(get_db)
):
    chat_service = ChatService(db)
    return chat_service.create_chat(chat_create)


@router.get("/chats", response_model=List[ChatListResponse])
async def list_chats(db: Session = Depends(get_db)):
    chat_service = ChatService(db)
    return chat_service.list_chats()


@router.get("/chats/{chat_id}", response_model=ChatResponse)
async def get_chat(
        chat_id: str,
        db: Session = Depends(get_db)
):
    chat_service = ChatService(db)
    return chat_service.get_chat(chat_id)


@router.patch("/chats/{chat_id}", response_model=ChatResponse)
async def update_chat(
        chat_id: str,
        chat_update: ChatUpdate,
        db: Session = Depends(get_db)
):
    chat_service = ChatService(db)
    return chat_service.update_chat(chat_id, chat_update)


@router.delete("/chats/{chat_id}")
async def delete_chat(
        chat_id: str,
        db: Session = Depends(get_db)
):
    chat_service = ChatService(db)
    chat_service.delete_chat(chat_id)
    return {"status": "success"}


@router.post("/chats/{chat_id}/messages")
async def create_message(
        chat_id: str,
        message_create: MessageCreate,
        db: Session = Depends(get_db)
) -> StreamingResponse:
    """Create a new message and stream the response"""
    chat_service = ChatService(db)

    async def message_generator() -> AsyncGenerator[str, None]:
        async for chunk in chat_service.create_message(
                chat_id,
                message_create.content,
                message_create.model_config
        ):
            yield chunk

    return StreamingResponse(
        message_generator(),
        media_type='text/event-stream'
    )
