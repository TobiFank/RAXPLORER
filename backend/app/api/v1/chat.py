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
from fastapi import APIRouter, Depends, HTTPException
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


@router.post("/chats/{chat_id}/messages")
async def create_message(
        chat_id: str,
        message: MessageCreate,
        db: Session = Depends(get_db)
) -> StreamingResponse:
    """Create a new message and stream the response"""
    chat_service = ChatService(db)

    async def message_generator() -> AsyncGenerator[str, None]:
        try:
            # Convert modelConfig to a proper dictionary
            model_config = {
                "provider": message.modelConfig.provider,
                "model": message.modelConfig.model,
                "temperature": message.modelConfig.temperature,
                "ollamaModel": message.modelConfig.ollama_model
            }

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
async def create_chat(db: Session = Depends(get_db)):
    """Create a new chat session"""
    chat_service = ChatService(db)
    chat_create = ChatCreate(title="New Chat")  # Create with default title
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
