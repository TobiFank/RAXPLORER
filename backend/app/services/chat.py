# app/services/chat.py
from datetime import datetime
import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.models.chat import Chat, Message
from app.schemas.chat import ChatCreate, ChatUpdate, MessageCreate
from app.services.model import ModelService

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.model_service = ModelService()

    def create_chat(self, chat_create: ChatCreate) -> Chat:
        chat = Chat(
            id=str(uuid.uuid4()),
            title=chat_create.title,
            created_at=datetime.utcnow()
        )
        self.db.add(chat)
        self.db.commit()
        self.db.refresh(chat)
        return chat

    def get_chat(self, chat_id: str) -> Optional[Chat]:
        chat = self.db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat

    def list_chats(self) -> List[Chat]:
        return self.db.query(Chat).order_by(Chat.created_at.desc()).all()

    def update_chat(self, chat_id: str, chat_update: ChatUpdate) -> Chat:
        chat = self.get_chat(chat_id)
        for key, value in chat_update.dict(exclude_unset=True).items():
            setattr(chat, key, value)
        self.db.commit()
        self.db.refresh(chat)
        return chat

    def delete_chat(self, chat_id: str) -> None:
        chat = self.get_chat(chat_id)
        self.db.delete(chat)
        self.db.commit()

    async def create_message(self, chat_id: str, message_create: MessageCreate) -> Message:
        chat = self.get_chat(chat_id)

        # Create user message
        user_message = Message(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            role="user",
            content=message_create.content,
            timestamp=datetime.utcnow(),
            model_provider=message_create.ai_config.provider,  # Changed from model_config to ai_config
            model_name=message_create.ai_config.model,
            temperature=message_create.ai_config.temperature
        )
        self.db.add(user_message)

        # Generate AI response
        response_content = await self.model_service.generate_response(
            message_create.content,
            message_create.ai_config  # Changed from model_config to ai_config
        )

        # Create assistant message
        assistant_message = Message(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            role="assistant",
            content=response_content,
            timestamp=datetime.utcnow(),
            model_provider=message_create.ai_config.provider,  # Changed from model_config to ai_config
            model_name=message_create.ai_config.model,
            temperature=message_create.ai_config.temperature
        )
        self.db.add(assistant_message)

        self.db.commit()
        return assistant_message