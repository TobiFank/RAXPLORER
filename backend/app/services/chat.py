# app/services/chat.py
import uuid
from datetime import datetime
from typing import List, Optional, AsyncGenerator

from app.models.chat import Chat, Message
from app.schemas.chat import ChatCreate, ChatUpdate
from app.services.llm.base import LLMConfig
from app.services.llm.factory import create_llm_service
from fastapi import HTTPException
from sqlalchemy.orm import Session


class ChatService:
    def __init__(self, db: Session):
        self.db = db

    def create_chat(self, chat_create: ChatCreate) -> Chat:
        chat = Chat(
            id=str(uuid.uuid4()),
            title=chat_create.title or "New Chat",
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

    async def create_message(
            self,
            chat_id: str,
            content: str,
            model_config: dict
    ) -> AsyncGenerator[str, None]:
        """Create a message and stream the AI response"""
        chat = self.get_chat(chat_id)

        # Create user message
        user_message = Message(
            id=str(uuid.uuid4()),
            chat_id=chat_id,
            role="user",
            content=content,
            timestamp=datetime.utcnow(),
            model_provider=model_config["provider"],
            model_name=model_config.get("ollamaModel") if model_config["provider"] == "ollama" else model_config[
                "model"],
            temperature=model_config["temperature"]
        )
        self.db.add(user_message)
        self.db.commit()

        # Initialize appropriate LLM service
        llm_service = await create_llm_service(
            provider=model_config["provider"],
            api_key=model_config.get("apiKey"),
            model=model_config.get("ollamaModel") if model_config["provider"] == "ollama" else model_config["model"]
        )

        # Prepare LLM config
        llm_config = LLMConfig(
            model=model_config.get("ollamaModel") if model_config["provider"] == "ollama" else model_config["model"],
            temperature=model_config["temperature"],
            top_p=1.0,
            stop_sequences=[],
            extra_params={}
        )

        # Initialize response content
        response_content = ""

        # Stream response from LLM
        try:
            async for chunk in llm_service.generate_stream(content, llm_config):
                response_content += chunk
                yield chunk

            # After streaming is complete, save the assistant message
            assistant_message = Message(
                id=str(uuid.uuid4()),
                chat_id=chat_id,
                role="assistant",
                content=response_content,
                timestamp=datetime.utcnow(),
                model_provider=model_config["provider"],
                model_name=model_config.get("ollamaModel") if model_config["provider"] == "ollama" else model_config[
                    "model"],
                temperature=model_config["temperature"]
            )
            self.db.add(assistant_message)
            self.db.commit()

        except Exception as e:
            # If there's an error, we should still save what we have
            if response_content:
                assistant_message = Message(
                    id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    role="assistant",
                    content=response_content,
                    timestamp=datetime.utcnow(),
                    model_provider=model_config["provider"],
                    model_name=model_config.get("ollamaModel") if model_config["provider"] == "ollama" else
                    model_config["model"],
                    temperature=model_config["temperature"]
                )
                self.db.add(assistant_message)
                self.db.commit()
            raise HTTPException(status_code=500, detail=str(e))
