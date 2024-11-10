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

    def _format_chat_history(self, messages: List[Message], max_messages: int = 10) -> str:
        """Format the most recent messages into a conversation history string."""
        # Take the most recent messages up to max_messages
        recent_messages = messages[-max_messages:] if max_messages else messages

        # Format into a conversation string
        formatted_history = []
        for msg in recent_messages:
            # Capitalize role for clarity
            role = msg.role.capitalize()
            formatted_history.append(f"{role}: {msg.content}")

        return "\n\n".join(formatted_history)

    async def create_message(
            self,
            chat_id: str,
            content: str,
            model_config: dict,
            max_history_messages: int = 10
    ) -> AsyncGenerator[str, None]:
        """Create a message and stream the response with conversation history."""
        chat = self.get_chat(chat_id)
        response_content = ""

        try:
            # Get existing messages for the chat
            existing_messages = self.db.query(Message) \
                .filter(Message.chat_id == chat_id) \
                .order_by(Message.timestamp.asc()) \
                .all()

            # Format conversation history
            conversation_history = self._format_chat_history(
                existing_messages,
                max_history_messages
            )

            # Determine model name
            model_name = model_config.get("model")
            if not model_name:
                raise ValueError(f"No model name provided for provider {model_config['provider']}")

            # Create user message
            user_message = Message(
                id=str(uuid.uuid4()),
                chat_id=chat_id,
                role="user",
                content=content,
                timestamp=datetime.utcnow(),
                model_provider=model_config["provider"],
                model_name=model_name,
                temperature=model_config["temperature"]
            )
            self.db.add(user_message)
            self.db.commit()

            # Initialize appropriate LLM service
            llm_service = await create_llm_service(
                provider=model_config["provider"],
                api_key=model_config.get("apiKey"),
                model=model_name
            )

            # Prepare LLM config
            llm_config = LLMConfig(
                model=model_name,
                temperature=model_config["temperature"],
                top_p=1.0,
                stop_sequences=[],
                extra_params={}
            )

            # Stream response from LLM with conversation history
            async for chunk in llm_service.generate_stream(
                    content,
                    llm_config,
                    context=conversation_history
            ):
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
                model_name=model_name,
                temperature=model_config["temperature"]
            )
            self.db.add(assistant_message)
            self.db.commit()

        except Exception as e:
            print(f"Error in message generation: {str(e)}")
            # If there's an error, we should still save what we have
            if response_content:
                assistant_message = Message(
                    id=str(uuid.uuid4()),
                    chat_id=chat_id,
                    role="assistant",
                    content=response_content,
                    timestamp=datetime.utcnow(),
                    model_provider=model_config["provider"],
                    model_name=model_name,
                    temperature=model_config["temperature"]
                )
                self.db.add(assistant_message)
                self.db.commit()
            raise HTTPException(status_code=500, detail=str(e))
