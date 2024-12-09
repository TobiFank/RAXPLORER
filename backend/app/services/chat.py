# app/services/chat.py
import json
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import update
from sqlalchemy.sql import select

from .llm import LLMService
from .rag.rag import RAGService
from ..db.models import ChatModel, FileModel
from ..db.session import AsyncSession
from ..schemas.chat import Chat
from ..schemas.model import ModelConfig

import logging

from ..schemas.rag import RAGResponse

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: AsyncSession, llm_service: LLMService, rag_service: RAGService):
        self.db = db
        self.llm = llm_service
        self.rag = rag_service

    async def create_chat(self) -> Chat:
        chat = ChatModel(title="New Chat")
        self.db.add(chat)
        await self.db.commit()
        return Chat(
            id=chat.id,
            title=chat.title,
            messages=[],
            created_at=chat.created_at
        )

    async def get_chats(self) -> list[Chat]:
        result = await self.db.execute(select(ChatModel).order_by(ChatModel.created_at.desc()))
        chats = result.scalars().all()
        return [
            Chat(
                id=chat.id,
                title=chat.title,
                messages=chat.messages,
                created_at=chat.created_at
            ) for chat in chats
        ]

    async def get_chat(self, chat_id: str) -> Chat:
        result = await self.db.execute(select(ChatModel).filter(ChatModel.id == chat_id))
        chat = result.scalar_one_or_none()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        return Chat(
            id=chat.id,
            title=chat.title,
            messages=chat.messages,
            created_at=chat.created_at
        )

    async def delete_chat(self, chat_id: str):
        result = await self.db.execute(select(ChatModel).filter(ChatModel.id == chat_id))
        chat = result.scalar_one_or_none()
        if chat:
            await self.db.delete(chat)
            await self.db.commit()

    async def update_title(self, chat_id: str, title: str) -> Chat:
        result = await self.db.execute(select(ChatModel).filter(ChatModel.id == chat_id))
        chat = result.scalar_one_or_none()
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        chat.title = title
        await self.db.commit()

        return Chat(
            id=chat.id,
            title=chat.title,
            messages=chat.messages,
            created_at=chat.created_at
        )

    async def stream_response(self, chat_id: str, content: str, model_config: ModelConfig):
        if not chat_id:
            raise HTTPException(status_code=400, detail="chat_id is required")

        result = await self.db.execute(select(ChatModel).filter(ChatModel.id == chat_id))
        chat = result.scalar_one_or_none()
        if not chat:
            raise HTTPException(status_code=404, detail=f"Chat {chat_id} not found")

        try:
            # Get all available files for RAG
            result = await self.db.execute(select(FileModel))
            files = result.scalars().all()

            # Get RAG response
            rag_response: RAGResponse = await self.rag.query(
                content,
                [f.id for f in files],
                model_config,
                chat.messages
            )

            # Store user message
            chat.messages.append({
                "role": "user",
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            })

            formatted_response = rag_response.answer
            logger.debug(f"RAG response images: {rag_response.images}")
            for idx, img in enumerate(rag_response.images, 1):
                formatted_response = formatted_response.replace(f"[Bild {img.image_id}]", f"Abbildung {idx}")
                formatted_response += f"\n[IMAGE:{img.file_path}|Abbildung {idx}: {img.caption}|{img.image_id}]\n"
            logger.debug(f"Formatted response after image processing: {formatted_response}")

            # Store assistant message with enhanced metadata
            chat.messages.append({
                "role": "assistant",
                "content": formatted_response,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "citations": [citation.dict() for citation in rag_response.citations],
                    "images": [image.dict() for image in rag_response.images],
                    "reasoning": rag_response.reasoning,
                    "confidence_score": rag_response.confidence_score
                }
            })

            # Update the chat in database
            await self.db.execute(
                update(ChatModel)
                .where(ChatModel.id == chat.id)
                .values(messages=chat.messages)
            )
            await self.db.commit()

            logger.debug(f"Response: {formatted_response}")

            yield formatted_response

        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
