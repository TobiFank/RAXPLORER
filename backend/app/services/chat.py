# app/services/chat.py
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.sql import select

from .llm import LLMService
from .rag import RAGService
from ..db.models import ChatModel, FileModel
from ..db.session import AsyncSession
from ..schemas.chat import Chat
from ..schemas.model import ModelConfig


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
            # Remove the redundant chat query and start directly with files query
            result = await self.db.execute(select(FileModel))
            files = result.scalars().all()
            relevant_chunks = await self.rag.query(
                content,
                [f.vector_store_id for f in files]
            )

            # Prepare messages with system context from RAG
            context = "\n\n".join([chunk.text for chunk in relevant_chunks])
            messages = [
                {"role": "system", "content": f"{model_config.systemMessage or ''}\n\nContext:\n{context}"},
                {"role": "user", "content": content}
            ]

            # Get LLM provider and generate response
            provider = await self.llm.get_provider(model_config)
            response = ""
            print("Starting to stream response")  # Add this logging
            async for chunk in provider.generate(messages, model_config):
                print("Received chunk:", chunk)  # Add this logging
                response += chunk
                yield chunk

            print("Final response:", response)  # Add this logging

            # Update chat history
            chat.messages.append({
                "role": "user",
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            })
            chat.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.utcnow().isoformat()
            })
            await self.db.commit()
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
