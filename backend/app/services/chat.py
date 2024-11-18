# app/services/chat.py
from datetime import datetime
from typing import AsyncGenerator
from fastapi import HTTPException
from ..schemas.chat import Chat, Message
from ..schemas.model import ModelConfig
from ..db.models import ChatModel, FileModel
from ..db.session import AsyncSession
from .llm import LLMService
from .rag import RAGService

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
        chats = await self.db.query(ChatModel).order_by(ChatModel.created_at.desc()).all()
        return [
            Chat(
                id=chat.id,
                title=chat.title,
                messages=chat.messages,
                created_at=chat.created_at
            ) for chat in chats
        ]

    async def get_chat(self, chat_id: str) -> Chat:
        chat = await self.db.query(ChatModel).get(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        return Chat(
            id=chat.id,
            title=chat.title,
            messages=chat.messages,
            created_at=chat.created_at
        )

    async def delete_chat(self, chat_id: str):
        chat = await self.db.query(ChatModel).get(chat_id)
        if chat:
            await self.db.delete(chat)
            await self.db.commit()

    async def update_title(self, chat_id: str, title: str) -> Chat:
        chat = await self.db.query(ChatModel).get(chat_id)
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

    async def stream_response(
            self,
            chat_id: str,
            content: str,
            model_config: ModelConfig
    ) -> AsyncGenerator[str, None]:
        chat = await self.db.query(ChatModel).get(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")

        # Get relevant docs for RAG
        files = await self.db.query(FileModel).all()
        relevant_chunks = await self.rag.query(
            content,
            [f.vector_store_id for f in files]
        )

        # Prepare messages with system context from RAG
        context = "\n\n".join([chunk.extra_info + "\n" + chunk.text for chunk in relevant_chunks])
        messages = [
            {"role": "system", "content": f"{model_config.system_message or ''}\n\nContext:\n{context}"},
            *[{"role": m["role"], "content": m["content"]} for m in chat.messages],
            {"role": "user", "content": content}
        ]

        # Get LLM provider and generate response
        provider = await self.llm.get_provider(model_config)
        response = ""
        async for chunk in provider.generate(messages, model_config):
            response += chunk
            yield chunk

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