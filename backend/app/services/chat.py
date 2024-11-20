# app/services/chat.py
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import update
from sqlalchemy.sql import select

from .llm import LLMService
from .rag import RAGService
from ..db.models import ChatModel, FileModel
from ..db.session import AsyncSession
from ..schemas.chat import Chat
from ..schemas.model import ModelConfig

import logging

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

        logger.info(f"Chat content - id: {chat.id}, title: {chat.title}, messages: {chat.messages}")

        if chat.messages is None:
            chat.messages = []

        try:
            # Enhance the query first
            enhanced_query = await self.enhance_query(content, chat.messages, model_config)

            # Query files for RAG context using enhanced query
            result = await self.db.execute(select(FileModel))
            files = result.scalars().all()
            relevant_chunks = await self.rag.query(
                enhanced_query,
                [f.vector_store_id for f in files],
                model_config,
                self.db
            )
            context = "\n\n".join([chunk.text for chunk in relevant_chunks])

            # Build message history
            messages = []
            if model_config.systemMessage:
                messages.append({
                    "role": "system",
                    "content": model_config.systemMessage
                })

            # Add existing chat history
            messages.extend([{
                "role": msg["role"],
                "content": msg["content"]
            } for msg in chat.messages if "role" in msg and "content" in msg])

            # Create enhanced user message with context for LLM
            enhanced_message = f"Given the chat history and this context:\n{context}\n\nAnswer this User Query: {content}"
            messages.append({
                "role": "user",
                "content": enhanced_message
            })

            logger.info(f"Messages to send to LLM: {messages}")

            # Get LLM provider and generate response
            provider = await self.llm.get_provider(model_config)
            response = ""
            async for chunk in provider.generate(messages, model_config):
                response += chunk
                yield chunk

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
            # Tell SQLAlchemy the field was modified
            await self.db.execute(
                update(ChatModel)
                .where(ChatModel.id == chat.id)
                .values(messages=chat.messages)
            )
            await self.db.commit()

        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail=str(e))

    async def enhance_query(self, query: str, chat_messages: list, model_config: ModelConfig) -> str:
        """Enhance the user query using chat history and other context to improve RAG results"""

        # Create a prompt that helps the model understand what we want
        enhancement_prompt = {
            "role": "system",
            "content": """You are a query enhancement specialist. Your task is to rephrase and expand the given query to improve search results. Consider:
    1. Recent chat context to understand the full conversation flow
    2. Add relevant synonyms or related terms
    3. Make implicit subjects explicit
    4. Include contextual information from recent messages
    5. Break compound questions into their core concepts
    
    Keep the enhanced query focused and relevant. Don't add speculative content.
    Output only the enhanced query, nothing else."""
        }

        # Get recent chat context (last 3 messages for context)
        recent_context = chat_messages[-3:] if chat_messages else []
        context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_context])

        user_prompt = {
            "role": "user",
            "content": f"""Chat History:
    {context_str}
    
    Original Query: {query}
    
    Enhance this query for better search results."""
        }

        # Get LLM provider and generate enhanced query
        provider = await self.llm.get_provider(model_config)
        enhanced_query = ""
        async for chunk in provider.generate([enhancement_prompt, user_prompt], model_config):
            enhanced_query += chunk

        logger.info(f"Original query: {query}")
        logger.info(f"Enhanced query: {enhanced_query}")

        return enhanced_query.strip()
