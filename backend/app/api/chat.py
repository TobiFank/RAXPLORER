# app/api/chat.py
from fastapi import APIRouter, HTTPException
from typing import AsyncGenerator
from sse_starlette.sse import EventSourceResponse

from ..schemas.chat import Chat, Message
from ..schemas.model import ModelConfig
from ..services.chat import ChatService

router = APIRouter(prefix="/chat")

@router.post("/chats")
async def create_chat() -> Chat:
    return await ChatService.create_chat()

@router.get("/chats")
async def get_chats() -> list[Chat]:
    return await ChatService.get_chats()

@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str) -> Chat:
    return await ChatService.get_chat(chat_id)

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    await ChatService.delete_chat(chat_id)

@router.patch("/chats/{chat_id}")
async def update_chat_title(chat_id: str, title: str) -> Chat:
    return await ChatService.update_title(chat_id, title)

@router.post("/chats/{chat_id}/messages")
async def send_message(chat_id: str, content: str, model_config: ModelConfig) -> EventSourceResponse:
    return EventSourceResponse(ChatService.stream_response(chat_id, content, model_config))