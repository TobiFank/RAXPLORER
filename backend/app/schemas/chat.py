# app/schemas/chat.py
from datetime import datetime

from pydantic import BaseModel

from .model import ModelConfig


class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime


class Chat(BaseModel):
    id: str
    title: str
    messages: list[Message]
    created_at: datetime


class MessageRequest(Message):
    modelConfig: ModelConfig
    timestamp: datetime | None = None
    role: str = "user"
