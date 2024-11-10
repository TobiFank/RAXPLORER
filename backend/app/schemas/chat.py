# app/schemas/chat.py
from datetime import datetime
from typing import List, Optional

from app.utils.case_utils import to_camel
from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    provider: str = Field(..., pattern="^(claude|chatgpt|ollama)$")
    api_key: Optional[str] = None
    model: str = ""
    ollama_model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=1.0)  # Add this line

    class Config:
        from_attributes = True
        alias_generator = to_camel
        populate_by_name = True


class MessageBase(BaseModel):
    content: str


class MessageCreate(MessageBase):
    modelConfig: ModelSettings  # Changed from model_config to match frontend

    class Config:
        populate_by_name = True
        alias_generator = None


class MessageResponse(MessageBase):
    id: str
    role: str
    timestamp: datetime

    class Config:
        from_attributes = True


class ChatBase(BaseModel):
    title: str = "New Chat"


class ChatCreate(ChatBase):
    title: Optional[str] = "New Chat"


class ChatUpdate(ChatBase):
    pass


class ChatResponse(ChatBase):
    id: str
    created_at: datetime
    messages: List[MessageResponse]

    class Config:
        from_attributes = True


class ChatListResponse(ChatBase):
    id: str
    created_at: datetime
    last_message: Optional[str] = None
    message_count: int

    class Config:
        from_attributes = True
