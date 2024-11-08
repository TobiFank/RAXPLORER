# app/schemas/chat.py
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from uuid import UUID

class MessageBase(BaseModel):
    content: str

class ModelSettings(BaseModel):
    provider: str = Field(..., pattern="^(claude|chatgpt|ollama)$")
    api_key: Optional[str] = None
    model: str
    ollama_model: Optional[str] = None
    temperature: float = Field(ge=0.0, le=1.0)

class MessageCreate(MessageBase):
    modelConfig: ModelSettings  # Changed from model_config to ai_config

class MessageResponse(MessageBase):
    id: str
    role: str
    timestamp: datetime

    class Config:
        from_attributes = True

class ChatBase(BaseModel):
    title: str = "New Chat"

class ChatCreate(ChatBase):
    pass

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