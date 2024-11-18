
# app/schemas/chat.py
from datetime import datetime
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime

class Chat(BaseModel):
    id: str
    title: str
    messages: list[Message]
    created_at: datetime