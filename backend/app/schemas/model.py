# app/schemas/model.py
from enum import Enum
from pydantic import BaseModel, Field

class Provider(str, Enum):
    CLAUDE = "claude"
    CHATGPT = "chatgpt"
    OLLAMA = "ollama"

class ModelConfig(BaseModel):
    provider: Provider
    model: str
    temperature: float = Field(ge=0, le=2)
    system_message: str | None
    api_key: str | None = None