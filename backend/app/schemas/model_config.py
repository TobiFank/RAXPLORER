# app/schemas/model_config.py
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from datetime import datetime

from app.utils.case_utils import to_camel


class ModelConfigBase(BaseModel):
    provider: str = Field(..., pattern="^(claude|chatgpt|ollama)$")
    model: str
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    ollama_model: Optional[str] = None  # Use snake_case

    class Config:
        alias_generator = to_camel
        populate_by_name = True

class ModelConfigCreate(ModelConfigBase):
    api_key: Optional[str] = None

    @validator('api_key')
    def validate_api_key(cls, v, values):
        provider = values.get('provider')
        if provider in ['claude', 'chatgpt'] and not v:
            raise ValueError(f"{provider} requires an API key")
        return v

    @validator('model', pre=True)
    def validate_model(cls, v, values):
        provider = values.get('provider')
        if provider == 'ollama':
            # For Ollama, use ollama_model if model is empty
            return values.get('ollama_model', '') or v
        return v

class ModelConfigResponse(ModelConfigBase):
    id: str
    updated_at: datetime

    class Config:
        from_attributes = True