# app/schemas/model_config.py
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from datetime import datetime

class ModelConfigBase(BaseModel):
    provider: str = Field(..., pattern="^(claude|chatgpt|ollama)$")
    model: str
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    extra_params: Dict[str, Any] = Field(default_factory=dict)

class ModelConfigCreate(ModelConfigBase):
    api_key: Optional[str] = None

    @validator('api_key')
    def validate_api_key(cls, v, values):
        provider = values.get('provider')
        if provider in ['claude', 'chatgpt'] and not v:
            raise ValueError(f"{provider} requires an API key")
        return v

class ModelConfigResponse(ModelConfigBase):
    id: str
    updated_at: datetime

    class Config:
        from_attributes = True