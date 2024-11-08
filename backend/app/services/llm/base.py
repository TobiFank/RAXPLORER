# app/services/llm/base.py
from abc import ABC, abstractmethod
from enum import Enum
from typing import AsyncGenerator, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class ConnectionError(LLMError):
    """Raised when connection to LLM service fails."""
    pass

class GenerationError(LLMError):
    """Raised when LLM fails to generate response."""
    pass

class ResponseFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"

class LLMConfig(BaseModel):
    model: str
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = None
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    stop_sequences: Optional[list[str]] = None
    response_format: ResponseFormat = ResponseFormat.TEXT
    extra_params: Dict[str, Any] = Field(default_factory=dict)

class TokenUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class LLMResponse(BaseModel):
    content: str
    model: str
    usage: Optional[TokenUsage] = None
    raw_response: Optional[Dict[str, Any]] = None

class BaseLLMService(ABC):
    """Base interface for all LLM implementations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service, load models, etc."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the service is available and responding."""
        pass

    @abstractmethod
    async def generate(
            self,
            prompt: str,
            config: LLMConfig,
            context: Optional[str] = None
    ) -> LLMResponse:
        """Generate a complete response."""
        pass

    @abstractmethod
    async def generate_stream(
            self,
            prompt: str,
            config: LLMConfig,
            context: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        pass

    @abstractmethod
    async def get_embedding(
            self,
            text: str,
            config: Optional[LLMConfig] = None
    ) -> list[float]:
        """Generate embeddings for RAG."""
        pass

    async def format_prompt(
            self,
            prompt: str,
            context: Optional[str] = None,
            system_prompt: Optional[str] = None
    ) -> str:
        """Format prompt with context and system prompt."""
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        if context:
            parts.append(f"Context: {context}")
        parts.append(f"User: {prompt}")
        return "\n\n".join(parts)
