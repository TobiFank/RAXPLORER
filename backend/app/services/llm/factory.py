# app/services/llm/factory.py
from typing import Optional
from app.services.llm.base import BaseLLMService
from app.services.llm.ollama import OllamaService
from app.services.llm.claude import ClaudeService
from app.services.llm.chatgpt import ChatGPTService
from app.core.config import settings

async def create_llm_service(
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None
) -> BaseLLMService:
    """Create and initialize appropriate LLM service based on provider"""
    if provider == "ollama":
        service = OllamaService(base_url=settings.OLLAMA_HOST)
    elif provider == "claude":
        service = ClaudeService(api_key or settings.ANTHROPIC_API_KEY)
    elif provider == "chatgpt":
        service = ChatGPTService(api_key or settings.OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    await service.initialize()
    return service