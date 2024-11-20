# app/services/llm.py
import json
from typing import Protocol, AsyncGenerator

import httpx
from anthropic import AsyncAnthropic
from fastapi import HTTPException
from openai import AsyncOpenAI

from ..core.config import Settings
from ..schemas.model import ModelConfig, Provider

import logging

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    async def generate(self, messages: list[dict], config: ModelConfig) -> AsyncGenerator[str, None]: ...

    async def validate_config(self, config: ModelConfig) -> dict: ...


class ClaudeProvider:
    async def generate(self, messages: list[dict], config: ModelConfig):
        client = AsyncAnthropic(api_key=config.apiKey)
        system = config.systemMessage or ""

        response = await client.messages.stream(
            model=config.model,
            messages=messages,
            system=system,
            temperature=config.temperature
        )

        async for chunk in response:
            if chunk.delta.text:
                yield chunk.delta.text


class ChatGPTProvider:
    async def generate(self, messages: list[dict], config: ModelConfig) -> AsyncGenerator[str, None]:
        client = AsyncOpenAI(api_key=config.api_key)
        response = await client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            stream=True
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider:
    async def generate(self, messages: list[dict], config: ModelConfig) -> AsyncGenerator[str, None]:
        settings = Settings()
        base_url = settings.OLLAMA_HOST or "http://ollama:11434"

        # Ensure messages are in correct format for Ollama
        formatted_messages = []
        for msg in messages:
            if "role" in msg and "content" in msg:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        logger.info(f"Sending formatted messages to Ollama: {formatted_messages}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": config.model,
                        "messages": formatted_messages,
                        "stream": True
                    },
                    timeout=None
                )

                if not response.status_code == 200:
                    logger.error(f"Ollama error: {response.status_code} - {response.text}")
                    raise HTTPException(status_code=response.status_code,
                                        detail=f"Ollama error: {response.text}")

                async for line in response.aiter_lines():
                    try:
                        data = json.loads(line)
                        if "message" in data and data["message"].get("content"):
                            yield data["message"]["content"]
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode Ollama response: {line} - {str(e)}")
                        continue

            except Exception as e:
                logger.error(f"Error communicating with Ollama: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))


class LLMService:
    _providers = {
        Provider.CLAUDE: ClaudeProvider(),
        Provider.CHATGPT: ChatGPTProvider(),
        Provider.OLLAMA: OllamaProvider()
    }

    @classmethod
    async def get_provider(cls, config: ModelConfig) -> LLMProvider:
        return cls._providers[config.provider]

    # Add these new methods below:
    @classmethod
    async def validate_config(cls, config: ModelConfig) -> dict:
        try:
            provider = cls._providers[config.provider]
            return {"valid": True}
        except Exception as e:
            return {"valid": False, "issues": [str(e)]}
