# app/services/llm.py
from typing import Protocol, AsyncGenerator
from ..schemas.model import ModelConfig, Provider
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
import httpx

class LLMProvider(Protocol):
    async def generate(self, messages: list[dict], config: ModelConfig) -> AsyncGenerator[str, None]: ...
    async def validate_config(self, config: ModelConfig) -> dict: ...

class ClaudeProvider:
    async def generate(self, messages: list[dict], config: ModelConfig) -> AsyncGenerator[str, None]:
        client = AsyncAnthropic(api_key=config.api_key)
        system = config.system_message or ""

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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": config.model,
                    "messages": messages,
                    "stream": True
                },
                stream=True
            )
            async for line in response.aiter_lines():
                if "response" in line:
                    yield line["response"]

class LLMService:
    _providers = {
        Provider.CLAUDE: ClaudeProvider(),
        Provider.CHATGPT: ChatGPTProvider(),
        Provider.OLLAMA: OllamaProvider()
    }

    @classmethod
    async def get_provider(cls, config: ModelConfig) -> LLMProvider:
        return cls._providers[config.provider]