# app/services/llm.py
import json
from typing import Protocol, AsyncGenerator

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from ..core.config import Settings
from ..schemas.model import ModelConfig, Provider


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

        print("Ollama request payload:", {
            "model": config.model,
            "messages": messages
        })

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": config.model,
                    "prompt": messages[-1]["content"],  # Get the last user message
                    "system": messages[0]["content"] if messages[0]["role"] == "system" else "",
                    "stream": True
                },
                timeout=None
            )
            print("Ollama raw response status:", response.status_code)

            # Keep track if we've seen actual content
            has_yielded_content = False

            async for line in response.aiter_lines():
                try:
                    print("Ollama response line:", line)
                    data = json.loads(line)

                    # Skip loading messages
                    if data.get("done_reason") == "load":
                        continue

                    if "response" in data and data["response"]:
                        has_yielded_content = True
                        print("Yielding response:", data["response"])
                        yield data["response"]

                except json.JSONDecodeError:
                    print("Failed to decode JSON line:", line)
                    continue

            # If we never yielded any content, try again
            if not has_yielded_content:
                print("No content yielded, retrying request...")
                # The model should be loaded now, so retry once
                async for chunk in self.generate(messages, config):
                    yield chunk


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
