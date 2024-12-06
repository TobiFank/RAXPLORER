# app/services/llm.py
import json
import logging
from typing import Protocol, AsyncGenerator, List

import httpx
from anthropic import AsyncAnthropic
from fastapi import HTTPException
from openai import AsyncOpenAI

from ..core.config import Settings
from ..schemas.model import ModelConfig, Provider

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    async def generate(self, messages: list[dict], config: ModelConfig) -> AsyncGenerator[str, None]: ...

    async def validate_config(self, config: ModelConfig) -> dict: ...

    async def get_embeddings(self, text: str, config: ModelConfig) -> list[float]: ...

    async def get_embeddings_batch(self, all_queries, model_config):
        pass


class ClaudeProvider:
    async def generate(self, messages: list[dict], config: ModelConfig):
        client = AsyncAnthropic(api_key=config.apiKey)
        system = config.systemMessage or ""
        message_stream = await client.messages.create(
            model=config.model,
            messages=messages,
            system=system,
            temperature=config.temperature,
            max_tokens=4096,
            stream=True
        )
        async for chunk in message_stream:
            if hasattr(chunk, 'content'):
                yield chunk.content[0].text

    async def get_embeddings(self, text: str, config: ModelConfig) -> list[float]:
        client = AsyncAnthropic(api_key=config.apiKey)
        response = await client.embeddings.create(
            model="claude-3-embedding-v1",
            input=text
        )
        return response.embeddings[0]


class ChatGPTProvider:
    async def generate(self, messages: list[dict], config: ModelConfig):
        client = AsyncOpenAI(api_key=config.apiKey)
        response = await client.chat.completions.create(model=config.model, messages=messages,
                                                        temperature=config.temperature, stream=True)
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def get_embeddings(self, text: str, config: ModelConfig) -> list[float]:
        client = AsyncOpenAI(api_key=config.apiKey)
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding


class OllamaProvider:
    async def _ensure_model_exists(self, model_name: str, base_url: str) -> None:
        """Ensure model exists, pull if it doesn't"""
        try:
            async with httpx.AsyncClient() as client:
                # Pull the model
                pull_response = await client.post(
                    f"{base_url}/api/pull",
                    json={"model": model_name, "stream": False},
                    timeout=None
                )

                if not 200 <= pull_response.status_code < 300:
                    raise HTTPException(
                        status_code=pull_response.status_code,
                        detail=f"Failed to pull model {model_name}: {pull_response.text}"
                    )

                logger.info(f"Successfully pulled model {model_name}")

                # Verify model was pulled successfully
                verify_response = await client.post(
                    f"{base_url}/api/chat",
                    json={"model": model_name, "messages": []},
                    timeout=None
                )

                if not 200 <= verify_response.status_code < 300:
                    raise HTTPException(
                        status_code=verify_response.status_code,
                        detail=f"Model verification failed after pulling: {verify_response.text}"
                    )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error ensuring model exists: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _ensure_embedding_model_exists(self, model_name: str, base_url: str) -> None:
        """Ensure embedding model exists, pull if it doesn't"""
        try:
            async with httpx.AsyncClient() as client:
                # Pull the model
                pull_response = await client.post(
                    f"{base_url}/api/pull",
                    json={"model": model_name, "stream": False},
                    timeout=None
                )

                if not 200 <= pull_response.status_code < 300:
                    raise HTTPException(
                        status_code=pull_response.status_code,
                        detail=f"Failed to pull model {model_name}: {pull_response.text}"
                    )

                logger.info(f"Successfully pulled model {model_name}")

                # Verify model with embeddings endpoint
                verify_response = await client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model_name, "prompt": "test"},
                    timeout=None
                )

                if not 200 <= verify_response.status_code < 300:
                    raise HTTPException(
                        status_code=verify_response.status_code,
                        detail=f"Model verification failed after pulling: {verify_response.text}"
                    )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error ensuring embedding model exists: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def generate(self, messages: list[dict], config: ModelConfig):
        settings = Settings()
        base_url = settings.OLLAMA_HOST or "http://ollama:11434"

        logger.debug(f"Sending formatted messages to Ollama: {messages}")

        # Ensure model exists before trying to use it
        await self._ensure_model_exists(config.model, base_url)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": config.model,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "num_ctx": 8192
                        }
                    },
                    timeout=None
                )

                if not response.status_code == 200:
                    logger.error(f"Ollama error: {response.status_code} - {response.text}")
                    raise HTTPException(status_code=response.status_code, detail=f"Ollama error: {response.text}")

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

    async def get_embeddings(self, text: str, config: ModelConfig) -> list[float]:
        settings = Settings()
        base_url = settings.OLLAMA_HOST or "http://ollama:11434"

        # Ensure embedding model exists
        await self._ensure_embedding_model_exists(settings.OLLAMA_EMBEDDING_MODEL, base_url)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/api/embeddings",
                    json={
                        "model": settings.OLLAMA_EMBEDDING_MODEL,
                        "prompt": text
                    },
                    timeout=None
                )

                if not response.status_code == 200:
                    logger.error(f"Ollama embeddings error: {response.status_code} - {response.text}")
                    raise HTTPException(status_code=response.status_code, detail=f"Ollama error: {response.text}")

                data = response.json()
                return data["embedding"]

        except Exception as e:
            logger.error(f"Error getting embeddings from Ollama: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_embeddings_batch(self, texts: List[str], config: ModelConfig) -> List[List[float]]:
        settings = Settings()
        base_url = settings.OLLAMA_HOST or "http://ollama:11434"

        # Load model only once
        await self._ensure_embedding_model_exists(settings.OLLAMA_EMBEDDING_MODEL, base_url)

        # Use a single client for all requests to reuse connection
        async with httpx.AsyncClient() as client:
            embeddings = []
            for text in texts:
                try:
                    response = await client.post(
                        f"{base_url}/api/embeddings",
                        json={
                            "model": settings.OLLAMA_EMBEDDING_MODEL,
                            "prompt": text
                        },
                        timeout=None
                    )

                    if response.status_code != 200:
                        logger.error(f"Ollama embeddings error: {response.status_code} - {response.text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Ollama error: {response.text}"
                        )

                    data = response.json()
                    embeddings.append(data["embedding"])

                except Exception as e:
                    logger.error(f"Error getting embedding for text: {str(e)}")
                    raise HTTPException(status_code=500, detail=str(e))

            return embeddings


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
