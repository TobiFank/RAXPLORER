# app/services/llm/ollama.py
import asyncio
import httpx
import json
from typing import AsyncGenerator, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from .base import (
    BaseLLMService, LLMConfig, LLMResponse, TokenUsage,
    ConnectionError, GenerationError
)

class OllamaService(BaseLLMService):
    def __init__(
            self,
            base_url: Optional[str] = None,
            timeout: float = 30.0,
            max_retries: int = 3
    ):
        self.base_url = base_url or settings.OLLAMA_HOST
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Ollama service and verify connection."""
        if self._initialized:
            return

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout
        )

        # Check connection and pull model if needed
        try:
            await self.health_check()
            self._initialized = True
        except Exception as e:
            await self._client.aclose()
            raise ConnectionError(f"Failed to initialize Ollama service: {str(e)}")

    async def health_check(self) -> bool:
        """Check if Ollama is responding and models are available."""
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(f"Ollama health check failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: None
    )
    async def generate(
            self,
            prompt: str,
            config: LLMConfig,
            context: Optional[str] = None
    ) -> LLMResponse:
        """Generate a complete response with retries."""
        if not self._initialized:
            await self.initialize()

        try:
            formatted_prompt = await self.format_prompt(prompt, context)
            system_prompt = config.system_message if config.system_message else ""

            response = await self._client.post(
                "/api/generate",
                json={
                    "model": config.model,
                    "prompt": formatted_prompt,
                    "system": system_prompt,  # Ollama supports system prompts this way
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "stop": config.stop_sequences,
                    **config.extra_params,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data["response"],
                model=config.model,
                usage=TokenUsage(
                    total_tokens=len(data["response"].split())  # Approximate
                ),
                raw_response=data
            )
        except Exception as e:
            raise GenerationError(f"Failed to generate response: {str(e)}")

    async def generate_stream(
            self,
            prompt: str,
            config: LLMConfig,
            context: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response with error handling."""
        if not self._initialized:
            await self.initialize()

        try:
            formatted_prompt = await self.format_prompt(prompt, context)
            async with self._client.stream(
                    "POST",
                    "/api/generate",
                    json={
                        "model": config.model,
                        "prompt": formatted_prompt,
                        "temperature": config.temperature,
                        "stream": True,
                        **config.extra_params
                    }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        raise GenerationError(f"Stream processing error: {str(e)}")
        except Exception as e:
            raise GenerationError(f"Stream generation failed: {str(e)}")

    async def get_embedding(
            self,
            text: str,
            config: Optional[LLMConfig] = None
    ) -> list[float]:
        """Generate embeddings with error handling."""
        if not self._initialized:
            await self.initialize()

        try:
            response = await self._client.post(
                "/api/embeddings",
                json={
                    "model": config.model if config else "llama2",
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except Exception as e:
            raise GenerationError(f"Failed to generate embedding: {str(e)}")

    async def __aenter__(self):
        """Support async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        if self._client:
            await self._client.aclose()