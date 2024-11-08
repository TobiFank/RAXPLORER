# app/services/llm/claude.py
from typing import AsyncGenerator, Optional, Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from .base import (
    BaseLLMService,
    LLMConfig,
    LLMResponse,
    TokenUsage,
    ConnectionError,
    GenerationError
)

class ClaudeService(BaseLLMService):
    """Implementation of BaseLLMService for Anthropic's Claude API"""

    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: str = "https://api.anthropic.com/v1",
            timeout: float = 30.0,
            max_retries: int = 3
    ):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Claude service and verify connection."""
        if self._initialized:
            return

        if not self.api_key:
            raise ConnectionError("Claude API key not provided")

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
        )

        try:
            await self.health_check()
            self._initialized = True
        except Exception as e:
            await self._client.aclose()
            raise ConnectionError(f"Failed to initialize Claude service: {str(e)}")

    async def health_check(self) -> bool:
        """Verify API key and connection by making a minimal request."""
        try:
            response = await self._client.post(
                "/messages",
                json={
                    "model": "claude-3-opus-20240229",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Hi"}]
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(f"Claude health check failed: {str(e)}")

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
            response = await self._client.post(
                "/messages",
                json={
                    "model": config.model,
                    "messages": [{"role": "user", "content": formatted_prompt}],
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "top_p": config.top_p,
                    "stop_sequences": config.stop_sequences,
                    **config.extra_params
                }
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data["content"][0]["text"],
                model=config.model,
                usage=TokenUsage(
                    prompt_tokens=data.get("usage", {}).get("input_tokens"),
                    completion_tokens=data.get("usage", {}).get("output_tokens"),
                    total_tokens=data.get("usage", {}).get("total_tokens")
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
        """Generate a streaming response."""
        if not self._initialized:
            await self.initialize()

        try:
            formatted_prompt = await self.format_prompt(prompt, context)
            async with self._client.stream(
                    "POST",
                    "/messages",
                    json={
                        "model": config.model,
                        "messages": [{"role": "user", "content": formatted_prompt}],
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                        "stream": True,
                        **config.extra_params
                    }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or line.startswith("data: [DONE]"):
                        continue
                    try:
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            chunk = data.get("content")[0].get("text", "")
                            if chunk:
                                yield chunk
                    except Exception:
                        continue

        except Exception as e:
            raise GenerationError(f"Stream generation failed: {str(e)}")

    async def get_embedding(
            self,
            text: str,
            config: Optional[LLMConfig] = None
    ) -> list[float]:
        """Generate embeddings for RAG."""
        if not self._initialized:
            await self.initialize()

        try:
            response = await self._client.post(
                "/embeddings",
                json={
                    "model": "claude-3-opus-20240229",  # Claude's embedding model
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
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