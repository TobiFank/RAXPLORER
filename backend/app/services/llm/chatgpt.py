# app/services/llm/chatgpt.py
import json
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

class ChatGPTService(BaseLLMService):
    """Implementation of BaseLLMService for OpenAI's ChatGPT API"""

    def __init__(
            self,
            api_key: Optional[str] = None,
            base_url: str = "https://api.openai.com/v1",
            timeout: float = 30.0,
            max_retries: int = 3
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the ChatGPT service and verify connection."""
        if self._initialized:
            return

        if not self.api_key:
            raise ConnectionError("OpenAI API key not provided")

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )

        try:
            await self.health_check()
            self._initialized = True
        except Exception as e:
            await self._client.aclose()
            raise ConnectionError(f"Failed to initialize ChatGPT service: {str(e)}")

    async def health_check(self) -> bool:
        """Verify API key and connection by making a minimal request."""
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(f"ChatGPT health check failed: {str(e)}")

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
            messages = []
            if config.system_message:
                messages.append({
                    "role": "system",
                    "content": config.system_message
                })
            messages.append({
                "role": "user",
                "content": await self.format_prompt(prompt, context)
            })

            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": config.model,
                    "messages": messages,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "top_p": config.top_p,
                    "stop": config.stop_sequences,
                    **config.extra_params
                }
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=config.model,
                usage=TokenUsage(
                    prompt_tokens=data["usage"]["prompt_tokens"],
                    completion_tokens=data["usage"]["completion_tokens"],
                    total_tokens=data["usage"]["total_tokens"]
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
        if not self._initialized:
            await self.initialize()

        try:
            messages = []
            if config.system_message:
                messages.append({
                    "role": "system",
                    "content": config.system_message
                })
            messages.append({
                "role": "user",
                "content": await self.format_prompt(prompt, context)
            })

            async with self._client.stream(
                    "POST",
                    "/chat/completions",
                    json={
                        "model": config.model,
                        "messages": messages,  # Use the messages array with system message
                        "temperature": config.temperature,
                        "stream": True,
                        **config.extra_params
                    }
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    try:
                        if line.startswith("data: "):
                            json_data = json.loads(line[6:])  # Remove "data: " prefix and parse JSON
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                delta = json_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                    except Exception as e:
                        print(f"Error processing chunk: {str(e)}")
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
                    "model": "text-embedding-3-large",  # OpenAI's latest embedding model
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
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