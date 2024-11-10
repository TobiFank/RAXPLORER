# app/services/llm/claude.py
import json
from typing import AsyncGenerator, Optional

import httpx
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
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
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
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 1000,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hi"
                        }
                    ]
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            raise ConnectionError(f"Claude health check failed: {str(e)}")

    async def generate(
            self,
            prompt: str,
            config: LLMConfig,
            context: Optional[str] = None
    ) -> LLMResponse:
        if not self._initialized:
            await self.initialize()

        try:
            formatted_prompt = await self.format_prompt(prompt, context)
            messages = []

            messages.append({
                "role": "user",
                "content": formatted_prompt
            })

            request_data = {
                "model": config.model,
                "messages": messages,
                "max_tokens": config.max_tokens or 1000,
                "temperature": config.temperature,
            }

            # Add system message if provided
            if config.system_message:
                request_data["system"] = config.system_message

            # Add any extra parameters
            if config.extra_params:
                request_data.update(config.extra_params)

            response = await self._client.post(
                "/messages",
                json=request_data
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
        if not self._initialized:
            await self.initialize()

        try:
            formatted_prompt = await self.format_prompt(prompt, context)

            request_data = {
                "model": config.model,
                "messages": [
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                "max_tokens": config.max_tokens or 1000,
                "temperature": config.temperature,
                "stream": True
            }

            if config.system_message:
                request_data["system"] = config.system_message

            if config.extra_params:
                request_data.update(config.extra_params)

            async with self._client.stream(
                    "POST",
                    "/messages",
                    json=request_data
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or line == "data: [DONE]":
                        continue

                    # Skip ping events
                    if "event: ping" in line:
                        continue

                    try:
                        # Extract the data portion from the SSE line
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            chunk_data = json.loads(data_str)

                            # Handle content block delta events
                            if chunk_data["type"] == "content_block_delta":
                                delta = chunk_data.get("delta", {})
                                if delta.get("type") == "text_delta" and "text" in delta:
                                    yield delta["text"]

                            # Handle complete content blocks
                            elif chunk_data["type"] == "content_block_start":
                                content_block = chunk_data.get("content_block", {})
                                if content_block.get("type") == "text" and "text" in content_block:
                                    yield content_block["text"]

                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {str(e)} for line: {line}")
                        continue
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
                "/messages",  # Note: This endpoint might need to be updated based on Claude's embedding API
                json={
                    "model": config.model if config else "claude-3-sonnet-20240229",
                    "input": [{"type": "text", "text": text}]
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
        except Exception as e:
            raise GenerationError(f"Failed to generate embedding: {str(e)}")
