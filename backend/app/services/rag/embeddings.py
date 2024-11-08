# app/services/rag/embeddings.py
from typing import List, Optional
import numpy as np
from app.services.llm.base import BaseLLMService, LLMConfig
from .chunker import Chunk

class EmbeddingService:
    """Handles the generation and caching of embeddings"""

    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service
        self._config = LLMConfig(
            model="llama2",  # Default model for embeddings
            temperature=0.0
        )

    async def generate_embeddings(
            self,
            chunks: List[Chunk],
            config: Optional[LLMConfig] = None
    ) -> List[tuple[Chunk, List[float]]]:
        """Generate embeddings for a list of chunks"""
        results = []
        embedding_config = config or self._config

        for chunk in chunks:
            try:
                embedding = await self.llm_service.get_embedding(
                    chunk.content,
                    embedding_config
                )
                results.append((chunk, embedding))
            except Exception as e:
                # Log error and continue with remaining chunks
                print(f"Error generating embedding for chunk {chunk.chunk_index}: {str(e)}")
                continue

        return results