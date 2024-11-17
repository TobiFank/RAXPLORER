# app/services/rag/embeddings.py
from typing import List, Optional
import numpy as np
from app.services.llm.base import BaseLLMService, LLMConfig
from .chunker import Chunk
from app.core.config import settings

import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles the generation and caching of embeddings"""

    def __init__(self, llm_service: BaseLLMService):
        self.llm_service = llm_service
        self._config = LLMConfig(
            model=settings.EMBEDDING_MODEL,  # Use configured embedding model
            temperature=0.0  # Keep this for consistency
        )

    async def generate_embeddings(
            self,
            chunks: List[Chunk],
            config: Optional[LLMConfig] = None
    ) -> List[tuple[Chunk, List[float]]]:
        """Generate embeddings for a list of chunks"""
        results = []
        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Generate embeddings
        for chunk in chunks:
            try:
                logger.info(f"Generating embedding for chunk {chunk.chunk_index}")
                embedding = await self.llm_service.get_embedding(
                    chunk.content,
                    self._config  # Always use our embedding config
                )
                results.append((chunk, embedding))
                logger.info(f"Generated embedding for chunk {chunk.chunk_index}")
            except Exception as e:
                # Log error and continue with remaining chunks
                print(f"Error generating embedding for chunk {chunk.chunk_index}: {str(e)}")
                continue

        logger.info(f"Generated embeddings for {len(results)} chunks")
        return results