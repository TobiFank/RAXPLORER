# app/services/rag/retriever.py
from typing import List, Optional
import numpy as np
from pydantic import BaseModel
from .chunker import Chunk

class RetrievalResult(BaseModel):
    """Represents a retrieved chunk with similarity score"""
    chunk: Chunk
    similarity: float

class Retriever:
    """Handles similarity search and context retrieval"""

    def __init__(self, vector_store):
        self.vector_store = vector_store

    async def find_similar(
            self,
            query_embedding: List[float],
            top_k: int = 3,
            similarity_threshold: float = 0.7
    ) -> List[RetrievalResult]:
        """Find most similar chunks for a query embedding"""
        results = await self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k
        )

        return [
            RetrievalResult(chunk=chunk, similarity=score)
            for chunk, score in results
            if score >= similarity_threshold
        ]

    def format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieved chunks into a context string"""
        if not results:
            return ""

        # Sort by chunk index within each source
        sorted_results = sorted(
            results,
            key=lambda x: (x.chunk.source_id, x.chunk.chunk_index)
        )

        context_parts = []
        current_source = None

        for result in sorted_results:
            chunk = result.chunk
            if chunk.source_id != current_source:
                current_source = chunk.source_id
                context_parts.append(f"\nSource: {chunk.metadata.get('title', chunk.source_id)}")
            context_parts.append(chunk.content)

        return "\n".join(context_parts)