# app/services/rag/chunker.py
import logging
from typing import List

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    """Represents a text chunk with metadata"""
    content: str
    metadata: dict = Field(default_factory=dict)
    chunk_index: int
    source_id: str
    total_chunks: int


class TextChunker:
    """Handles text chunking with different strategies"""

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str, source_id: str, metadata: dict = None) -> List[Chunk]:
        """Split text into overlapping chunks"""
        logger.info(f"Start chunking")
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        chunks = []
        start = 0
        text_length = len(text)
        chunk_index = 0

        logger.info(f"Chunking text with {len(text)} characters")

        while start < text_length:
            # Calculate the maximum possible end point
            end = min(start + self.chunk_size, text_length)

            if end == text_length:
                # Handle last chunk
                chunk_text = text[start:end].strip()
            else:
                # Try to find a natural break point
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)

                # Only consider break points that give us a reasonable chunk size
                valid_break_points = []
                if last_period > start + self.min_chunk_size:
                    valid_break_points.append(last_period)
                if last_newline > start + self.min_chunk_size:
                    valid_break_points.append(last_newline)

                if valid_break_points:
                    # Use the latest valid break point
                    end = max(valid_break_points) + 1
                # If no valid break points, keep the calculated end (chunk_size or text_length)

                chunk_text = text[start:end].strip()

            # Only add non-empty chunks that meet minimum size
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata=metadata or {},
                    chunk_index=chunk_index,
                    source_id=source_id,
                    total_chunks=0  # Will be updated after all chunks are created
                ))
                chunk_index += 1

            # Critical fix: Ensure we always move forward by at least min_chunk_size
            # Calculate next start position
            next_start = end - self.chunk_overlap

            # If the overlap would cause us to move forward by less than min_chunk_size
            # or even move backwards, force a minimum movement
            if next_start <= start + self.min_chunk_size:
                start = start + self.min_chunk_size
            else:
                start = next_start

        logger.info(f"Created {len(chunks)} chunks")
        logger.info("left while loop")

        # Update total_chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks

        logger.info(f"End chunking")
        return chunks
