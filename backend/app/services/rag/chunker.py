# app/services/rag/chunker.py
from typing import List
from pydantic import BaseModel, Field

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
        if not text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(text)
        chunk_index = 0

        while start < text_length:
            # Find the end of the chunk
            end = start + self.chunk_size

            if end >= text_length:
                # Last chunk
                chunk_text = text[start:text_length].strip()
            else:
                # Find the last period or newline within chunk_size
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                break_point = max(last_period, last_newline)

                if break_point > start + self.min_chunk_size:
                    end = break_point + 1

                chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata=metadata or {},
                    chunk_index=chunk_index,
                    source_id=source_id,
                    total_chunks=0  # Will be updated after all chunks are created
                ))
                chunk_index += 1

            start = end - self.chunk_overlap

        # Update total_chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total_chunks

        return chunks
