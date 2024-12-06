# app/schemas/rag.py
from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_name: str
    page_number: int
    section: Optional[str] = None
    text: str
    quote_start: str  # First few words of the quote
    quote_end: str  # Last few words of the quote
    file_path: str | None = None
    metadata: dict = Field(default_factory=dict)


class ImageReference(BaseModel):
    image_id: str
    document_name: str
    page_number: int
    image_type: str  # 'image', 'table', 'diagram'
    caption: Optional[str] = None
    file_path: Optional[str] = None


class RAGResponse(BaseModel):
    answer: str
    citations: List[Citation]
    images: List[ImageReference]
    reasoning: Optional[str] = None
    confidence_score: float
