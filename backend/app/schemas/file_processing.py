# app/schemas/file_processing.py
from typing import Optional, Dict
from pydantic import BaseModel
from datetime import datetime

class ProcessedFile(BaseModel):
    """Represents a processed file with metadata"""
    id: str
    name: str
    size: int
    mime_type: str
    page_count: Optional[int]
    content: str
    metadata: Dict = {}

class ProcessingStatus(BaseModel):
    """Status of document processing"""
    document_id: str
    status: str  # 'processing', 'completed', 'failed'
    total_chunks: int
    processed_chunks: int
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None