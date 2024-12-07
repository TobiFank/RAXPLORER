# app/schemas/file.py
from datetime import datetime
from pydantic import BaseModel

class FileMetadata(BaseModel):
    id: str
    name: str
    size: str
    pages: int
    uploaded_at: datetime
    status: str = 'processing'