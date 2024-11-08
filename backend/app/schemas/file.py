# app/schemas/file.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class FileBase(BaseModel):
    name: str
    size: int
    pages: Optional[int] = None
    vectorized: bool = False

class FileCreate(FileBase):
    pass

class FileResponse(FileBase):
    id: str
    uploaded_at: datetime

    class Config:
        from_attributes = True

class FileListResponse(FileResponse):
    pass

class FileProcessingStatus(BaseModel):
    file_id: str
    status: str  # "processing", "completed", "failed"
    progress: float  # 0 to 1
    error: Optional[str] = None

    class Config:
        from_attributes = True