# app/api/v1/files.py
from datetime import datetime
from typing import List

from app.api.v1.deps import get_db, get_file_processor, get_rag_processor
from app.core.config import settings
from app.models.file import File
from app.schemas.file import FileResponse, FileListResponse
from app.services.file.processor import FileProcessor  # Added this import
from app.services.rag.processor import RAGProcessor
from fastapi import APIRouter, Depends, UploadFile, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

router = APIRouter()


@router.post("/upload", response_model=FileResponse)
async def upload_file(
        file: UploadFile,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db),
        file_processor: FileProcessor = Depends(get_file_processor)
):
    """Upload and process a file"""
    try:
        # Check file size
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(400, "File too large")

        # Process file
        processed_file = await file_processor.process_file(file)

        # Create database record
        db_file = File(
            id=processed_file.id,
            name=processed_file.name,
            size=processed_file.size,
            pages=processed_file.page_count,
            uploaded_at=datetime.utcnow(),
            vectorized=False  # Will be updated after background processing
        )
        db.add(db_file)
        db.commit()
        db.refresh(db_file)

        # Start background processing
        background_tasks.add_task(
            file_processor.process_and_vectorize,
            processed_file,
            db_file.id,
            db
        )

        return FileResponse(
            id=db_file.id,
            name=db_file.name,
            size=db_file.size,
            pages=db_file.pages,
            uploaded_at=db_file.uploaded_at,
            vectorized=db_file.vectorized
        )

    except Exception as e:
        raise HTTPException(500, f"File processing failed: {str(e)}")


@router.get("/", response_model=List[FileListResponse])
async def list_files(
        skip: int = 0,
        limit: int = 10,
        db: Session = Depends(get_db)
):
    """List processed files"""
    files = db.query(File).offset(skip).limit(limit).all()
    return [
        FileListResponse(
            id=file.id,
            name=file.name,
            size=file.size,
            pages=file.pages,
            uploaded_at=file.uploaded_at,
            vectorized=file.vectorized
        )
        for file in files
    ]


@router.get("/{file_id}", response_model=FileResponse)
async def get_file(
        file_id: str,
        db: Session = Depends(get_db)
):
    """Get file details"""
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        raise HTTPException(404, "File not found")
    return FileResponse.from_orm(file)


@router.delete("/{file_id}")
async def delete_file(
        file_id: str,
        db: Session = Depends(get_db),
        rag_processor: RAGProcessor = Depends(get_rag_processor)
):
    """Delete a file and its associated data"""
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        raise HTTPException(404, "File not found")

    try:
        await rag_processor.cleanup_document(file_id)
        db.delete(file)
        db.commit()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, f"File deletion failed: {str(e)}")
