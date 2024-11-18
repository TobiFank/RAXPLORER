# app/services/storage.py
import logging
from http.client import HTTPException
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import select

from .rag import RAGService
from ..db.models import FileModel
from ..db.session import AsyncSession
from ..schemas.file import FileMetadata

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self, db: AsyncSession, rag_service: RAGService):
        self.db = db
        self.rag = rag_service

    async def upload(self, file: UploadFile) -> FileMetadata:
        try:
            content = await file.read()

            # Basic page estimation
            pages = len(content.decode().split('\n')) // 45

            # Create file record
            file_model = FileModel(
                name=file.filename,
                size=f"{len(content) / 1024:.1f}KB",
                pages=pages,
                vector_store_id=str(uuid4())
            )

            self.db.add(file_model)
            await self.db.commit()

            return FileMetadata(
                id=file_model.id,
                name=file_model.name,
                size=file_model.size,
                pages=file_model.pages,
                uploaded_at=file_model.uploaded_at
            )
        except Exception as e:
            logger.error(f"Failed to upload file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_files(self) -> list[FileMetadata]:
        result = await self.db.execute(select(FileModel))
        files = result.scalars().all()
        return [
            FileMetadata(
                id=f.id,
                name=f.name,
                size=f.size,
                pages=f.pages,
                uploaded_at=f.uploaded_at
            ) for f in files
        ]

    async def delete(self, file_id: str):
        file = await self.db.query(FileModel).get(file_id)
        if file:
            try:
                # Clean up vector store first
                try:
                    await self.rag.qdrant.delete_collection(file.vector_store_id)
                except Exception as e:
                    logger.error(f"Failed to delete vector store {file.vector_store_id}: {e}")

                # Then remove database record
                await self.db.delete(file)
                await self.db.commit()
            except Exception as e:
                logger.error(f"Failed to delete file {file_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
