# app/services/storage.py
import logging
import os
from tempfile import NamedTemporaryFile
from uuid import uuid4

import pymupdf
from fastapi import UploadFile, HTTPException
from sqlalchemy import select

from .rag import RAGService
from ..core.config import Settings
from ..db.models import FileModel
from ..db.session import AsyncSession
from ..schemas.file import FileMetadata
from ..schemas.model import ModelConfig

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self, db: AsyncSession, rag_service: RAGService):
        self.db = db
        self.rag = rag_service
        self.settings = Settings()

    async def upload(self, file: UploadFile, model_config: ModelConfig) -> FileMetadata:
        try:
            logger.info(f"Uploading file {file.filename}")
            # Create a temporary file to handle the upload
            content = await file.read()
            logger.info(f"File size: {len(content) / 1024:.1f}KB")
            with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                logger.info(f"Temporary file: {tmp_file.name}")
                tmp_file.write(content)
                tmp_file.flush()

                with open(tmp_file.name, 'rb') as f:
                    first_bytes = f.read(4)
                    logger.info(f"First bytes of file: {first_bytes.hex()}")

                try:
                    # Now open with PyMuPDF
                    pdf = pymupdf.open(tmp_file.name)
                    pages = len(pdf)
                    logger.info(f"successfully opened PDF with {pages} pages")
                except Exception as e:
                    logger.error(f"Failed to open PDF file: {e}")
                    raise HTTPException(status_code=500, detail="Failed to open PDF file")

                # Extract text for RAG
                text_content = ""
                for page_num in range(pages):
                    page = pdf.load_page(page_num)
                    text_content += page.get_text()

                pdf.close()

                # Create file record
                file_model = FileModel(
                    name=file.filename,
                    size=f"{len(content) / 1024:.1f}KB",
                    pages=pages,
                    vector_store_id=str(uuid4()),
                    embedding_provider=model_config.provider
                )

                # Process through RAG
                await self.rag.process_document(
                    content,
                    file_model.vector_store_id,
                    model_config
                )

                # Save to database
                self.db.add(file_model)
                await self.db.commit()

                return FileMetadata(
                    id=file_model.id,
                    name=file_model.name,
                    size=file_model.size,
                    pages=pages,
                    uploaded_at=file_model.uploaded_at
                )

        except Exception as e:
            import traceback
            logger.error(f"Failed to upload file {file.filename}: {str(e)}\n{traceback.format_exc()}")
            await self.db.rollback()
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temporary file
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)

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
        result = await self.db.execute(select(FileModel).filter(FileModel.id == file_id))
        file = result.scalar_one_or_none()
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
