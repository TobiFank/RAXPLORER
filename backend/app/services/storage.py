# app/services/storage.py
import logging
from pathlib import Path
from uuid import uuid4

import pymupdf
from fastapi import UploadFile, HTTPException
from sqlalchemy import select

from .rag.rag import RAGService
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

    def _ensure_storage_path(self):
        """Ensure document storage directory exists"""
        storage_path = Path(self.settings.DOCUMENT_STORAGE_PATH)
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    async def upload(self, file: UploadFile, model_config: ModelConfig) -> FileMetadata:
        try:
            logger.info(f"Uploading file {file.filename}")
            content = await file.read()
            logger.info(f"File size: {len(content) / 1024:.1f}KB")

            # Ensure storage directory exists and get path for new file
            storage_path = self._ensure_storage_path()
            file_id = str(uuid4())
            pdf_path = storage_path / f"{file_id}.pdf"

            # Save the PDF file permanently
            with open(pdf_path, 'wb') as f:
                f.write(content)

            try:
                # Now open with PyMuPDF
                pdf = pymupdf.open(str(pdf_path))
                pages = len(pdf)
                logger.info(f"successfully opened PDF with {pages} pages")
            except Exception as e:
                logger.error(f"Failed to open PDF file: {e}")
                pdf_path.unlink(missing_ok=True)  # Clean up file if we can't open it
                raise HTTPException(status_code=500, detail="Failed to open PDF file")

            # Create file record
            file_model = FileModel(
                id=file_id,
                name=file.filename,
                size=f"{len(content) / 1024:.1f}KB",
                pages=pages,
                vector_store_id=str(uuid4()),
                embedding_provider=model_config.provider,
                file_path=str(pdf_path),
                status='processing'
            )

            # Process through RAG
            await self.rag.process_document(
                content,
                file_model,
                model_config
            )

            file_model.status = 'complete'

            # Save to database
            self.db.add(file_model)
            await self.db.commit()

            pdf.close()

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
            # Clean up file if it exists
            if 'pdf_path' in locals():
                pdf_path.unlink(missing_ok=True)
            await self.db.rollback()
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
        result = await self.db.execute(select(FileModel).filter(FileModel.id == file_id))
        file = result.scalar_one_or_none()
        if file:
            try:
                # Delete the PDF file
                if file.file_path:
                    Path(file.file_path).unlink(missing_ok=True)

                # Clean up vector store
                if file.vector_store_id:
                    try:
                        await self.rag.chroma_provider.delete_collection(file.vector_store_id)
                    except Exception as e:
                        # If vector store deletion fails, we should probably fail the whole operation
                        logger.error(f"Failed to delete vector store {file.vector_store_id}: {e}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to clean up document data: {str(e)}"
                        )

                image_dir = Path("storage/images")
                if image_dir.exists():
                    all_files = list(image_dir.iterdir())
                    logger.debug(f"Total files in image directory: {len(all_files)}")
                    logger.debug(f"Files in directory: {[f.name for f in all_files]}")
                    logger.debug(f"Searching for pattern: {file_id}_*")
                    images_found = list(image_dir.glob(f"{file_id}_*"))
                    logger.debug(f"Found {len(images_found)} images to delete")
                    for image_file in images_found:
                        try:
                            image_file.unlink()
                            logger.debug(f"Deleted image: {image_file}")
                        except Exception as e:
                            logger.error(f"Failed to delete image {image_file}: {e}")
                else:
                    logger.warning(f"Image directory does not exist: {image_dir.absolute()}")

                # Remove database record
                await self.db.delete(file)
                await self.db.commit()
            except Exception as e:
                logger.error(f"Failed to delete file {file_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
