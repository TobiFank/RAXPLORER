# app/services/storage.py
from fastapi import UploadFile
from ..schemas.file import FileMetadata
from ..schemas.model import ModelConfig
from ..db.models import FileModel
from ..db.session import AsyncSession
from .rag import RAGService

class StorageService:
    def __init__(self, db: AsyncSession, rag_service: RAGService):
        self.db = db
        self.rag = rag_service

    async def upload(self, file: UploadFile, model_config: ModelConfig) -> FileMetadata:
        content = await file.read()

        # Process with RAG
        index = await self.rag.process_document(
            content,
            str(uuid4()),  # Use as vector_store_id
            model_config
        )

        # Calculate pages (simplified)
        pages = len(content.decode().split('\n')) // 45

        # Create DB record
        file_model = FileModel(
            name=file.filename,
            size=f"{len(content) / 1024:.1f}KB",
            pages=pages,
            vector_store_id=index.collection_name
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

    async def get_files(self) -> list[FileMetadata]:
        files = await self.db.query(FileModel).all()
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
            # Delete from vector store
            await self.rag.delete_index(file.vector_store_id)
            # Delete DB record
            await self.db.delete(file)
            await self.db.commit()