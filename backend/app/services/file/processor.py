# app/services/file/processor.py
from typing import Optional, BinaryIO
import os
from pathlib import Path
import magic
import pypdf
from docx import Document
from PIL import Image
import pytesseract
from fastapi import UploadFile, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.services.rag.processor import RAGProcessor

class ProcessedFile(BaseModel):
    """Represents a processed file with metadata"""
    id: str
    name: str
    size: int
    mime_type: str
    page_count: Optional[int]
    metadata: dict

class FileProcessor:
    """Handles file processing and text extraction"""

    def __init__(self, rag_processor: RAGProcessor):
        self.rag_processor = rag_processor
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)

    def _get_mime_type(self, file: BinaryIO) -> str:
        """Detect file MIME type"""
        mime = magic.Magic(mime=True)
        file.seek(0)
        mime_type = mime.from_buffer(file.read(2048))
        file.seek(0)
        return mime_type

    def _extract_text_from_pdf(self, file_path: Path) -> tuple[str, int]:
        """Extract text from PDF file"""
        with open(file_path, 'rb') as file:
            pdf = pypdf.PdfReader(file)
            text_parts = []
            for page in pdf.pages:
                text_parts.append(page.extract_text())
            return '\n'.join(text_parts), len(pdf.pages)

    def _extract_text_from_docx(self, file_path: Path) -> tuple[str, int]:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text_parts = []
        for para in doc.paragraphs:
            text_parts.append(para.text)
        return '\n'.join(text_parts), len(doc.paragraphs)

    def _extract_text_from_image(self, file_path: Path) -> tuple[str, int]:
        """Extract text from image using OCR"""
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text, 1

    async def process_file(self, file: UploadFile) -> ProcessedFile:
        """Process an uploaded file through the pipeline"""
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(400, "File too large")

        # Save file temporarily
        file_path = self.upload_dir / file.filename
        try:
            content = await file.read()
            with open(file_path, 'wb') as f:
                f.write(content)

            # Detect file type
            mime_type = self._get_mime_type(file.file)

            # Extract text based on file type
            text = ""
            page_count = 0

            if mime_type == 'application/pdf':
                text, page_count = self._extract_text_from_pdf(file_path)
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                text, page_count = self._extract_text_from_docx(file_path)
            elif mime_type.startswith('image/'):
                text, page_count = self._extract_text_from_image(file_path)
            else:
                raise HTTPException(400, f"Unsupported file type: {mime_type}")

            # Process through RAG pipeline
            document_id = str(file_path.stem)
            metadata = {
                'filename': file.filename,
                'mime_type': mime_type,
                'size': file.size,
                'page_count': page_count
            }

            await self.rag_processor.process_document(
                text,
                document_id,
                metadata
            )

            return ProcessedFile(
                id=document_id,
                name=file.filename,
                size=file.size,
                mime_type=mime_type,
                page_count=page_count,
                metadata=metadata
            )

        finally:
            # Cleanup temporary file
            if file_path.exists():
                file_path.unlink()

    async def remove_file(self, file_id: str) -> None:
        """Remove a file and its associated data"""
        # Remove from vector store
        await self.rag_processor.vector_store.delete_document(file_id)