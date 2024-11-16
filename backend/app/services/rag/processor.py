# app/services/rag/processor.py
import asyncio
import uuid
from datetime import datetime
from typing import List, Optional

from app.schemas.file_processing import ProcessedFile, ProcessingStatus
from app.services.llm.base import BaseLLMService
from app.utils.vector_store import VectorStore

from .chunker import TextChunker, Chunk
from .embeddings import EmbeddingService
from .retriever import Retriever, RetrievalResult


class RAGProcessor:
    """Main RAG pipeline coordinator with improved processing status tracking"""

    def __init__(
            self,
            llm_service: BaseLLMService,
            vector_store: VectorStore,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            batch_size: int = 10  # Process chunks in batches
    ):
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding_service = EmbeddingService(llm_service)
        self.retriever = Retriever(vector_store)
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.batch_size = batch_size
        self._processing_statuses: dict[str, ProcessingStatus] = {}

    def _create_processing_status(self, document_id: str, total_chunks: int) -> ProcessingStatus:
        """Create initial processing status for a document"""
        status = ProcessingStatus(
            document_id=document_id,
            status="processing",
            total_chunks=total_chunks,
            processed_chunks=0,
            started_at=datetime.utcnow()
        )
        self._processing_statuses[document_id] = status
        return status

    def get_processing_status(self, document_id: str) -> Optional[ProcessingStatus]:
        """Get current processing status of a document"""
        return self._processing_statuses.get(document_id)

    async def process_chunks_batch(
            self,
            chunks: List[Chunk],
            document_id: str,
            status: ProcessingStatus
    ):
        """Process a batch of chunks"""
        try:
            # Generate embeddings for the batch
            chunk_embeddings = await self.embedding_service.generate_embeddings(chunks)

            # Store each chunk and its embedding
            for chunk, embedding in chunk_embeddings:
                await self.vector_store.store(document_id, chunk, embedding)
                status.processed_chunks += 1

            # Force garbage collection after batch processing
            chunk_embeddings = None

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            raise

    async def process_document(
            self,
            text: str,
            metadata: dict,
            file_info: ProcessedFile
    ) -> ProcessingStatus:
        """Process a document through the RAG pipeline with batched processing"""
        document_id = str(uuid.uuid4())

        try:
            # Create chunks
            chunks = self.chunker.chunk_text(
                text=text,
                source_id=document_id,
                metadata={
                    **metadata,
                    "file_name": file_info.name,
                    "file_type": file_info.mime_type,
                    "processed_at": datetime.utcnow().isoformat()
                }
            )

            # Initialize processing status
            status = self._create_processing_status(document_id, len(chunks))

            # Process chunks in batches
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                await self.process_chunks_batch(batch, document_id, status)

                # Clear batch from memory
                batch = None

                # Add a small delay between batches to allow memory cleanup
                await asyncio.sleep(0.1)

            # Mark processing as complete
            status.status = "completed"
            status.completed_at = datetime.utcnow()

            return status

        except Exception as e:
            # Update status with error
            if document_id in self._processing_statuses:
                status = self._processing_statuses[document_id]
                status.status = "failed"
                status.error_message = str(e)
                status.completed_at = datetime.utcnow()
            raise

    async def get_relevant_context(
            self,
            query: str,
            top_k: int = 3,
            doc_ids: Optional[List[str]] = None
    ) -> str:
        """Get relevant context for a query"""
        # Generate query embedding
        query_embedding = await self.llm_service.get_embedding(query)

        # Prepare filter expression if doc_ids provided
        filter_expr = f'doc_id in {doc_ids}' if doc_ids else None

        # Retrieve similar chunks
        raw_results = await self.vector_store.similarity_search(
            query_embedding,
            top_k=top_k,
            filter_expr=filter_expr
        )

        # Convert raw results to RetrievalResult objects
        retrieval_results = [
            RetrievalResult(chunk=chunk, similarity=score)
            for chunk, score in raw_results
        ]

        # Format context using retriever
        return self.retriever.format_context(retrieval_results)

    async def cleanup_document(self, document_id: str) -> None:
        """Clean up all data related to a document"""
        # Remove from vector store
        await self.vector_store.delete_document(document_id)

        # Clear processing status
        if document_id in self._processing_statuses:
            del self._processing_statuses[document_id]
