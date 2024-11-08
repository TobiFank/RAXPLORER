# app/services/rag/processor.py
from typing import List, Optional
from pydantic import BaseModel
from app.services.llm.base import BaseLLMService, LLMConfig
from .chunker import TextChunker, Chunk
from .embeddings import EmbeddingService
from .retriever import Retriever, RetrievalResult

class ProcessedDocument(BaseModel):
    """Represents a processed document with chunks and metadata"""
    id: str
    chunks: List[Chunk]
    metadata: dict

class RAGProcessor:
    """Main RAG pipeline coordinator"""

    def __init__(
            self,
            llm_service: BaseLLMService,
            vector_store,
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ):
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedding_service = EmbeddingService(llm_service)
        self.retriever = Retriever(vector_store)
        self.llm_service = llm_service

    async def process_document(
            self,
            text: str,
            document_id: str,
            metadata: dict = None
    ) -> ProcessedDocument:
        """Process a document through the RAG pipeline"""
        # Create chunks
        chunks = self.chunker.chunk_text(text, document_id, metadata)

        # Generate embeddings
        chunk_embeddings = await self.embedding_service.generate_embeddings(chunks)

        # Store in vector database
        for chunk, embedding in chunk_embeddings:
            await self.vector_store.store(
                document_id,
                chunk,
                embedding
            )

        return ProcessedDocument(
            id=document_id,
            chunks=chunks,
            metadata=metadata or {}
        )

    async def get_relevant_context(
            self,
            query: str,
            top_k: int = 3
    ) -> str:
        """Get relevant context for a query"""
        # Generate query embedding
        query_embedding = await self.llm_service.get_embedding(query)

        # Retrieve similar chunks
        results = await self.retriever.find_similar(
            query_embedding,
            top_k=top_k
        )

        # Format context
        return self.retriever.format_context(results)

    async def generate_augmented_response(
            self,
            query: str,
            config: LLMConfig,
            top_k: int = 3
    ) -> str:
        """Generate a response augmented with relevant context"""
        context = await self.get_relevant_context(query, top_k)

        response = await self.llm_service.generate(
            query,
            config,
            context=context
        )

        return response.content