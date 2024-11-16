# app/utils/vector_store.py
import json
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from app.services.rag.chunker import Chunk
from app.utils.errors import VectorStoreError
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)


class VectorStore(ABC):
    """Abstract base class for vector storage implementations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store connection."""
        pass

    @abstractmethod
    async def store(self, doc_id: str, chunk: Chunk, embedding: List[float]) -> str:
        """Store a document chunk and its embedding."""
        pass

    @abstractmethod
    async def similarity_search(
            self,
            query_embedding: List[float],
            top_k: int = 3,
            filter_expr: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """Find similar chunks based on embedding."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Delete all chunks for a document."""
        pass


class MilvusVectorStore(VectorStore):
    """Milvus implementation of vector store."""

    def __init__(
            self,
            host: str = "localhost",
            port: int = 19530,
            collection_name: str = "document_chunks",
            dim: int = 1024,
            similarity_metric: str = "COSINE"
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.similarity_metric = similarity_metric
        self._collection: Optional[Collection] = None
        self._initialized = False
        self._is_loaded = False

    async def initialize(self) -> None:
        """Initialize Milvus connection and create collection if needed."""
        try:
            if not self._initialized:
                connections.connect(
                    alias="default",
                    host=self.host,
                    port=self.port
                )

                if not utility.has_collection(self.collection_name):
                    fields = [
                        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
                        FieldSchema(name="chunk_index", dtype=DataType.INT64),
                        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
                    ]

                    schema = CollectionSchema(
                        fields=fields,
                        description="Document chunks collection for RAG"
                    )

                    self._collection = Collection(
                        name=self.collection_name,
                        schema=schema,
                        using="default"
                    )

                    index_params = {
                        "metric_type": self.similarity_metric,
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 4096}
                    }
                    self._collection.create_index(
                        field_name="embedding",
                        index_params=index_params
                    )
                else:
                    self._collection = Collection(self.collection_name)

                self._initialized = True

        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Milvus: {str(e)}")

    async def _ensure_collection_loaded(self):
        """Ensure collection is loaded only when needed."""
        if not self._is_loaded:
            self._collection.load()
            self._is_loaded = True

    async def _release_collection(self):
        """Release collection from memory."""
        if self._is_loaded:
            self._collection.release()
            self._is_loaded = False

    async def store(self, doc_id: str, chunk: Chunk, embedding: List[float]) -> str:
        """Store a document chunk and its embedding in Milvus."""
        if not self._initialized:
            await self.initialize()

        try:
            await self._ensure_collection_loaded()

            # Prepare data
            data = [
                [str(chunk.chunk_index) + "_" + doc_id],  # id
                [doc_id],
                [chunk.chunk_index],
                [chunk.content],
                [json.dumps(chunk.metadata)],
                [embedding]
            ]

            # Insert data
            self._collection.insert(data)
            self._collection.flush()

            await self._release_collection()
            return data[0][0]  # Return the generated ID

        except Exception as e:
            await self._release_collection()
            raise VectorStoreError(f"Failed to store vector: {str(e)}")

    async def similarity_search(
            self,
            query_embedding: List[float],
            top_k: int = 3,
            filter_expr: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks using Milvus."""
        if not self._initialized:
            await self.initialize()

        try:
            await self._ensure_collection_loaded()

            search_params = {
                "metric_type": self.similarity_metric,
                "params": {"nprobe": 10}
            }

            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["doc_id", "chunk_index", "content", "metadata"]
            )

            chunks_with_scores = []
            for hits in results:
                for hit in hits:
                    chunk = Chunk(
                        content=hit.entity.get('content'),
                        metadata=json.loads(hit.entity.get('metadata')),
                        chunk_index=hit.entity.get('chunk_index'),
                        source_id=hit.entity.get('doc_id'),
                        total_chunks=0  # We don't store this in Milvus
                    )
                    chunks_with_scores.append((chunk, hit.score))

            await self._release_collection()
            return chunks_with_scores

        except Exception as e:
            await self._release_collection()
            raise VectorStoreError(f"Failed to search vectors: {str(e)}")

    async def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document."""
        if not self._initialized:
            await self.initialize()

        try:
            await self._ensure_collection_loaded()

            # Delete all entries with matching doc_id
            expr = f'doc_id == "{doc_id}"'
            self._collection.delete(expr)
            self._collection.flush()

            await self._release_collection()
        except Exception as e:
            await self._release_collection()
            raise VectorStoreError(f"Failed to delete document: {str(e)}")

    async def __aenter__(self):
        """Support async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        await self._release_collection()
        if self._initialized:
            connections.disconnect("default")

    async def cleanup_collection(self) -> None:
        """Drop the existing collection to allow recreation with new settings."""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self._initialized = False
                self._is_loaded = False
        except Exception as e:
            raise VectorStoreError(f"Failed to cleanup collection: {str(e)}")
