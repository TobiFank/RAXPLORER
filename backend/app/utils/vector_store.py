# app/utils/vector_store.py
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import json
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
    MilvusException
)

from app.core.config import settings
from app.services.rag.chunker import Chunk
from app.utils.errors import VectorStoreError

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
            dim: int = 4096,  # Default for Llama2 embeddings
            similarity_metric: str = "COSINE"
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dim = dim
        self.similarity_metric = similarity_metric
        self._collection: Optional[Collection] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Milvus connection and create collection if needed."""
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )

            # Define collection schema
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

            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                self._collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using="default"
                )

                # Create IVF_FLAT index for GPU-accelerated search
                index_params = {
                    "metric_type": self.similarity_metric,
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self._collection.create_index(
                    field_name="embedding",
                    index_params=index_params
                )
            else:
                self._collection = Collection(self.collection_name)

            self._collection.load()
            self._initialized = True

        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Milvus: {str(e)}")

    async def store(self, doc_id: str, chunk: Chunk, embedding: List[float]) -> str:
        """Store a document chunk and its embedding in Milvus."""
        if not self._initialized:
            await self.initialize()

        try:
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

            return data[0][0]  # Return the generated ID

        except Exception as e:
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
            # Prepare search parameters
            search_params = {
                "metric_type": self.similarity_metric,
                "params": {"nprobe": 10}
            }

            # Execute search
            results = self._collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["doc_id", "chunk_index", "content", "metadata"]
            )

            # Format results
            chunks_with_scores = []
            for hits in results:
                for hit in hits:
                    # Create Chunk object from result
                    chunk = Chunk(
                        content=hit.entity.get('content'),
                        metadata=json.loads(hit.entity.get('metadata')),
                        chunk_index=hit.entity.get('chunk_index'),
                        source_id=hit.entity.get('doc_id'),
                        total_chunks=0  # We don't store this in Milvus
                    )
                    chunks_with_scores.append((chunk, hit.score))

            return chunks_with_scores

        except Exception as e:
            raise VectorStoreError(f"Failed to search vectors: {str(e)}")

    async def delete_document(self, doc_id: str) -> None:
        """Delete all chunks belonging to a document."""
        if not self._initialized:
            await self.initialize()

        try:
            # Delete all entries with matching doc_id
            expr = f'doc_id == "{doc_id}"'
            self._collection.delete(expr)
            self._collection.flush()
        except Exception as e:
            raise VectorStoreError(f"Failed to delete document: {str(e)}")

    async def __aenter__(self):
        """Support async context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        if self._initialized:
            connections.disconnect("default")