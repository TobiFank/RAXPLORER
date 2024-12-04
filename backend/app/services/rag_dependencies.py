# app/services/rag_dependencies.py
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar, Generic, List

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)

# Types for provider-specific implementations
T = TypeVar('T')


class DocumentSection:
    """Represents a section in a document with its metadata"""

    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata
        # Store images separately but linked to this section
        self.images: List[DocumentImage] = []


@dataclass
class DocumentImage:
    """Represents an image extracted from a document"""
    image_data: bytes
    page_num: int
    image_type: str  # 'image', 'table', 'diagram'
    metadata: dict


class EmbeddingProvider(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VectorStoreProvider(Generic[T]):
    """Base class for vector store implementations"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    async def store_embeddings(self, sections: List[DocumentSection], embeddings: List[List[float]]):
        raise NotImplementedError

    async def query(self, query_embedding: List[float], top_k: int = 5) -> List[T]:
        raise NotImplementedError


class ChromaProvider(VectorStoreProvider[Document]):
    """ChromaDB implementation of vector store"""

    def __init__(self, collection_name: str):
        super().__init__(collection_name)
        self.client = chromadb.PersistentClient(path="./chromadb")
        # Initialize embeddings model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Initialize or get collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

    async def query(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        """Execute a query against the ChromaDB collection"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # Convert results to Documents
            documents = []
            for idx, content in enumerate(results['documents'][0]):  # First list since we only send one query
                metadata = results['metadatas'][0][idx] if results.get('metadatas') else {}
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

            return documents

        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}")
            return []


# Add BM25 implementation for hybrid search
class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []

    def fit(self, documents: List[str]):
        """Fit BM25 on a list of documents"""
        if not documents:
            return

        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents

    def query(self, query: str, top_k: int = 5) -> List[tuple[str, float]]:
        """Query BM25 index and return top_k documents with scores"""
        if not self.bm25:
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i], scores[i]) for i in top_n]
