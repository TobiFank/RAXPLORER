# app/services/rag_dependencies.py
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, Generic, List

import chromadb
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ..schemas.model import Provider

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
    def __init__(self):
        super().__init__("")
        self.client = chromadb.PersistentClient(path="./chromadb")
        self.collections = {}

    def get_collection(self, provider: Provider):
        collection_name = f"embeddings_{provider.value}"
        if collection_name not in self.collections:
            self.collections[collection_name] = self.client.get_or_create_collection(name=collection_name)
        return self.collections[collection_name]

    async def store_embeddings(self, provider: Provider, sections: List[DocumentSection],
                               embeddings: List[List[float]]):
        """Store embeddings for a specific provider"""
        collection = self.get_collection(provider)

        documents = [section.content for section in sections]
        metadatas = [section.metadata.copy() for section in sections]

        for metadata in metadatas:
            logger.info(f"Storing document with metadata: {metadata}")

        for idx, section in enumerate(sections):
            if section.images:
                metadatas[idx]['images'] = json.dumps([
                    {
                        'page_num': img.page_num,
                        'image_index': img.metadata.get('image_index'),
                        'image_type': img.image_type,
                        'caption': img.metadata.get('caption'),
                        'extension': img.metadata.get('extension', 'png'),
                        'file_path': img.metadata.get('file_path')
                    }
                    for img in section.images
                ])

        ids = [str(i) for i in range(len(sections))]
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    async def query(self, provider: Provider, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        collection = self.get_collection(provider)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = []
        for idx, content in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][idx] if results.get('metadatas') else {}

            if 'document_id' not in metadata or 'page_num' not in metadata:
                continue

            processed_images = []
            if 'images' in metadata:
                try:
                    images_data = json.loads(metadata['images'])
                    processed_images = [
                        {
                            'page_num': img['page_num'],
                            'image_index': img['image_index'],
                            'image_type': img['image_type'],
                            'caption': img.get('caption'),
                            'file_path': f"storage/images/{metadata['document_id']}_{img['page_num']}_{img['image_index']}.{img.get('extension', 'png')}"
                        }
                        for img in images_data
                    ]
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode images metadata: {metadata['images']}")

            doc = Document(
                page_content=content,
                metadata={
                    'document_id': metadata['document_id'],
                    'page_num': metadata['page_num'],
                    'section_type': metadata.get('section_type', 'text'),
                    'images': processed_images,
                    'file_path': metadata.get('file_path'),
                    'name': metadata.get('name')
                }
            )
            documents.append(doc)

        return documents

    async def delete_collection(self, vector_store_id: str):
        """Delete all collections associated with a specific vector store ID"""
        # Since we store embeddings for each provider separately,
        # we need to delete from all provider collections
        for provider in Provider:
            collection = self.get_collection(provider)
            try:
                # Delete all documents with matching vector_store_id
                results = collection.get(
                    where={"document_id": vector_store_id}
                )
                if results and results['ids']:
                    collection.delete(
                        ids=results['ids']
                    )
                    # Remove from cache if exists
                    collection_name = f"embeddings_{provider.value}"
                    if collection_name in self.collections:
                        del self.collections[collection_name]
            except Exception as e:
                logger.error(f"Failed to delete embeddings for provider {provider}: {str(e)}")
                # Continue with other providers even if one fails
                continue



# Add BM25 implementation for hybrid search
class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents: List[Document] = []  # Change to store Document objects
        self._texts: List[str] = []  # Add private texts list for BM25

    def fit(self, documents: List[Document]):  # Change parameter to List[Document]
        """Fit BM25 on a list of documents"""
        if not documents:
            return

        self.documents = documents
        self._texts = [doc.page_content for doc in documents]
        tokenized_docs = [text.split() for text in self._texts]
        self.bm25 = BM25Okapi(tokenized_docs)

    def query(self, query: str, top_k: int = 5) -> List[tuple[Document, float]]:  # Update return type
        """Query BM25 index and return top_k documents with scores"""
        if not self.bm25:
            return []

        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[-top_k:][::-1]
        return [(self.documents[i], scores[i]) for i in top_n]
