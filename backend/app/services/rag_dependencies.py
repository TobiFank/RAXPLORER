# app/services/rag_dependencies.py
import json
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
    def __init__(self, collection_name: str):
        super().__init__(collection_name)
        self.client = chromadb.PersistentClient(path="./chromadb")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    async def store_embeddings(self, sections: List[DocumentSection], embeddings: List[List[float]]):
        """Store document sections and their embeddings in ChromaDB"""
        try:
            # Extract content and metadata from sections
            documents = [section.content for section in sections]
            metadatas = [section.metadata.copy() for section in sections]  # Make a copy to avoid modifying original

            # Important: Store images in metadata if they exist
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

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Successfully stored {len(sections)} sections in ChromaDB")

        except Exception as e:
            logger.error(f"Failed to store embeddings in ChromaDB: {str(e)}")
            raise

    async def query(self, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        """Execute a query against the ChromaDB collection"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # Convert results to Documents with proper metadata
            documents = []
            for idx, content in enumerate(results['documents'][0]):  # First list since we only send one query
                metadata = results['metadatas'][0][idx] if results.get('metadatas') else {}

                # Ensure required metadata fields exist
                if 'document_id' not in metadata:
                    logger.warning(f"Document missing document_id in metadata: {content[:100]}")
                    continue

                if 'page_num' not in metadata:
                    logger.warning(f"Document missing page_num in metadata: {content[:100]}")
                    continue

                # Process images if they exist
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

                # Create Document with proper metadata structure
                doc = Document(
                    page_content=content,
                    metadata={
                        'document_id': metadata['document_id'],
                        'page_num': metadata['page_num'],
                        'section_type': metadata.get('section_type', 'text'),
                        'images': processed_images,
                        'file_path': metadata.get('file_path')
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"ChromaDB query failed: {str(e)}")
            return []


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
