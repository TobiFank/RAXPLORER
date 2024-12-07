# app/services/rag_dependencies.py
import json
import logging
from typing import TypeVar, Generic, Union

import chromadb
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ...schemas.model import Provider

logger = logging.getLogger(__name__)

# Types for provider-specific implementations
T = TypeVar('T')

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class SectionType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"


@dataclass
class BoundingBox:
    """Represents the spatial location of content on a page"""
    x0: float  # left
    y0: float  # top
    x1: float  # right
    y1: float  # bottom
    page_num: int


@dataclass
class DocumentImage:
    """Represents an image extracted from a document"""
    image_data: bytes
    page_num: int
    image_type: str
    bbox: BoundingBox
    metadata: Dict[str, any]
    caption: Optional[str] = None
    referenced_by: List[str] = None  # List of section IDs that reference this image


class DocumentSection:
    """Represents a section in a document with its spatial information"""

    def __init__(self,
                 content: str,
                 bbox: BoundingBox,
                 section_type: SectionType,
                 metadata: dict):
        self.id = metadata.get('section_id')
        self.content = content
        self.bbox = bbox
        self.section_type = section_type
        self.metadata = metadata
        self.images: List[DocumentImage] = []
        self.nearby_sections: List['DocumentSection'] = []  # Sections spatially close to this one

    def overlaps_with(self, bbox: BoundingBox, threshold: float = 0.3) -> bool:
        """Check if this section overlaps with a given bounding box"""
        if self.bbox.page_num != bbox.page_num:
            return False

        # Calculate intersection area
        x_left = max(self.bbox.x0, bbox.x0)
        y_top = max(self.bbox.y0, bbox.y0)
        x_right = min(self.bbox.x1, bbox.x1)
        y_bottom = min(self.bbox.y1, bbox.y1)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)
        section_area = (self.bbox.x1 - self.bbox.x0) * (self.bbox.y1 - self.bbox.y0)

        return intersection / section_area > threshold

    def is_nearby(self, other: Union['DocumentSection', BoundingBox], distance_threshold: float = 50) -> bool:
        """Check if another section/bbox is spatially close to this one"""
        other_bbox = other.bbox if isinstance(other, DocumentSection) else other

        if self.bbox.page_num != other_bbox.page_num:
            return False

        vertical_distance = min(
            abs(self.bbox.y1 - other_bbox.y0),
            abs(other_bbox.y1 - self.bbox.y0)
        )

        horizontal_distance = min(
            abs(self.bbox.x1 - other_bbox.x0),
            abs(other_bbox.x1 - self.bbox.x0)
        )

        return min(vertical_distance, horizontal_distance) <= distance_threshold


class EmbeddingProvider(Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VectorStoreProvider(Generic[T]):
    """Base class for vector store implementations"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    async def store_embeddings(self, provider: Provider, sections: List[DocumentSection],
                               embeddings: List[List[float]]):
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
        """Store embeddings with enhanced section metadata"""
        collection = self.get_collection(provider)

        documents = [section.content for section in sections]
        metadatas = []

        for section in sections:
            # Convert basic metadata
            metadata = section.metadata.copy()

            if 'images' in metadata:
                try:
                    # First ensure any existing JSON strings are parsed
                    images_data = metadata['images'] if isinstance(metadata['images'], list) else json.loads(metadata['images'])
                    # Then serialize the full list
                    metadata['images'] = json.dumps(images_data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode images metadata: {metadata['images']}")
                    metadata['images'] = json.dumps([])

            # Add section type
            metadata['section_type'] = section.section_type.value if section.section_type else 'text'

            # Add spatial information - convert bbox to JSON string
            if hasattr(section, 'bbox'):
                metadata['bbox'] = json.dumps({
                    'x0': section.bbox.x0,
                    'y0': section.bbox.y0,
                    'x1': section.bbox.x1,
                    'y1': section.bbox.y1,
                    'page_num': section.bbox.page_num
                })

            # Add nearby section references
            if section.nearby_sections:
                metadata['nearby_section_ids'] = json.dumps([
                    s.metadata.get('section_id') for s in section.nearby_sections
                ])

            metadatas.append(metadata)
            logger.debug(f"Storing section with metadata: {metadata}")

        ids = [str(i) for i in range(len(sections))]
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    async def query(self, provider: Provider, query_embedding: List[float], top_k: int = 5) -> List[Document]:
        """Query with enhanced metadata handling"""
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
                            'page_num': img.get('page_num', 0),  # Add fallback
                            'image_index': img.get('image_index', 0),
                            'image_type': img.get('image_type', 'image'),
                            'caption': img.get('caption'),
                            'bbox': img.get('bbox'),
                            'file_path': img.get('file_path')
                        }
                        for img in images_data
                    ]
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode images metadata: {metadata['images']}")

            # Create document with enhanced metadata
            doc = Document(
                page_content=content,
                metadata={
                    'document_id': metadata['document_id'],
                    'page_num': metadata['page_num'],
                    'section_type': metadata.get('section_type', 'text'),
                    'images': processed_images,
                    'file_path': metadata.get('file_path'),
                    'name': metadata.get('name'),
                    'bbox': metadata.get('bbox'),  # Include spatial information
                    'nearby_section_ids': json.loads(metadata['nearby_section_ids']) if metadata.get(
                        'nearby_section_ids') else []
                }
            )
            documents.append(doc)

        return documents

    async def delete_collection(self, file_id: str):
        """Delete all collections associated with a specific vector store ID"""
        # Since we store embeddings for each provider separately,
        # we need to delete from all provider collections
        for provider in Provider:
            collection = self.get_collection(provider)
            try:
                # Delete all documents with matching vector_store_id
                results = collection.get(
                    where={"document_id": file_id}
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

    async def query_batch(
            self,
            provider: Provider,
            query_embeddings: List[List[float]],
            top_k: int = 5
    ) -> List[List[Document]]:
        """Query with multiple embeddings at once"""
        collection = self.get_collection(provider)
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )

        all_documents = []
        for batch_idx in range(len(query_embeddings)):
            documents = []
            for idx, content in enumerate(results['documents'][batch_idx]):
                metadata = results['metadatas'][batch_idx][idx] if results.get('metadatas') else {}

                if 'document_id' not in metadata or 'page_num' not in metadata:
                    continue

                # Process metadata (keep existing metadata processing code)
                processed_metadata = self._process_metadata(metadata)

                doc = Document(
                    page_content=content,
                    metadata=processed_metadata
                )
                documents.append(doc)
            all_documents.append(documents)

        return all_documents

    def _process_metadata(self, metadata: dict) -> dict:
        """Helper to process metadata consistently"""
        processed = {
            'document_id': metadata['document_id'],
            'page_num': metadata['page_num'],
            'section_type': metadata.get('section_type', 'text'),
            'file_path': metadata.get('file_path'),
            'name': metadata.get('name'),
            'bbox': metadata.get('bbox')
        }

        # Process images if present
        if 'images' in metadata:
            try:
                images_data = json.loads(metadata['images']) if isinstance(metadata['images'], str) else metadata['images']
                processed['images'] = [
                    {
                        'page_num': img.get('page_num', 0),
                        'image_index': img.get('image_index', 0),
                        'image_type': img.get('image_type', 'image'),
                        'caption': img.get('caption'),
                        'bbox': img.get('bbox'),
                        'file_path': img.get('file_path')
                    }
                    for img in images_data
                ]
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode images metadata: {metadata['images']}")
                processed['images'] = []

        # Process nearby sections
        if metadata.get('nearby_section_ids'):
            try:
                processed['nearby_section_ids'] = json.loads(metadata['nearby_section_ids'])
            except json.JSONDecodeError:
                processed['nearby_section_ids'] = []

        return processed


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
