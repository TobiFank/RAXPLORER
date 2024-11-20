# app/services/rag.py
import logging

import httpx
from fastapi import HTTPException
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage, Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.legacy.embeddings import VoyageEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from .llm import LLMService
from ..db.session import Settings
from ..schemas.model import ModelConfig

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """
<document>
{full_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
"""


class RAGService:
    def __init__(self, llm_service: LLMService):
        self.settings = Settings()
        self.qdrant = QdrantClient(host=self.settings.QDRANT_HOST, port=self.settings.QDRANT_PORT)
        self.vector_store = QdrantVectorStore(client=self.qdrant, collection_name="default")
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.reranker = SentenceTransformerRerank(top_n=20, model=self.settings.RERANKING_MODEL)
        self.llm_service = llm_service
        self._model_initialized = False

    async def ensure_model_available(self, model_config: ModelConfig):
        if self._model_initialized:
            return

        settings = self.settings
        async with httpx.AsyncClient() as client:
            # Pull chat model if using Ollama
            if model_config.provider == "ollama":
                await client.post(
                    f"{self.settings.OLLAMA_HOST}/api/pull",
                    json={"name": model_config.model},
                    timeout=None
                )
            # Pull embedding model
            response = await client.post(
                f"{self.settings.OLLAMA_HOST}/api/pull",
                json={"name": settings.OLLAMA_EMBEDDING_MODEL},
                timeout=None
            )
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Failed to pull embedding model: {response.text}")

        self._model_initialized = True

    async def query(self, query: str, file_ids: list[str]) -> list:
        all_nodes = []

        for file_id in file_ids:
            try:
                index = load_index_from_storage(
                    storage_context=self.storage_context,
                    collection_name=file_id
                )
                nodes = index.as_retriever(similarity_top_k=50).retrieve(query)
                all_nodes.extend(nodes)
            except Exception as e:
                logger.error(f"Failed to query index {file_id}: {e}")
                continue

        if not all_nodes:
            return []

        reranked = self.reranker.postprocess_nodes(all_nodes)
        return reranked[:20]

    async def _generate_context(self, chunk_text: str, full_document: str, model_config: ModelConfig) -> str:
        provider = await self.llm_service.get_provider(model_config)
        messages = [{
            "role": "user",
            "content": CONTEXT_PROMPT.format(full_document=full_document, chunk=chunk_text)
        }]
        context = ""
        logger.info(f"Generating context with model config: {model_config}")
        async for text in provider.generate(messages, model_config):
            context += text
        return context.strip()

    async def set_embedding_model(self, model_config: ModelConfig):
        if model_config.provider == "claude":
            Settings.embed_model = VoyageEmbedding(api_key=model_config.apiKey)
        elif model_config.provider == "chatgpt":
            Settings.embed_model = OpenAIEmbedding(api_key=model_config.apiKey)
        else:  # ollama
            Settings.embed_model = OllamaEmbedding(model_name=self.settings.OLLAMA_EMBEDDING_MODEL,
                                                   base_url=self.settings.OLLAMA_HOST)

    async def process_document(self, file_content: bytes, file_id: str, model_config: ModelConfig):
        try:
            logger.info(f"Processing document {file_id} with model config: {model_config}")
            await self.ensure_model_available(model_config)
            await self.set_embedding_model(model_config)

            text_content = file_content.decode()
            doc = Document(text=text_content, id_=file_id)
            nodes = SentenceSplitter(chunk_overlap=20).get_nodes_from_documents([doc])

            for node in nodes:
                node.metadata["context"] = await self._generate_context(node.text, text_content, model_config)

            # Then get embeddings
            provider = await self.llm_service.get_provider(model_config)
            for node in nodes:
                node.embedding = await provider.get_embeddings(node.text, model_config)

            index = VectorStoreIndex(
                nodes,
                storage_context=self.storage_context,
                collection_name=file_id,
                embed_model=Settings.embed_model
            )
            return index

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
