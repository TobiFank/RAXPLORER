# app/services/rag.py
from llama_index import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.node_parser import SentenceSplitter
from llama_index.vector_stores import QdrantVectorStore
from qdrant_client import QdrantClient

from .llm import LLMService
from ..schemas.model import ModelConfig

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
        self.qdrant = QdrantClient("localhost", port=6333)
        self.vector_store = QdrantVectorStore(client=self.qdrant)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.reranker = SentenceTransformerRerank(top_n=20, model="cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm_service = llm_service

    async def query(self, query: str, file_ids: list[str]) -> str:
        indices = [
            load_index_from_storage(
                storage_context=self.storage_context,
                collection_name=file_id
            ) for file_id in file_ids
        ]

        retriever = indices[0].as_retriever(
            similarity_top_k=150
        )
        nodes = retriever.retrieve(query)
        reranked_nodes = self.reranker.postprocess_nodes(nodes)

        return reranked_nodes[:20]

    async def _generate_context(self, chunk_text: str, full_document: str, model_config: ModelConfig) -> str:
        provider = await self.llm_service.get_provider(model_config)

        messages = [{
            "role": "user",
            "content": CONTEXT_PROMPT.format(
                full_document=full_document,
                chunk=chunk_text
            )
        }]

        context = ""
        async for text in provider.generate(messages, model_config):
            context += text

        return context.strip()

    async def process_document(self, file_content: bytes, file_id: str, model_config: ModelConfig):
        full_document = file_content.decode()
        nodes = SentenceSplitter(chunk_overlap=20).get_nodes_from_documents(file_content)

        for node in nodes:
            node.extra_info = await self._generate_context(
                node.text,
                full_document,
                model_config
            )

        index = VectorStoreIndex.from_documents(
            nodes,
            storage_context=self.storage_context,
            collection_name=file_id
        )
        return index
