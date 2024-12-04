# app/services/rag.py
import logging
from typing import List

import fitz
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .llm import LLMService
from .rag_dependencies import (
    ChromaProvider, BM25Retriever, DocumentSection,
    DocumentImage
)
from ..schemas.model import ModelConfig
from ..schemas.rag import RAGResponse, ImageReference, Citation

logger = logging.getLogger(__name__)


class SubQuestion(BaseModel):
    """Model for decomposed questions"""
    question: str = Field(description="The sub-question to be answered")
    reasoning: str = Field(description="Why this sub-question is relevant")


class QueryAnalysis(BaseModel):
    """Model for query analysis output"""
    main_intent: str = Field(description="The main intent of the query")
    sub_questions: List[SubQuestion] = Field(description="List of sub-questions to answer")


QUERY_ANALYSIS_PROMPT = """Analyze this query and break it down into sub-questions.
Main Query: {query}

Please respond with a JSON object that contains:
1. A "main_intent" field with a string describing the core purpose of the query
2. A "sub_questions" array containing objects with "question" and "reasoning" fields

Example response format:
{{"main_intent": "understand what the user means by X", "sub_questions": [{{"question": "what is X?", "reasoning": "need to clarify the basic concept"}}]}}

Remember: Respond ONLY with the JSON object, no other text or schema information.

Question to analyze: {query}"""

STEP_BACK_PROMPT = """Before directly answering the query, let's take a step back and think about the broader context.
Query: {query}

What broader topics or concepts should we consider to provide a more comprehensive answer?
Focus on generating a more general query that will help retrieve relevant context."""

ANSWER_GENERATION_PROMPT = """Given the following context and question, provide a comprehensive answer. Use the context carefully and cite your sources.

Context:
{context}

Question: {question}

Images Available:
{images}

Instructions:
1. Answer the question using information from the context
2. Use logical reasoning to connect information
3. Cite sources using [Doc: Page X] format
4. Reference relevant images using [Image X] format
5. Maintain high confidence in your answer

Answer in this structured format:
Answer: [Your detailed answer with inline citations]
Reasoning: [Your step-by-step reasoning process]
Confidence: [Score between 0-1 based on context relevance]"""


class RAGService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.chroma_provider = ChromaProvider("default")
        self.bm25_retriever = BM25Retriever()

        # Initialize text splitter for hierarchical chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Initialize prompt templates and parsers
        self.query_analyzer = PydanticOutputParser(pydantic_object=QueryAnalysis)
        self.query_analysis_prompt = PromptTemplate(
            template=QUERY_ANALYSIS_PROMPT,
            input_variables=["query"],
            partial_variables={"format_instructions": self.query_analyzer.get_format_instructions()}
        )
        self.step_back_prompt = PromptTemplate(
            template=STEP_BACK_PROMPT,
            input_variables=["query"]
        )

    async def process_document(self, file_content: bytes, file_id: str, model_config: ModelConfig):
        """Process a document and store its embeddings and metadata"""
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            sections: List[DocumentSection] = []

            # Process each page
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # Extract text and split into sections
                text = page.get_text()
                raw_sections = self.text_splitter.split_text(text)

                # Extract images and tables
                for image_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image = DocumentImage(
                        image_data=base_image["image"],
                        page_num=page_num,
                        image_type="image",
                        metadata={
                            "page_num": page_num,
                            "image_index": image_index
                        }
                    )

                    # Find closest section to link image
                    if sections:
                        sections[-1].images.append(image)

                # Create sections with metadata
                for section_text in raw_sections:
                    section = DocumentSection(
                        content=section_text,
                        metadata={
                            "page_num": page_num,
                            "document_id": file_id,
                            "section_type": "text"
                        }
                    )
                    sections.append(section)

            # Create embeddings and store in ChromaDB
            texts = [section.content for section in sections]
            metadatas = [section.metadata for section in sections]
            embeddings = await self._get_embeddings(texts, model_config)

            # Store in ChromaDB
            await self.chroma_provider.store_embeddings(sections, embeddings)

            # Update BM25 index
            self.bm25_retriever.fit(texts)

            return True

        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise

    async def query(self, query: str, file_ids: List[str], model_config: ModelConfig, db) -> RAGResponse:
        """Execute a RAG query and generate an answer"""
        try:
            # 1. Get retrieved chunks using our existing methods
            query_analysis = await self._analyze_query(query, model_config)
            broader_query = await self._generate_step_back_query(query, model_config)

            # Execute all queries in parallel
            queries = [query, broader_query] + [sq.question for sq in query_analysis.sub_questions]
            all_results = []

            for q in queries:
                # Get dense and sparse retrievals
                dense_results = await self._dense_search(q, model_config)
                sparse_results = self._sparse_search(q)

                # Combine results using reciprocal rank fusion
                combined = self._reciprocal_rank_fusion(dense_results, sparse_results)
                all_results.extend(combined)

            # Deduplicate and rank final results
            final_results = self._deduplicate_results(all_results)[:20]  # Top 20 unique results

            # 2. Generate answer with citations and images
            return await self.generate_answer(query, final_results, model_config)

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    async def _analyze_query(self, query: str, model_config: ModelConfig) -> QueryAnalysis:
        """Analyze and decompose the query"""
        provider = await self.llm_service.get_provider(model_config)
        prompt = self.query_analysis_prompt.format(query=query)
        messages = [{"role": "user", "content": prompt}]

        response = ""
        async for chunk in provider.generate(messages, model_config):
            response += chunk

        return self.query_analyzer.parse(response)

    async def _generate_step_back_query(self, query: str, model_config: ModelConfig) -> str:
        """Generate a step-back query for broader context"""
        provider = await self.llm_service.get_provider(model_config)
        prompt = self.step_back_prompt.format(query=query)
        messages = [{"role": "user", "content": prompt}]

        response = ""
        async for chunk in provider.generate(messages, model_config):
            response += chunk

        return response.strip()

    async def _get_embeddings(self, texts: List[str], model_config: ModelConfig) -> List[List[float]]:
        """Get embeddings for texts using the specified provider"""
        provider = await self.llm_service.get_provider(model_config)
        embeddings = []
        for text in texts:
            embedding = await provider.get_embeddings(text, model_config)
            embeddings.append(embedding)
        return embeddings

    async def _dense_search(self, query: str, model_config: ModelConfig) -> List[Document]:
        """Execute dense retrieval search"""
        query_embedding = await self._get_embeddings([query], model_config)
        return await self.chroma_provider.query(query_embedding[0])

    def _sparse_search(self, query: str) -> List[tuple[str, float]]:
        """Execute sparse (BM25) search"""
        return self.bm25_retriever.query(query)

    def _reciprocal_rank_fusion(
            self,
            dense_results: List[Document],
            sparse_results: List[tuple[str, float]],
            k: int = 60
    ) -> List[Document]:
        """Combine dense and sparse results using RRF"""
        # Create a map for scoring
        doc_scores = {}

        # Score dense results
        for rank, doc in enumerate(dense_results):
            doc_scores[doc.page_content] = 1 / (k + rank + 1)

        # Score sparse results
        for rank, (text, _) in enumerate(sparse_results):
            if text in doc_scores:
                doc_scores[text] += 1 / (k + rank + 1)
            else:
                doc_scores[text] = 1 / (k + rank + 1)

        # Sort by final scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Convert back to Documents
        return [doc for doc, _ in sorted_docs]

    def _deduplicate_results(self, results: List[Document]) -> List[Document]:
        """Remove duplicate results while preserving order"""
        seen = set()
        deduped = []

        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                deduped.append(doc)

        return deduped

    async def generate_answer(
            self,
            query: str,
            retrieved_chunks: List[Document],
            model_config: ModelConfig
    ) -> RAGResponse:
        """Generate a complete answer with citations and images"""
        # Extract images associated with retrieved chunks
        images = self._extract_images_from_chunks(retrieved_chunks)

        # Merge sequential chunks if they're from the same page/section
        merged_chunks = self._merge_sequential_chunks(retrieved_chunks)

        # Format context for the LLM
        formatted_context = self._format_context_with_citations(merged_chunks)
        formatted_images = self._format_image_references(images)

        # Generate the answer
        provider = await self.llm_service.get_provider(model_config)
        prompt = ANSWER_GENERATION_PROMPT.format(
            context=formatted_context,
            question=query,
            images=formatted_images
        )

        messages = [{"role": "user", "content": prompt}]
        response = ""
        async for chunk in provider.generate(messages, model_config):
            response += chunk

        # Parse the response and extract citations
        citations = self._extract_citations(response, merged_chunks)

        # Create structured response
        parts = response.split("Reasoning:", 1)
        answer = parts[0].replace("Answer:", "").strip()

        confidence = 0.0
        reasoning = None

        if len(parts) > 1:
            reasoning_parts = parts[1].split("Confidence:", 1)
            reasoning = reasoning_parts[0].strip()
            if len(reasoning_parts) > 1:
                try:
                    confidence = float(reasoning_parts[1].strip())
                except ValueError:
                    confidence = 0.7  # Default if parsing fails

        return RAGResponse(
            answer=answer,
            citations=citations,
            images=list(images),
            reasoning=reasoning,
            confidence_score=confidence
        )

    def _extract_images_from_chunks(
            self,
            chunks: List[Document]
    ) -> List[ImageReference]:
        """Extract and deduplicate images from chunks"""
        seen_images = set()
        images = []

        for chunk in chunks:
            if hasattr(chunk, 'metadata') and 'images' in chunk.metadata:
                for img in chunk.metadata['images']:
                    image_id = f"{img['page_num']}_{img['image_index']}"
                    if image_id not in seen_images:
                        seen_images.add(image_id)
                        images.append(ImageReference(
                            image_id=image_id,
                            document_name=chunk.metadata['document_id'],
                            page_number=img['page_num'],
                            image_type=img['image_type'],
                            caption=img.get('caption')
                        ))
        return images

    def _merge_sequential_chunks(self, chunks: List[Document]) -> List[Document]:
        """Merge chunks that are sequential in the document"""
        if not chunks:
            return chunks

        merged = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            if (current_chunk.metadata['document_id'] == next_chunk.metadata['document_id'] and
                    current_chunk.metadata['page_num'] == next_chunk.metadata['page_num']):
                # Merge the chunks
                current_chunk.page_content += " " + next_chunk.page_content
                # Merge any images
                if 'images' in current_chunk.metadata and 'images' in next_chunk.metadata:
                    current_chunk.metadata['images'].extend(next_chunk.metadata['images'])
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk

        merged.append(current_chunk)
        return merged

    def _format_context_with_citations(self, chunks: List[Document]) -> str:
        """Format chunks with citation information for the LLM"""
        formatted_chunks = []
        for chunk in chunks:
            citation = f"[Doc: {chunk.metadata['document_id']}, Page {chunk.metadata['page_num']}]"
            formatted_chunks.append(f"{citation} {chunk.page_content}")
        return "\n\n".join(formatted_chunks)

    def _format_image_references(self, images: List[ImageReference]) -> str:
        """Format image references for the LLM prompt"""
        return "\n".join(
            f"[Image {img.image_id}] - {img.image_type.capitalize()} on page {img.page_number}" +
            (f" - {img.caption}" if img.caption else "")
            for img in images
        )

    def _extract_citations(
            self,
            response: str,
            chunks: List[Document]
    ) -> List[Citation]:
        """Extract and validate citations from the response"""
        citations = []
        # Simple regex might not be enough here - might need more sophisticated parsing
        for chunk in chunks:
            if chunk.page_content in response:
                # Get a snippet of the quote
                quote_words = chunk.page_content.split()
                quote_start = " ".join(quote_words[:3])
                quote_end = " ".join(quote_words[-3:])

                citations.append(Citation(
                    document_name=chunk.metadata['document_id'],
                    page_number=chunk.metadata['page_num'],
                    section=chunk.metadata.get('section'),
                    text=chunk.page_content,
                    quote_start=quote_start,
                    quote_end=quote_end
                ))
        return citations
