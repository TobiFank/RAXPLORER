# app/services/rag.py
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple

from fastapi import HTTPException
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .document_processor import DocumentProcessor
from .rag_dependencies import (
    ChromaProvider, BM25Retriever, DocumentSection
)
from ..llm import LLMService
from ...core.config import Settings
from ...db.models import FileModel
from ...schemas import file
from ...schemas.model import ModelConfig, Provider
from ...schemas.rag import RAGResponse, ImageReference, Citation

logger = logging.getLogger(__name__)


class SubQuery(BaseModel):
    """Model for decomposed queries"""
    query: str = Field(description="The sub-queries to be answered")
    reasoning: str = Field(description="Why this sub-query is relevant")


class QueryAnalysis(BaseModel):
    """Model for query analysis output"""
    main_intent: str = Field(description="The main intent of the query")
    sub_queries: List[SubQuery] = Field(description="List of sub-queries to answer")


QUERY_ANALYSIS_PROMPT = """Analyze this query and break it down into sub-queries.
Main Query: {query}

Please respond with a JSON object that contains:
1. A "main_intent" field with a string describing the core purpose of the query
2. A "sub_queries" array containing objects with "queries" and "reasoning" fields

Example response format:
{{"main_intent": "understand what the user means by X", "sub_queries": [{{"query": "what is X?", "reasoning": "need to clarify the basic concept"}}]}}

Remember: Respond ONLY with the JSON object, no other text or schema information.

Query to analyze: {query}"""

STEP_BACK_PROMPT = """Before directly answering the query, let's take a step back and think about the broader context.
Query: {query}

What broader topics or concepts should we consider to provide a more comprehensive answer?
Focus on generating a more general query that will help retrieve relevant context."""

ANSWER_GENERATION_PROMPT = """Given the following context and query, provide a comprehensive answer. Use the context carefully and cite your sources.

Context:
{context}

Query: {query}

Images Available:
{images}

Instructions:
1. Answer the query using ONLY information from the provided context
2. Use logical reasoning to connect information
3. Cite ALL sources using [Doc: ID, Page X] format IMMEDIATELY after each piece of information
4. When referencing images, use (Figure X) format where X matches the image caption
5. Maintain high confidence in your answer
6. Be explicit about which document each piece of information comes from

Example citation format:
"As shown in (Figure 1), John works at ABC Corp [Doc: doc123, Page 1] and has 5 years of experience [Doc: doc456, Page 3]"

Answer in this structured format:
Answer: [Your detailed answer with inline citations]
Reasoning: [Your step-by-step reasoning process]
Confidence: [Score between 0-1 based on context relevance]

If any of the retrieved chunks have associated images that would help explain the concept, 
include a reference to them in your answer using [Image X] notation. For example:
- When explaining a diagram: "As we can see in [Image 1], the process flows from..."
- When an image provides evidence: "The document shows this clearly in [Image 2]"
Only reference images that are directly relevant to answering the query."""


class RAGService:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.chroma_provider = ChromaProvider()
        self.bm25_retriever = BM25Retriever()
        self.settings = Settings()

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

    async def process_document(self, file_content: bytes, file_model: FileModel, model_config: ModelConfig):
        try:
            temp_pdf_path = f"storage/documents/{file_model.id}.pdf"
            os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)
            with open(temp_pdf_path, 'wb') as f:
                f.write(file_content)

            file_id = file_model.id
            original_filename = file_model.name
            logger.info(f"Processing document {original_filename}")

            processor = DocumentProcessor()
            sections = processor.process_pdf(temp_pdf_path, file_id, file_model.name)

            # Enhance section metadata with image relationships
            for section in sections:
                if section.images:
                    section.metadata['images'] = [{
                        'image_id': f"{img.page_num}_{img.metadata['image_index']}",
                        'image_type': img.image_type,
                        'caption': img.caption,
                        'file_path': img.metadata['file_path'],
                        'bbox': img.bbox.__dict__
                    } for img in section.images]

            # Process with valid providers
            valid_providers = self._get_valid_providers(model_config)

            if not valid_providers:
                raise HTTPException(
                    status_code=400,
                    detail="No valid providers found. Please configure at least one provider."
                )

            # Process with each provider
            for provider in valid_providers:
                try:
                    logger.info(f"Processing with provider: {provider}")
                    provider_config = self._create_provider_config(provider, model_config)
                    provider_service = await self.llm_service.get_provider(provider_config)

                    embeddings = await self._generate_embeddings(sections, provider_service, provider_config)
                    await self.chroma_provider.store_embeddings(
                        provider=provider,
                        sections=sections,
                        embeddings=embeddings
                    )

                except Exception as e:
                    logger.error(f"Failed to process with provider {provider}: {str(e)}")
                    continue

            # Update BM25 index with enhanced metadata
            documents = [
                Document(
                    page_content=section.content,
                    metadata={
                        **section.metadata,
                        'section_type': section.section_type.value,
                        'nearby_sections': [s.id for s in section.nearby_sections]
                    }
                )
                for section in sections
            ]
            self.bm25_retriever.fit(documents)

            return True

        except Exception as e:
            await self._cleanup_on_error(file_id)
            logger.error(f"Document processing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def query(self, query: str, file_ids: List[str], model_config: ModelConfig, db) -> RAGResponse:
        """Execute a RAG query and generate an answer"""
        try:
            if not file_ids:
                return await self._handle_no_documents(query, model_config)

            provider = await self.llm_service.get_provider(model_config)

            # 1. Run analysis tasks in parallel
            async def run_llm(messages):
                response = ""
                async for chunk in provider.generate(messages, model_config):
                    response += chunk
                return response

            analysis_messages = [{"role": "user", "content": self.query_analysis_prompt.format(query=query)}]
            step_back_messages = [{"role": "user", "content": self.step_back_prompt.format(query=query)}]

            analysis_response, step_back_response = await asyncio.gather(
                run_llm(analysis_messages),
                run_llm(step_back_messages)
            )

            # 2. Parse results with validation
            try:
                query_analysis = self.query_analyzer.parse(analysis_response)
                broader_query = step_back_response.strip()

                if not query_analysis.sub_queries:
                    logger.warning("No sub-queries generated, using original query")
                    query_analysis.sub_queries = [SubQuery(query=query, reasoning="Original query")]
            except Exception as e:
                logger.error(f"Failed to parse query analysis: {e}")
                query_analysis = QueryAnalysis(
                    main_intent="Direct query processing",
                    sub_queries=[SubQuery(query=query, reasoning="Fallback to original query")]
                )
                broader_query = query

            # 3. Prepare all queries for embedding
            all_queries = [
                query,  # Original query first
                broader_query,  # Step-back query second
                *[sq.query for sq in query_analysis.sub_queries]  # Sub-queries last
            ]

            # 4. Run embeddings generation and sparse search preparation in parallel
            async def get_embeddings():
                return await provider.get_embeddings_batch(all_queries, model_config)

            def prepare_sparse_search():  # Remove async, this is CPU-bound work
                return [self._sparse_search(q) for q in all_queries]

            # Run embedding generation and sparse search prep in parallel
            embeddings, sparse_results = await asyncio.gather(
                get_embeddings(),
                asyncio.to_thread(prepare_sparse_search)  # Now correctly runs CPU-bound function in thread
            )

            # 5. Run dense search
            dense_results = await self.chroma_provider.query_batch(
                provider=model_config.provider,
                query_embeddings=embeddings,
                top_k=5
            )

            # 6. Combine results maintaining query alignment
            all_results = []
            for query_idx in range(len(all_queries)):
                dense_batch = dense_results[query_idx] if query_idx < len(dense_results) else []
                sparse_batch = sparse_results[query_idx] if query_idx < len(sparse_results) else []
                combined = self._reciprocal_rank_fusion(dense_batch, sparse_batch)
                all_results.extend(combined)

            # 7. Deduplicate while preserving most relevant results
            final_results = self._deduplicate_results(all_results)[:20]

            # 8. Generate final answer
            return await self.generate_answer(query, final_results, model_config)

        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise

    async def _handle_no_documents(self, query: str, model_config: ModelConfig) -> RAGResponse:
        provider = await self.llm_service.get_provider(model_config)
        response = ""
        messages = [{"role": "user", "content": f"Please provide a response to this query: {query}\n\nNote: No specific documents available."}]
        async for chunk in provider.generate(messages, model_config):
            response += chunk

        return RAGResponse(
            answer=response,
            citations=[],
            images=[],
            reasoning="Direct response without document context",
            confidence_score=0.7
        )

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
        """Execute dense search using selected provider"""
        provider = await self.llm_service.get_provider(model_config)
        query_embedding = await provider.get_embeddings(query, model_config)
        try:
            return await self.chroma_provider.query(
                provider=model_config.provider,
                query_embedding=query_embedding,
                top_k=5
            )
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    def _sparse_search(self, query: str) -> List[Document]:
        """Execute sparse (BM25) search"""
        results = self.bm25_retriever.query(query)
        return [doc for doc, _ in results]  # Return just the Documents

    def _reciprocal_rank_fusion(
            self,
            dense_results: List[Document],
            sparse_results: List[Document],  # Updated type
            k: int = 60
    ) -> List[Document]:
        """Combine dense and sparse results using RRF"""
        # Create a map for scoring
        doc_scores = {}

        # Score dense results
        for rank, doc in enumerate(dense_results):
            key = (doc.page_content, doc.metadata.get('document_id', ''), doc.metadata.get('page_num', 0))
            doc_scores[key] = {'doc': doc, 'score': 1 / (k + rank + 1)}

        # Score sparse results
        for rank, doc in enumerate(sparse_results):
            key = (doc.page_content, doc.metadata.get('document_id', ''), doc.metadata.get('page_num', 0))
            if key in doc_scores:
                doc_scores[key]['score'] += 1 / (k + rank + 1)
            else:
                doc_scores[key] = {'doc': doc, 'score': 1 / (k + rank + 1)}

        # Sort by final scores
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)

        # Return Documents
        return [item['doc'] for item in sorted_docs]

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
            query=query,
            images=formatted_images
        )

        messages = [{"role": "user", "content": prompt}]
        response = ""
        async for chunk in provider.generate(messages, model_config):
            response += chunk

        # Parse the response and extract citations
        citations = self._extract_citations(response, merged_chunks)
        logger.debug(f"Extracted citations: {citations}")

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

        # Post-process response and add references
        processed_text, references_text, referenced_image_ids = self._process_citations(answer, citations)
        final_answer = processed_text + references_text

        referenced_images = [img for img in images if img.image_id in referenced_image_ids]

        logger.debug(f"Final answer: {final_answer}")

        return RAGResponse(
            answer=final_answer,
            citations=citations,  # Original citations with file paths for frontend
            images=referenced_images,
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
                            caption=img.get('caption'),
                            file_path=img.get('file_path')
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

    def _extract_citations(self, response: str, chunks: List[Document]) -> List[Citation]:
        """Extract and validate citations from the response"""
        citations = []

        # First look for explicit citations in the format [Doc: X, Page Y]
        import re
        citation_pattern = r'\[Doc: ([^,]+), Page (\d+)\]'
        explicit_citations = re.finditer(citation_pattern, response)

        # Create a map of document_id -> chunk for easier lookup
        chunk_map = {
            chunk.metadata['document_id']: chunk
            for chunk in chunks
        }

        # Process explicit citations
        for match in explicit_citations:
            doc_id = match.group(1)
            page_num = int(match.group(2))

            if doc_id in chunk_map:
                chunk = chunk_map[doc_id]
                # Get meaningful sentences for quotes
                text = chunk.page_content
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
                quote_start = sentences[0] if sentences else ""
                quote_end = sentences[-1] if sentences else ""

                logger.debug(f"Chunk metadata: {chunk.metadata}")
                citations.append(Citation(
                    document_name=chunk.metadata['name'],
                    page_number=page_num,
                    text=text,
                    quote_start=quote_start,
                    quote_end=quote_end,
                    file_path=chunk.metadata.get('file_path'),
                    metadata={'document_id': doc_id}
                ))

        # If no explicit citations found, try to match content
        if not citations:
            for chunk in chunks:
                logger.info(f"Checking chunk {chunk}")
                # Look for significant portions of the chunk content in the response
                # Split into sentences and look for matches
                chunk_sentences = chunk.page_content.split('.')
                for sentence in chunk_sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20 and sentence in response:  # Only match substantial sentences
                        logger.debug(f"Chunk metadata: {chunk.metadata}")
                        citations.append(Citation(
                            document_name=chunk.metadata['name'],
                            page_number=chunk.metadata['page_num'],
                            text=chunk.page_content,
                            quote_start=sentence[:50],
                            quote_end=sentence[-50:] if len(sentence) > 50 else sentence,
                            file_path=chunk.metadata.get('file_path'),
                            metadata={'document_id': chunk.metadata['document_id']}
                        ))
                        break  # One citation per chunk is enough

        logger.debug(f"Found citations: {citations}")
        return citations

    def _process_citations(self, response: str, citations: List[Citation]) -> Tuple[str, str, List[str]]:
        """
        Process response to replace citations and track used images
        Returns: (processed_text, references_text, referenced_image_ids)
        """
        # First pass: Replace citations with numerical references
        processed_text = response
        citation_map = {}  # (doc_name, page) -> number
        current_citation = 1

        # Find all citation patterns and replace with numbers
        citation_pattern = r'\[Doc: ([^,]+), Page (\d+)\]'
        matches = list(re.finditer(citation_pattern, response))

        logger.debug(f"Found citation matches: {matches}")

        for match in matches:
            doc_id = match.group(1)  # This is the UUID
            page = int(match.group(2))
            key = (doc_id, page)

            if key not in citation_map:
                citation_map[key] = current_citation
                current_citation += 1

            processed_text = processed_text.replace(match.group(0), f'[{citation_map[key]}]')

        logger.debug(f"Processed text after replacing citations: {processed_text}")
        logger.debug(f"Citation map: {citation_map}")

        # Build references section
        references = []
        for (doc_id, page), number in sorted(citation_map.items(), key=lambda x: x[1]):
            # Find citation by matching document_id in metadata
            citation = next((c for c in citations
                             if c.text and 'document_id' in c.metadata
                             and c.metadata['document_id'] == doc_id
                             and c.page_number == page), None)

            if citation:
                # Use the friendly name for display
                ref_text = f"[{number}] {citation.document_name}, Page {page}"
                if citation.file_path:
                    api_url = "http://localhost:8000"  # Or get from settings if you prefer
                    web_path = f"{api_url}/storage/documents/{os.path.basename(citation.file_path)}"
                    ref_text += f" [View Document]({web_path}#page={page})"
                # if citation.quote_start and citation.quote_end:
                #    ref_text += f"\nQuote: \"{citation.quote_start}...{citation.quote_end}\""
                references.append(ref_text)

        references_text = "\n\nReferences:\n" + "\n\n".join(references) if references else ""

        # Track referenced images
        referenced_image_ids = []
        image_pattern = r'\[Image ([^\]]+)\]'
        for match in re.finditer(image_pattern, processed_text):
            image_id = match.group(1)
            referenced_image_ids.append(image_id)

        logger.debug(f"Built references text: {references_text}")

        return processed_text, references_text, referenced_image_ids

    def _get_valid_providers(self, model_config: ModelConfig) -> List[Provider]:
        valid_providers = []
        for provider in Provider:
            try:
                provider_config = self._create_provider_config(provider, model_config)
                if self._is_provider_configured(provider, provider_config):
                    valid_providers.append(provider)
            except Exception as e:
                logger.error(f"Error validating provider {provider}: {str(e)}")
        return valid_providers

    def _create_provider_config(self, provider: Provider, base_config: ModelConfig) -> ModelConfig:
        return ModelConfig(
            provider=provider,
            model=base_config.model,
            temperature=base_config.temperature,
            systemMessage=base_config.systemMessage,
            apiKey=base_config.apiKey
        )

    def _is_provider_configured(self, provider: Provider, config: ModelConfig) -> bool:
        if provider in [Provider.CLAUDE, Provider.CHATGPT]:
            return bool(config.apiKey)
        if provider == Provider.OLLAMA:
            return bool(self.settings.OLLAMA_HOST)
        return False

    async def _generate_embeddings(self, sections: List[DocumentSection],
                                   provider_service, provider_config) -> List[List[float]]:
        # Extract all texts first
        texts = [section.content for section in sections]

        # Default batch size - can be moved to settings if needed
        batch_size = 50

        try:
            # Check if provider supports batch processing
            if hasattr(provider_service, 'get_embeddings_batch'):
                return await provider_service.get_embeddings_batch(texts, provider_config)
            else:
                # Fallback to non-batch processing
                logger.warning("Provider doesn't support batch processing, falling back to single processing")
                embeddings = []
                for section in sections:
                    embedding = await provider_service.get_embeddings(section.content, provider_config)
                    embeddings.append(embedding)
                return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate embeddings: {str(e)}")

    async def _cleanup_on_error(self, file_id: str):
        try:
            image_dir = Path("storage/images")
            if image_dir.exists():
                for image_file in image_dir.glob(f"{file_id}_*"):
                    image_file.unlink(missing_ok=True)
        except Exception as cleanup_error:
            logger.error(f"Cleanup after failure failed: {str(cleanup_error)}")
