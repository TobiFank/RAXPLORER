# app/services/rag.py
import logging
import os
import re
from typing import List, Tuple

import fitz
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from sqlalchemy import select

from .llm import LLMService
from .rag_dependencies import (
    ChromaProvider, BM25Retriever, DocumentSection,
    DocumentImage
)
from ..db.models import FileModel
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
1. Answer the question using ONLY information from the provided context
2. Use logical reasoning to connect information
3. You MUST cite ALL sources using [Doc: ID, Page X] format IMMEDIATELY after each piece of information
4. Reference relevant images using [Image X] format
5. Maintain high confidence in your answer
6. Be explicit about which document each piece of information comes from

Example citation format:
"John works at ABC Corp [Doc: doc123, Page 1] and has 5 years of experience [Doc: doc456, Page 3]"

Answer in this structured format:
Answer: [Your detailed answer with inline citations]
Reasoning: [Your step-by-step reasoning process]
Confidence: [Score between 0-1 based on context relevance]

If any of the retrieved chunks have associated images that would help explain the concept, 
include a reference to them in your answer using [Image X] notation. For example:
- When explaining a diagram: "As we can see in [Image 1], the process flows from..."
- When an image provides evidence: "The document shows this clearly in [Image 2]"
Only reference images that are directly relevant to answering the question."""


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

    async def process_document(self, file_content: bytes, file_model: FileModel, model_config: ModelConfig):
        """Process a document and store its embeddings and metadata"""
        try:
            # Get original filename from database
            file_id = file_model.vector_store_id
            original_filename = file_model.name

            # Open PDF with PyMuPDF
            logger.info(f"Processing document {file_model.name}")
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

                    # Determine the proper extension
                    extension = base_image.get('ext', 'png')
                    image_path = f"storage/images/{file_id}_{page_num}_{image_index}.{extension}"

                    # Ensure the storage directory exists
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)

                    # Save the image
                    with open(image_path, "wb") as f:
                        f.write(base_image["image"])

                    image = DocumentImage(
                        image_data=base_image["image"],
                        page_num=page_num,
                        image_type="image",
                        metadata={
                            "page_num": page_num,
                            "image_index": image_index,
                            "extension": extension,
                            "file_path": image_path
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
                            "section_type": "text",
                            "name": original_filename,
                            "file_path": file_model.file_path
                        }
                    )
                    sections.append(section)

            # Create embeddings
            texts = [section.content for section in sections]
            embeddings = await self._get_embeddings(texts, model_config)

            # Store in ChromaDB - sections already contain metadata
            await self.chroma_provider.store_embeddings(sections, embeddings)

            # Create Document objects for BM25
            documents = [
                Document(
                    page_content=section.content,
                    metadata=section.metadata
                )
                for section in sections
            ]

            # Update BM25 index with Documents
            self.bm25_retriever.fit(documents)

            return True

        except Exception as e:
            import traceback
            logger.error(f"Document processing failed: {str(e)}\n{traceback.format_exc()}")
            raise

    async def query(self, query: str, file_ids: List[str], model_config: ModelConfig, db) -> RAGResponse:
        """Execute a RAG query and generate an answer"""
        try:

            # If no documents, fall back to direct LLM response
            if not file_ids:
                provider = await self.llm_service.get_provider(model_config)

                prompt = f"Please provide a response to this query: {query}\n\nNote: Respond directly, mentioning that you don't have any specific documents or context to refer to, but use your model knowledge instead."

                messages = [{"role": "user", "content": prompt}]
                response = ""
                async for chunk in provider.generate(messages, model_config):
                    response += chunk

                return RAGResponse(
                    answer=response,
                    citations=[],
                    images=[],
                    reasoning="Direct response without document context",
                    confidence_score=0.7  # Lower confidence since no source documents
                )

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
        query_embedding = await self._get_embeddings([query], model_config)
        try:
            return await self.chroma_provider.query(query_embedding[0])
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []  # Fallback to empty result but continue with sparse search

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

        # Post-process response and add references
        processed_text, references_text, referenced_image_ids = self._process_citations(answer, citations)
        final_answer = processed_text + references_text

        referenced_images = [img for img in images if img.image_id in referenced_image_ids]

        logger.info(f"Final answer: {final_answer}")

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

                logger.info(f"Chunk metadata: {chunk.metadata}")
                citations.append(Citation(
                    document_name=chunk.metadata['name'],
                    page_number=page_num,
                    text=text,
                    quote_start=quote_start,
                    quote_end=quote_end,
                    file_path=chunk.metadata.get('file_path')
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
                        logger.info(f"Chunk metadata: {chunk.metadata}")
                        citations.append(Citation(
                            document_name=chunk.metadata['name'],
                            page_number=chunk.metadata['page_num'],
                            text=chunk.page_content,
                            quote_start=sentence[:50],
                            quote_end=sentence[-50:] if len(sentence) > 50 else sentence,
                            file_path=chunk.metadata.get('file_path')
                        ))
                        break  # One citation per chunk is enough

        logger.info(f"Found citations: {citations}")
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

        for match in matches:
            doc_name = match.group(1)
            page = int(match.group(2))
            key = (doc_name, page)

            if key not in citation_map:
                citation_map[key] = current_citation
                current_citation += 1

            processed_text = processed_text.replace(match.group(0), f'[{citation_map[key]}]')

        # Build references section
        references = []
        for (doc_name, page), number in sorted(citation_map.items(), key=lambda x: x[1]):
            citation = next((c for c in citations if c.document_name == doc_name and c.page_number == page), None)
            if citation:
                # Include file path for clickable reference
                ref_text = f"[{number}] {citation.document_name}, Page {page}"
                if citation.file_path:
                    ref_text += f" [View Document]({citation.file_path})"
                if citation.quote_start and citation.quote_end:
                    ref_text += f"\nQuote: \"{citation.quote_start}...{citation.quote_end}\""
                references.append(ref_text)

        references_text = "\n\nReferences:\n" + "\n\n".join(references) if references else ""

        # Track referenced images
        referenced_image_ids = []
        image_pattern = r'\[Image ([^\]]+)\]'
        for match in re.finditer(image_pattern, processed_text):
            image_id = match.group(1)
            referenced_image_ids.append(image_id)

        logger.info(f"Built references text: {references_text}")

        return processed_text, references_text, referenced_image_ids