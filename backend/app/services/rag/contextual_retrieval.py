import asyncio
import logging
from typing import List, Optional

from ...schemas.model import ModelConfig, Provider
from ..llm import LLMService
from .rag_dependencies import DocumentSection

logger = logging.getLogger(__name__)

CONTEXT_GENERATION_PROMPT = """<document>
{document_text}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


class ContextualEnricher:
    """Handles contextual enrichment of document sections using configured LLM provider"""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self._provider_cache = {}

    async def enrich_sections(
            self,
            sections: List[DocumentSection],
            full_text: str,
            model_config: ModelConfig
    ) -> List[DocumentSection]:
        """
        Enrich sections with contextual information using the configured LLM provider in parallel.
        """
        try:
            provider = await self.llm_service.get_provider(model_config)

            # Create tasks for all sections
            enrichment_tasks = [
                self._enrich_single_section(section, full_text, provider, model_config)
                for section in sections
            ]

            # Run all enrichment tasks in parallel
            enriched_sections = await asyncio.gather(*enrichment_tasks)
            return enriched_sections

        except Exception as e:
            logger.error(f"Error enriching sections: {str(e)}")
            # If enrichment fails, return original sections
            return sections

    async def _enrich_single_section(
            self,
            section: DocumentSection,
            full_text: str,
            provider: any,
            model_config: ModelConfig
    ) -> DocumentSection:
        """Enrich a single section"""
        try:
            # Generate context for the section
            prompt = CONTEXT_GENERATION_PROMPT.format(
                document_text=full_text,
                chunk_content=section.content
            )

            messages = [{"role": "user", "content": prompt}]
            context = ""

            # Get context from provider
            async for chunk in provider.generate(messages, model_config):
                context += chunk

            # Create new enriched section
            enriched_section = DocumentSection(
                content=f"{context.strip()}\n\n{section.content}",
                bbox=section.bbox,
                section_type=section.section_type,
                metadata=section.metadata
            )

            # Copy over any existing relationships and images
            enriched_section.images = section.images
            enriched_section.nearby_sections = section.nearby_sections

            return enriched_section

        except Exception as e:
            logger.error(f"Error enriching single section: {str(e)}")
            return section

    async def _get_cached_provider(self, provider_type: Provider) -> Optional[any]:
        """Get cached provider instance if available"""
        return self._provider_cache.get(provider_type)