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
        Enrich sections with contextual information using the configured LLM provider.

        Args:
            sections: List of document sections to enrich
            full_text: The complete document text for context
            model_config: Configuration for the LLM provider

        Returns:
            List of enriched document sections
        """
        try:
            provider = await self.llm_service.get_provider(model_config)
            enriched_sections = []

            for section in sections:
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

                logger.debug(f"Enriched section content: {enriched_section.content}")

                # Copy over any existing relationships and images
                enriched_section.images = section.images
                enriched_section.nearby_sections = section.nearby_sections

                enriched_sections.append(enriched_section)

            return enriched_sections

        except Exception as e:
            logger.error(f"Error enriching sections: {str(e)}")
            # If enrichment fails, return original sections
            return sections

    async def _get_cached_provider(self, provider_type: Provider) -> Optional[any]:
        """Get cached provider instance if available"""
        return self._provider_cache.get(provider_type)