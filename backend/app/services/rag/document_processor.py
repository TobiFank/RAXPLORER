import logging
import os
import re
from typing import List, Tuple
from uuid import uuid4

import fitz
from unstructured.partition.pdf import partition_pdf

from .rag_dependencies import DocumentSection, DocumentImage, BoundingBox, SectionType

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing and image-text association"""

    def __init__(self):
        self.image_pattern = re.compile(
            r'(?i)(?:figure|fig\.|image|img|picture|pic\.|diagram|chart|graph|illustration)[\s.-]*(\d+(?:\.\d+)?)|' +
            r'\[(?:figure|fig\.|image|img|picture|pic\.|diagram|chart|graph|illustration)[\s.-]*(\d+(?:\.\d+)?)\]'
        )

    def process_pdf(self, pdf_path: str, file_id: str) -> Tuple[List[DocumentSection], List[DocumentImage]]:
        """Process a PDF document and return sections and images"""
        # Extract layout elements using Unstructured
        elements = partition_pdf(pdf_path)

        # Open with PyMuPDF for image extraction and coordinate information
        pdf_document = fitz.open(pdf_path)

        try:
            # Extract and process images first
            images = self._extract_images(pdf_document, file_id)

            # Process text sections with spatial information
            sections = self._process_sections(elements, pdf_document)

            # Associate images with sections
            self._associate_images_with_sections(images, sections)

            return sections, images

        finally:
            pdf_document.close()

    def _extract_images(self, pdf_document: fitz.Document, file_id: str) -> List[DocumentImage]:
        """Extract images with spatial information"""
        images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images()

            for img_index, img_info in enumerate(image_list):
                try:
                    # First extract image with original working approach
                    xref = img_info[0]
                    pix = fitz.Pixmap(pdf_document, xref)

                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    # Save image first
                    image_filename = f"{file_id}_{page_num}_{img_index}.png"
                    image_path = f"storage/images/{image_filename}"
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    pix.save(image_path)

                    # Now get spatial information
                    rects = page.get_image_rects(xref)
                    image_rect = rects[0] if rects else page.rect  # Fallback to page rect if no specific rect

                    image_bbox = BoundingBox(
                        x0=image_rect.x0,
                        y0=image_rect.y0,
                        x1=image_rect.x1,
                        y1=image_rect.y1,
                        page_num=page_num
                    )

                    # Find caption after successful image handling
                    caption = self._find_image_caption(page, image_rect)

                    image = DocumentImage(
                        image_data=pix.samples,
                        page_num=page_num,
                        image_type=self._determine_image_type(caption),
                        bbox=image_bbox,
                        caption=caption,
                        metadata={
                            'image_index': img_index,
                            'file_path': image_path,
                            'width': pix.width,
                            'height': pix.height,
                            'has_alpha': pix.alpha > 0
                        },
                        referenced_by=[]
                    )

                    images.append(image)
                    pix = None

                except Exception as e:
                    logger.error(f"Failed to process image {img_index} on page {page_num}: {str(e)}")
                    continue

        return images

    def _process_sections(self, elements: List, pdf_document: fitz.Document) -> List[DocumentSection]:
        """Process document sections with spatial awareness"""
        sections = []

        for element in elements:
            try:
                # Get element coordinates from Unstructured metadata
                coords = element.metadata.coordinates
                if coords and hasattr(coords, 'points'):
                    points = coords.points
                    bbox = BoundingBox(
                        x0=points[0][0],  # Top-left x
                        y0=points[0][1],  # Top-left y
                        x1=points[2][0],  # Bottom-right x
                        y1=points[2][1],  # Bottom-right y
                        page_num=element.metadata.page_number
                    )

                # Determine section type
                section_type = self._determine_section_type(element)

                section = DocumentSection(
                    content=str(element.text),
                    bbox=bbox,
                    section_type=section_type,
                    metadata={
                        'section_id': str(uuid4()),
                        'page_num': bbox.page_num,
                        'document_id': getattr(element.metadata, 'document_id', None),
                        'filename': getattr(element.metadata, 'filename', None)
                    }
                )

                sections.append(section)

            except Exception as e:
                logger.error(f"Failed to process section: {str(e)}")
                continue

        # Link nearby sections
        self._link_nearby_sections(sections)

        return sections

    def _associate_images_with_sections(self, images: List[DocumentImage], sections: List[DocumentSection]):
        """Associate images with relevant sections using multiple strategies"""
        for image in images:
            # Strategy 1: Direct spatial overlap
            for section in sections:
                if section.overlaps_with(image.bbox):
                    section.images.append(image)
                    image.referenced_by.append(section.id)

            # Strategy 2: Caption proximity
            if image.caption:
                for section in sections:
                    if section.is_nearby(image.bbox) and section.section_type == SectionType.CAPTION:
                        section.images.append(image)
                        image.referenced_by.append(section.id)

            # Strategy 3: Text references
            image_numbers = self._extract_all_image_numbers(image.caption) if image.caption else []
            if image_numbers:
                for section in sections:
                    if any(self._contains_image_reference(section.content, num) for num in image_numbers):
                        section.images.append(image)
                        image.referenced_by.append(section.id)

            # Strategy 4: Context association
            for section in sections:
                if section.is_nearby(image.bbox, distance_threshold=100):  # Larger threshold for context
                    section.images.append(image)
                    image.referenced_by.append(section.id)

    def _find_image_caption(self, page: fitz.Page, image_rect: fitz.Rect) -> str:
        caption_area = image_rect + (0, 0, 0, 50)  # Look 50 points below
        caption_text = page.get_textbox(caption_area)

        if not caption_text:
            # Look above if nothing below
            caption_area = image_rect + (0, -50, 0, 0)
            caption_text = page.get_textbox(caption_area)

        return caption_text.strip()

    def _determine_image_type(self, caption: str) -> str:
        """Determine image type based on caption content"""
        if not caption:
            return "image"

        caption_lower = caption.lower()
        if "table" in caption_lower:
            return "table"
        elif "figure" in caption_lower or "fig." in caption_lower:
            return "figure"
        elif "diagram" in caption_lower:
            return "diagram"

        return "image"

    def _determine_section_type(self, element) -> SectionType:
        """Determine section type from Unstructured element"""
        element_type = type(element).__name__.lower()

        coordinates = getattr(element.metadata, 'coordinates', None)

        if "title" in element_type:
            return SectionType.TITLE
        elif "heading" in element_type:
            return SectionType.HEADING
        elif "list" in element_type:
            return SectionType.LIST
        elif "table" in element_type:
            return SectionType.TABLE
        elif "text" in element_type and coordinates:
            # Check if it might be a caption
            text = str(element.text).lower().strip()
            if text.startswith(("figure", "fig.", "table", "image")):
                return SectionType.CAPTION

        return SectionType.PARAGRAPH

    def _link_nearby_sections(self, sections: List[DocumentSection]):
        """Link sections that are spatially close to each other"""
        for i, section in enumerate(sections):
            for other in sections[i + 1:]:
                if section.is_nearby(other):
                    section.nearby_sections.append(other)
                    other.nearby_sections.append(section)

    def _extract_all_image_numbers(self, text: str) -> List[str]:
        """Extract all image numbers from text"""
        if not text:
            return []
        matches = self.image_pattern.finditer(text)
        return [m.group(1) or m.group(2) for m in matches if m.group(1) or m.group(2)]

    def _contains_image_reference(self, text: str, image_number: str) -> bool:
        """Check if text contains reference to image number"""
        if not text or not image_number:
            return False

        matches = self.image_pattern.finditer(text)
        for match in matches:
            if match.group(1) == image_number or match.group(2) == image_number:
                return True
        return False
