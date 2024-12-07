import logging
import os
import re
from typing import List, Tuple
from uuid import uuid4

import fitz
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode

from .rag_dependencies import DocumentSection, DocumentImage, BoundingBox, SectionType

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing and image-text association"""

    def __init__(self):
        self.image_pattern = re.compile(
            r'(?i)(?:figure|fig\.|image|img|picture|pic\.|diagram|chart|graph|illustration|' +
            r'abbildung|abb\.|bild|diagramm|grafik|schaubild|tabelle|tab\.|darstellung|illustration|ill\.)[\s.-]*(\d+(?:\.\d+)?)|' +
            r'\[(?:figure|fig\.|image|img|picture|pic\.|diagram|chart|graph|illustration|' +
            r'abbildung|abb\.|bild|diagramm|grafik|schaubild|tabelle|tab\.|darstellung|illustration|ill\.)[\s.-]*(\d+(?:\.\d+)?)\]'
        )
        self.node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 1024, 512],
            chunk_overlap=20,
            include_metadata=True,
            include_prev_next_rel=True
        )
        self.pdf_path = None

    def process_pdf(self, pdf_path: str, file_id: str, original_filename: str) -> List[DocumentSection]:
        self.pdf_path = pdf_path
        pdf_document = fitz.open(pdf_path)
        logger.info(f"PDF {original_filename} opened with {len(pdf_document)} pages")
        try:
            # Extract images first
            images = self._extract_images(pdf_document, file_id)

            # Extract text with layout info
            text_with_layout = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if b["type"] == 0:  # Text block
                        if "lines" in b:
                            # Extract text from lines structure
                            text = "\n".join(" ".join(span["text"] for span in line["spans"]) for line in b["lines"])
                            bbox = BoundingBox(
                                x0=b["bbox"][0], y0=b["bbox"][1],
                                x1=b["bbox"][2], y1=b["bbox"][3],
                                page_num=page_num
                            )
                            text_with_layout += f"[Page {page_num + 1}]\n{text}\n"

            # Create LlamaIndex document
            doc = Document(text=text_with_layout)

            # Parse into nodes
            nodes = self.node_parser.get_nodes_from_documents([doc])
            leaf_nodes = get_leaf_nodes(nodes)

            # Convert to our DocumentSection format
            sections = []
            for node in leaf_nodes:
                # Get spatial info from metadata
                page_match = re.search(r'\[Page (\d+)\]', node.text)
                page_num = int(page_match.group(1)) - 1 if page_match else 0
                logger.info(f"Processing node with text start: {node.text[:100]}...")
                logger.info(f"Extracted page number: {page_num + 1}")

                section = DocumentSection(
                    content=node.text,
                    bbox=self._estimate_bbox(node, page_num),
                    section_type=self._determine_section_type(node),
                    metadata={
                        'section_id': node.node_id,
                        'page_num': page_num,
                        'document_id': file_id,
                        'name': original_filename,
                        'file_path': pdf_path,
                        'images': []
                    }
                )
                sections.append(section)

            # Associate images with sections
            self._associate_images(images, sections)

            # Create relationships between sections
            self._link_sections(sections)

            return sections

        finally:
            pdf_document.close()

    def _extract_images(self, pdf_document: fitz.Document, file_id: str) -> List[DocumentImage]:
        """Extract images with spatial information"""
        images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images()

            logger.info(f"Starting image extraction for document {file_id}")
            logger.info(f"Found {len(image_list)} images on page {page_num}")

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
                    logger.info(f"Image {img_index} rect: {image_rect}")

                    image_bbox = BoundingBox(
                        x0=image_rect.x0,
                        y0=image_rect.y0,
                        x1=image_rect.x1,
                        y1=image_rect.y1,
                        page_num=page_num
                    )

                    # Find caption after successful image handling
                    caption = self._find_image_caption(page, image_rect)
                    logger.debug(f"Image {img_index} caption: {caption}")

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

        logger.info(f"Total images extracted: {len(images)}")

        return images

    def _associate_images(self, images: List[DocumentImage], sections: List[DocumentSection]):
        """Associate images with relevant sections using multiple strategies"""
        # Inside _associate_images method, at the start
        def add_image_metadata(section: DocumentSection, image: DocumentImage):
            """Helper to avoid duplicate image associations"""
            image_metadata = {
                'page_num': image.page_num,
                'image_index': image.metadata['image_index'],
                'image_type': image.image_type,
                'caption': image.caption,
                'file_path': image.metadata['file_path'],
                'bbox': image.bbox.__dict__
            }

            # Check if this exact image is already associated
            if not any(img.get('image_index') == image_metadata['image_index'] and
                       img.get('page_num') == image_metadata['page_num']
                       for img in section.metadata.get('images', [])):
                section.metadata.setdefault('images', []).append(image_metadata)
                image.referenced_by.append(section.metadata['section_id'])

        for image in images:
            # Strategy 1: Direct spatial overlap
            for section in sections:
                if section.overlaps_with(image.bbox):
                    add_image_metadata(section, image)

            # Strategy 2: Caption proximity
            if image.caption:
                for section in sections:
                    if section.is_nearby(image.bbox) and section.section_type == SectionType.CAPTION:
                        add_image_metadata(section, image)

            # Strategy 3: Text references
            image_numbers = self._extract_all_image_numbers(image.caption) if image.caption else []
            if image_numbers:
                for section in sections:
                    if any(self._contains_image_reference(section.content, num) for num in image_numbers):
                        add_image_metadata(section, image)

            # Strategy 4: Context association
            for section in sections:
                if section.is_nearby(image.bbox, distance_threshold=100):  # Larger threshold for context
                    add_image_metadata(section, image)

    def _find_image_caption(self, page: fitz.Page, image_rect: fitz.Rect) -> str:
        caption_area = image_rect + (0, 0, 0, 50)  # Look 50 points below
        caption_text = page.get_textbox(caption_area)
        logger.info(f"Looking for caption in area: {caption_area}")

        if not caption_text:
            # Look above if nothing below
            caption_area = image_rect + (0, -50, 0, 0)
            caption_text = page.get_textbox(caption_area)

        return caption_text.strip()

    def _determine_image_type(self, caption: str) -> str:
        if not caption:
            return SectionType.IMAGE.value

        caption_lower = caption.lower()
        if "table" in caption_lower or "tabelle" in caption_lower:
            return SectionType.TABLE.value
        elif any(term in caption_lower for term in ["figure", "fig.", "diagram", "abbildung", "abb.", "diagramm", "grafik", "schaubild"]):
            return SectionType.IMAGE.value

        return SectionType.IMAGE.value

    def _determine_section_type(self, node: BaseNode) -> SectionType:
        text = node.text.lower().strip()

        if any(text.startswith(h) for h in ['#', 'chapter', 'section', 'kapitel', 'abschnitt', 'teil']):
            return SectionType.HEADING
        elif text.startswith(('•', '-', '*')) or bool(re.match(r'^\d+\.', text)):
            return SectionType.LIST
        elif bool(re.search(r'\|\s*[-+]*\s*\|', text)):  # Basic table detection
            return SectionType.TABLE
        elif len(text.split()) < 20:
            return SectionType.TITLE

        return SectionType.PARAGRAPH

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

    def _link_sections(self, sections: List[DocumentSection]):
        for i, section in enumerate(sections):
            # Link to previous and next sections
            if i > 0:
                section.nearby_sections.append(sections[i - 1])
            if i < len(sections) - 1:
                section.nearby_sections.append(sections[i + 1])

    def _estimate_bbox(self, node: BaseNode, page_num: int) -> BoundingBox:
        """Estimate bounding box for node based on content and page dimensions"""
        try:
            doc = fitz.open(self.pdf_path)
            page = doc[page_num]

            # Estimate height based on text length and average line height
            text_length = len(node.text)
            avg_chars_per_line = 80
            avg_line_height = 12
            estimated_lines = text_length / avg_chars_per_line
            estimated_height = estimated_lines * avg_line_height

            # Center the text block on page with reasonable margins
            margin = 50
            width = page.rect.width - (2 * margin)
            y_pos = margin

            bbox = BoundingBox(
                x0=margin,
                y0=y_pos,
                x1=margin + width,
                y1=y_pos + estimated_height,
                page_num=page_num
            )
            doc.close()
            return bbox
        except Exception as e:
            logger.warning(f"Failed to get page dimensions: {e}")
            return BoundingBox(x0=0, y0=0, x1=595, y1=842, page_num=page_num)
