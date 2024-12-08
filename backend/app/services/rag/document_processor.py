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
            chunk_sizes=[1536, 768, 384],
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
                            text_with_layout += f"[Seite {page_num + 1}]\n{text}\n"

            # Create LlamaIndex document
            doc = Document(text=text_with_layout)

            # Parse into nodes
            nodes = self.node_parser.get_nodes_from_documents([doc])
            leaf_nodes = get_leaf_nodes(nodes)

            # Convert to our DocumentSection format
            sections = []
            for node in leaf_nodes:
                # Get spatial info from metadata
                page_match = re.search(r'\[Seite (\d+)\]', node.text)
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
        """Extract all visual elements from the document"""
        images = []

        # Extract bitmap images
        bitmap_images = self._extract_bitmap_images(pdf_document, file_id)
        images.extend(bitmap_images)

        # Extract vector drawings and diagrams
        #vector_images = self._extract_vector_drawings(pdf_document, file_id)
        #images.extend(vector_images)

        logger.info(f"Total visual elements extracted: {len(images)}")
        return images

    def _extract_bitmap_images(self, pdf_document: fitz.Document, file_id: str) -> List[DocumentImage]:
        """Extract regular bitmap images from the document"""
        images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images()
            logger.info(f"Found {len(image_list)} bitmap images on page {page_num}")

            for img_index, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    pix = fitz.Pixmap(pdf_document, xref)

                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    image_filename = f"{file_id}_{page_num}_{img_index}.png"
                    image_path = f"storage/images/{image_filename}"
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    pix.save(image_path)

                    rects = page.get_image_rects(xref)
                    image_rect = rects[0] if rects else page.rect
                    logger.info(f"Image {img_index} rect: {image_rect}")

                    image_bbox = BoundingBox(
                        x0=image_rect.x0,
                        y0=image_rect.y0,
                        x1=image_rect.x1,
                        y1=image_rect.y1,
                        page_num=page_num
                    )

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

        return images

    def _extract_vector_drawings(self, pdf_document: fitz.Document, file_id: str) -> List[DocumentImage]:
        """Extract vector drawings and diagrams from the document"""
        images = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            drawings = page.get_drawings()
            logger.info(f"Found {len(drawings)} potential drawings on page {page_num}")

            # Group drawings that are likely part of the same diagram
            grouped_drawings = self._group_related_drawings(page, drawings)

            for group_index, drawing_group in enumerate(grouped_drawings):
                try:
                    # Calculate combined bounding box for the group
                    combined_rect = self._get_combined_rect(drawing_group)
                    if not combined_rect:
                        continue

                    # Add padding to ensure we capture everything
                    padding = 10
                    clip_rect = fitz.Rect(
                        combined_rect.x0 - padding,
                        combined_rect.y0 - padding,
                        combined_rect.x1 + padding,
                        combined_rect.y1 + padding
                    )

                    # Get high-resolution pixmap of the area
                    pix = page.get_pixmap(
                        matrix=fitz.Matrix(3, 3),  # 3x scale for better quality
                        clip=clip_rect
                    )

                    # Check if the content is meaningful
                    if not self._is_meaningful_content(pix):
                        continue

                    image_filename = f"{file_id}_{page_num}_diagram_{group_index}.png"
                    image_path = f"storage/images/{image_filename}"
                    pix.save(image_path)

                    image_bbox = BoundingBox(
                        x0=combined_rect.x0,
                        y0=combined_rect.y0,
                        x1=combined_rect.x1,
                        y1=combined_rect.y1,
                        page_num=page_num
                    )

                    caption = self._find_diagram_caption(page, combined_rect)

                    image = DocumentImage(
                        image_data=pix.samples,
                        page_num=page_num,
                        image_type='diagram',
                        bbox=image_bbox,
                        caption=caption,
                        metadata={
                            'image_index': f"diagram_{group_index}",
                            'file_path': image_path,
                            'width': pix.width,
                            'height': pix.height,
                            'has_alpha': False,
                            'drawing_elements': len(drawing_group)
                        },
                        referenced_by=[]
                    )

                    images.append(image)
                    pix = None

                except Exception as e:
                    logger.error(f"Failed to process drawing group {group_index} on page {page_num}: {str(e)}")
                    continue

        return images

    def _group_related_drawings(self, page: fitz.Page, drawings: list) -> List[List[dict]]:
        """Group drawings that are likely part of the same diagram"""
        if not drawings:
            return []

        groups = []
        current_group = [drawings[0]]

        for i in range(1, len(drawings)):
            current_rect = drawings[i].get("rect")
            if not current_rect:
                continue

            # Check if this drawing is close to any in the current group
            should_group = False
            for prev_drawing in current_group:
                prev_rect = prev_drawing.get("rect")
                if not prev_rect:
                    continue

                # Calculate distance between drawings
                distance = self._calculate_distance(current_rect, prev_rect)
                if distance < 20:  # Threshold for grouping
                    should_group = True
                    break

            if should_group:
                current_group.append(drawings[i])
            else:
                if len(current_group) > 0:
                    groups.append(current_group)
                current_group = [drawings[i]]

        if len(current_group) > 0:
            groups.append(current_group)

        return groups

    def _calculate_distance(self, rect1: fitz.Rect, rect2: fitz.Rect) -> float:
        """Calculate the minimum distance between two rectangles"""
        # Rectangles overlap
        if rect1.intersects(rect2):
            return 0

        # Calculate closest points
        dx = max(rect1.x0 - rect2.x1, rect2.x0 - rect1.x1, 0)
        dy = max(rect1.y0 - rect2.y1, rect2.y0 - rect1.y1, 0)

        return (dx ** 2 + dy ** 2) ** 0.5

    def _get_combined_rect(self, drawing_group: List[dict]) -> fitz.Rect | None:
        """Get the combined bounding rectangle for a group of drawings"""
        if not drawing_group:
            return None

        # Initialize with the first drawing's rectangle
        first_rect = drawing_group[0].get("rect")
        if not first_rect:
            return None

        combined = fitz.Rect(first_rect)

        # Include all other drawings
        for drawing in drawing_group[1:]:
            rect = drawing.get("rect")
            if rect:
                combined.include_rect(rect)

        return combined

    def _is_meaningful_content(self, pix: fitz.Pixmap) -> bool:
        """Check if the pixmap contains meaningful content"""
        # Convert to bytes for analysis
        samples = bytes(pix.samples)

        # Check if there's enough variation in pixel values
        unique_values = len(set(samples))
        if unique_values <= 2:  # Only background and one color
            return False

        # Check if the image has enough non-white pixels
        non_white_ratio = sum(1 for b in samples if b < 240) / len(samples)
        return non_white_ratio > 0.05  # At least 5% non-white pixels

    def _find_diagram_caption(self, page: fitz.Page, rect: fitz.Rect) -> str:
        """Find caption for a diagram, checking both above and below"""
        caption_height = 50

        # Look below first
        below_area = fitz.Rect(
            rect.x0 - 10,
            rect.y1,
            rect.x1 + 10,
            rect.y1 + caption_height
        )
        caption = page.get_text(clip=below_area).strip()

        # If no caption found below, look above
        if not caption or len(caption) < 10:  # Minimum meaningful caption length
            above_area = fitz.Rect(
                rect.x0 - 10,
                rect.y0 - caption_height,
                rect.x1 + 10,
                rect.y0
            )
            caption = page.get_text(clip=above_area).strip()

        return caption

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
