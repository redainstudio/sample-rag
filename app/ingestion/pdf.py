import io
from typing import List, Optional
from pypdf import PdfReader
from app.core.logging import logger
from app.ingestion.base import BaseParser
from app.schemas.document import ParsedSection


class PDFParser(BaseParser):
    """Structure-aware PDF parser extracting text by page and detecting headers."""

    def parse_bytes(self, content: bytes, filename: str) -> List[ParsedSection]:
        sections: List[ParsedSection] = []
        try:
            reader = PdfReader(io.BytesIO(content))
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages):
                page_num = page_idx + 1
                page_text = page.extract_text() or ""
                page_text = page_text.strip()

                if not page_text:
                    continue

                # Basic heuristic to extract the first non-empty line as section title
                lines = [l.strip() for l in page_text.splitlines() if l.strip()]
                title = lines[0] if lines and len(lines[0]) < 80 else f"Page {page_num}"

                sections.append(
                    ParsedSection(
                        title=title,
                        content=page_text,
                        page_number=page_num,
                        section_type="text",
                        metadata={
                            "filename": filename,
                            "page_number": page_num,
                            "total_pages": total_pages
                        }
                    )
                )

        except Exception as e:
            logger.error(f"Failed to parse PDF {filename}: {e}")
            raise e

        return sections

    def parse_text(self, text: str, filename: Optional[str] = None) -> List[ParsedSection]:
        return [
            ParsedSection(
                title=filename or "PDF Content",
                content=text,
                page_number=1,
                section_type="text"
            )
        ]
