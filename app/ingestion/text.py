import re
from typing import List, Optional
from app.ingestion.base import BaseParser
from app.schemas.document import ParsedSection


class TextParser(BaseParser):
    """Parser for plain text and Markdown files."""

    def parse_bytes(self, content: bytes, filename: str) -> List[ParsedSection]:
        text = content.decode("utf-8", errors="replace")
        return self.parse_text(text, filename)

    def parse_text(self, text: str, filename: Optional[str] = None) -> List[ParsedSection]:
        lines = text.splitlines()
        sections: List[ParsedSection] = []
        current_title = filename or "General"
        current_lines: List[str] = []

        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = heading_pattern.match(line.strip())
            if match:
                # Flush previous section if it has content
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append(
                            ParsedSection(
                                title=current_title,
                                content=content,
                                page_number=1,
                                section_type="text",
                                metadata={"filename": filename} if filename else {}
                            )
                        )
                    current_lines = []
                current_title = match.group(2).strip()
            else:
                current_lines.append(line)

        # Flush final section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(
                    ParsedSection(
                        title=current_title,
                        content=content,
                        page_number=1,
                        section_type="text",
                        metadata={"filename": filename} if filename else {}
                    )
                )

        if not sections and text.strip():
            sections.append(
                ParsedSection(
                    title=filename or "General",
                    content=text.strip(),
                    page_number=1,
                    section_type="text"
                )
            )

        return sections
