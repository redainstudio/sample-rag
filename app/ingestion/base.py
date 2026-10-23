from abc import ABC, abstractmethod
from typing import List, Optional
from app.schemas.document import ParsedSection


class BaseParser(ABC):
    """Abstract interface for file parsers."""

    @abstractmethod
    def parse_bytes(self, content: bytes, filename: str) -> List[ParsedSection]:
        """Extract structured sections from raw file bytes."""
        pass

    @abstractmethod
    def parse_text(self, text: str, filename: Optional[str] = None) -> List[ParsedSection]:
        """Extract structured sections from plain text string."""
        pass
