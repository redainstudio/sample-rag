import base64
import time
from typing import List, Optional
from app.core.logging import logger
from app.ingestion.base import BaseParser
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.docx import DOCXParser
from app.ingestion.pdf import PDFParser
from app.ingestion.text import TextParser
from app.ingestion.xlsx import XLSXParser
from app.schemas.common import DocumentStatus
from app.schemas.document import DocumentChunk, IngestionProgress, IngestionRequest, ParsedSection


class DocumentIngestionService:
    """Orchestrates document extraction and semantic chunking."""

    def __init__(self):
        self.text_parser = TextParser()
        self.pdf_parser = PDFParser()
        self.docx_parser = DOCXParser()
        self.xlsx_parser = XLSXParser()
        self.chunker = StructureAwareChunker()

    def _select_parser(self, filename: str, mime_type: Optional[str] = None) -> BaseParser:
        lower_name = filename.lower()
        if lower_name.endswith(".pdf") or (mime_type and "pdf" in mime_type):
            return self.pdf_parser
        elif lower_name.endswith((".docx", ".doc")) or (mime_type and "word" in mime_type):
            return self.docx_parser
        elif lower_name.endswith((".xlsx", ".xls", ".csv")) or (mime_type and ("spreadsheet" in mime_type or "excel" in mime_type)):
            return self.xlsx_parser
        else:
            return self.text_parser

    def process_document(self, request: IngestionRequest) -> tuple[List[DocumentChunk], IngestionProgress]:
        start_time = time.perf_counter()
        logger.info(f"Starting ingestion for doc={request.document_id}, bot={request.bot_id}, file={request.filename}")

        parser = self._select_parser(request.filename, request.mime_type)
        sections: List[ParsedSection] = []

        try:
            if request.file_bytes_base64:
                file_bytes = base64.b64decode(request.file_bytes_base64)
                sections = parser.parse_bytes(file_bytes, request.filename)
            elif request.content_text:
                sections = parser.parse_text(request.content_text, request.filename)
            else:
                raise ValueError("Either content_text or file_bytes_base64 must be provided")

            source_ext = request.filename.split(".")[-1].lower() if "." in request.filename else "text"
            chunks = self.chunker.chunk_sections(
                sections=sections,
                organization_id=request.organization_id,
                bot_id=request.bot_id,
                document_id=request.document_id,
                source_type=source_ext
            )

            # Determine distinct page count if applicable
            pages = {s.page_number for s in sections if s.page_number is not None}
            page_count = len(pages) if pages else 1

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(f"Document {request.document_id} parsed into {len(chunks)} chunks in {duration_ms:.2f}ms")

            progress = IngestionProgress(
                document_id=request.document_id,
                status=DocumentStatus.READY,
                chunks_created=len(chunks),
                page_count=page_count,
                duration_ms=duration_ms
            )

            return chunks, progress

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"Ingestion failed for {request.document_id}: {e}")
            progress = IngestionProgress(
                document_id=request.document_id,
                status=DocumentStatus.FAILED,
                chunks_created=0,
                page_count=0,
                error_message=str(e),
                duration_ms=duration_ms
            )
            return [], progress
