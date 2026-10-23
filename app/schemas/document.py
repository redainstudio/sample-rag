from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.common import DocumentStatus, TenantContext


class ParsedSection(BaseModel):
    title: Optional[str] = None
    content: str
    page_number: Optional[int] = 1
    section_type: str = "text"  # "text", "table", "list", "image"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    organization_id: str
    bot_id: str
    text: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    chunk_index: int = 0
    token_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionRequest(TenantContext):
    document_id: str
    filename: str
    content_text: Optional[str] = None
    file_bytes_base64: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionProgress(BaseModel):
    document_id: str
    status: DocumentStatus
    chunks_created: int = 0
    page_count: int = 0
    error_message: Optional[str] = None
    duration_ms: float = 0.0
