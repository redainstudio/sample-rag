from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.common import TenantContext


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class Citation(BaseModel):
    document_id: str
    filename: Optional[str] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    snippet: str
    relevance_score: float = 0.0


class ChatRequest(TenantContext):
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1, description="Visitor query")
    history: List[ChatMessage] = Field(default_factory=list, description="Prior conversation messages")
    custom_system_prompt: Optional[str] = None
    stream: bool = Field(default=True, description="Stream response tokens via SSE")


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    abstained: bool = False
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class StreamEventType(str, Enum):
    TOKEN = "token"
    CITATION = "citation"
    DONE = "done"
    ERROR = "error"


class ChatStreamPayload(BaseModel):
    event: StreamEventType
    data: Any
