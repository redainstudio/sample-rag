from enum import Enum
from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    EXTRACTING = "EXTRACTING"
    OCR = "OCR"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"


class TenantContext(BaseModel):
    organization_id: str = Field(..., description="Organization identifier")
    bot_id: str = Field(..., description="Bot identifier")


class BaseResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully"
