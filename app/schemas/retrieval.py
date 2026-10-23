from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.schemas.common import TenantContext
from app.schemas.document import DocumentChunk


class SearchRequest(TenantContext):
    query: str = Field(..., min_length=1, description="Visitor or user query text")
    top_k: int = Field(default=25, ge=1, le=100, description="Candidate retrieval count")
    rerank_top_k: int = Field(default=5, ge=1, le=20, description="Count after reranker")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance filter")


class ScoredChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: List[ScoredChunk]
    total_candidates: int
    duration_ms: float
