from abc import ABC, abstractmethod
from typing import List, Optional
from app.embeddings.base import SparseVectorData
from app.schemas.document import DocumentChunk
from app.schemas.retrieval import ScoredChunk


class BaseVectorStore(ABC):
    """Abstract interface for multi-tenant vector storage."""

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        """Create or ensure existence of the vector collection."""
        pass

    @abstractmethod
    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[SparseVectorData],
        batch_size: int = 100
    ) -> int:
        """Batch upsert chunks with dense and sparse vectors."""
        pass

    @abstractmethod
    def delete_document(
        self,
        organization_id: str,
        bot_id: str,
        document_id: str
    ) -> bool:
        """Remove all chunks associated with a document version."""
        pass

    @abstractmethod
    def hybrid_search(
        self,
        organization_id: str,
        bot_id: str,
        dense_query: List[float],
        sparse_query: SparseVectorData,
        top_k: int = 25
    ) -> List[ScoredChunk]:
        """Perform hybrid search with strict multi-tenant filtering."""
        pass
