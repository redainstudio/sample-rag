from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pydantic import BaseModel


class SparseVectorData(BaseModel):
    indices: List[int]
    values: List[float]


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality."""
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute dense vectors for a batch of document texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Compute dense vector for a single query."""
        pass


class BaseSparseEmbeddingProvider(ABC):
    """Abstract interface for sparse/BM25 embedding providers."""

    @abstractmethod
    def embed_sparse_documents(self, texts: List[str]) -> List[SparseVectorData]:
        """Compute sparse vectors (BM25/SPLADE) for a batch of document texts."""
        pass

    @abstractmethod
    def embed_sparse_query(self, text: str) -> SparseVectorData:
        """Compute sparse vector for a single query."""
        pass
