from functools import lru_cache
from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.base import BaseEmbeddingProvider, BaseSparseEmbeddingProvider
from app.embeddings.fastembed_provider import FastEmbedDenseProvider, FastEmbedSparseProvider
from app.embeddings.gemini import GeminiEmbeddingProvider
from app.generation.base import BaseLLMProvider
from app.generation.gemini import GeminiLLMProvider
from app.generation.service import GroundedChatService
from app.ingestion.service import DocumentIngestionService
from app.retrieval.hybrid_engine import HybridRetrievalEngine
from app.retrieval.reranker import BaseReranker, CrossEncoderReranker
from app.vector_store.base import BaseVectorStore
from app.vector_store.qdrant_store import QdrantVectorStore


class Container:
    _ingestion_service: Optional[DocumentIngestionService] = None
    _dense_embedder: Optional[BaseEmbeddingProvider] = None
    _sparse_embedder: Optional[BaseSparseEmbeddingProvider] = None
    _reranker: Optional[BaseReranker] = None
    _vector_store: Optional[BaseVectorStore] = None
    _hybrid_engine: Optional[HybridRetrievalEngine] = None
    _llm_provider: Optional[BaseLLMProvider] = None
    _chat_service: Optional[GroundedChatService] = None

    @classmethod
    def get_ingestion_service(cls) -> DocumentIngestionService:
        if cls._ingestion_service is None:
            cls._ingestion_service = DocumentIngestionService()
        return cls._ingestion_service

    @classmethod
    def get_dense_embedder(cls) -> BaseEmbeddingProvider:
        if cls._dense_embedder is None:
            if settings.GEMINI_API_KEY:
                logger.info("Configuring GeminiEmbeddingProvider as primary dense embedder")
                cls._dense_embedder = GeminiEmbeddingProvider()
            else:
                logger.info("Using FastEmbedDenseProvider as fallback dense embedder (BAAI/bge-small-en-v1.5)")
                cls._dense_embedder = FastEmbedDenseProvider()
        return cls._dense_embedder

    @classmethod
    def get_sparse_embedder(cls) -> BaseSparseEmbeddingProvider:
        if cls._sparse_embedder is None:
            cls._sparse_embedder = FastEmbedSparseProvider()
        return cls._sparse_embedder

    @classmethod
    def get_reranker(cls) -> BaseReranker:
        if cls._reranker is None:
            cls._reranker = CrossEncoderReranker()
        return cls._reranker

    @classmethod
    def get_vector_store(cls) -> BaseVectorStore:
        if cls._vector_store is None:
            dense_dim = cls.get_dense_embedder().dimension
            cls._vector_store = QdrantVectorStore(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
                dense_size=dense_dim
            )
        return cls._vector_store

    @classmethod
    def get_hybrid_engine(cls) -> HybridRetrievalEngine:
        if cls._hybrid_engine is None:
            cls._hybrid_engine = HybridRetrievalEngine(
                vector_store=cls.get_vector_store(),
                dense_embedder=cls.get_dense_embedder(),
                sparse_embedder=cls.get_sparse_embedder(),
                reranker=cls.get_reranker()
            )
        return cls._hybrid_engine

    @classmethod
    def get_llm_provider(cls) -> BaseLLMProvider:
        if cls._llm_provider is None:
            cls._llm_provider = GeminiLLMProvider()
        return cls._llm_provider

    @classmethod
    def get_chat_service(cls) -> GroundedChatService:
        if cls._chat_service is None:
            cls._chat_service = GroundedChatService(
                retrieval_engine=cls.get_hybrid_engine(),
                llm_provider=cls.get_llm_provider()
            )
        return cls._chat_service


def get_ingestion_service() -> DocumentIngestionService:
    return Container.get_ingestion_service()


def get_dense_embedder() -> BaseEmbeddingProvider:
    return Container.get_dense_embedder()


def get_sparse_embedder() -> BaseSparseEmbeddingProvider:
    return Container.get_sparse_embedder()


def get_vector_store() -> BaseVectorStore:
    return Container.get_vector_store()


def get_hybrid_engine() -> HybridRetrievalEngine:
    return Container.get_hybrid_engine()


def get_chat_service() -> GroundedChatService:
    return Container.get_chat_service()
