import time
from typing import List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.base import BaseEmbeddingProvider, BaseSparseEmbeddingProvider
from app.retrieval.reranker import BaseReranker, CrossEncoderReranker
from app.schemas.retrieval import ScoredChunk, SearchRequest, SearchResponse
from app.vector_store.base import BaseVectorStore


class HybridRetrievalEngine:
    """
    Production Hybrid Retrieval Engine implementing Sections 10, 13, 14, and 51 of the Architecture:
    Dense Vector Search + BM25 Sparse Search -> RRF Candidate Fusion -> Cross-Encoder Reranking.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        dense_embedder: BaseEmbeddingProvider,
        sparse_embedder: BaseSparseEmbeddingProvider,
        reranker: Optional[BaseReranker] = None
    ):
        self.vector_store = vector_store
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.reranker = reranker or CrossEncoderReranker()

    def search(self, request: SearchRequest) -> SearchResponse:
        start_time = time.perf_counter()
        logger.info(
            f"Hybrid search query='{request.query}', org={request.organization_id}, bot={request.bot_id}"
        )

        # 1. Generate dense & sparse query representations
        dense_query = self.dense_embedder.embed_query(request.query)
        sparse_query = self.sparse_embedder.embed_sparse_query(request.query)

        # 2. Execute tenant-isolated hybrid retrieval via Qdrant RRF
        candidates = self.vector_store.hybrid_search(
            organization_id=request.organization_id,
            bot_id=request.bot_id,
            dense_query=dense_query,
            sparse_query=sparse_query,
            top_k=request.top_k
        )

        total_candidates = len(candidates)
        if not candidates:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResponse(
                query=request.query,
                results=[],
                total_candidates=0,
                duration_ms=duration_ms
            )

        # 3. Fine-grained cross-encoder reranking
        reranked_chunks = self.reranker.rerank(
            query=request.query,
            candidates=candidates,
            top_k=request.rerank_top_k
        )

        # 4. Filter by minimum relevance score threshold if specified
        if request.min_score > 0.0:
            reranked_chunks = [c for c in reranked_chunks if c.score >= request.min_score]

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"Hybrid search completed in {duration_ms:.2f}ms: "
            f"{total_candidates} candidates -> {len(reranked_chunks)} final reranked chunks"
        )

        return SearchResponse(
            query=request.query,
            results=reranked_chunks,
            total_candidates=total_candidates,
            duration_ms=duration_ms
        )
