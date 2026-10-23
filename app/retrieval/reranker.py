import math
from abc import ABC, abstractmethod
from typing import List, Optional
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app.core.logging import logger
from app.schemas.retrieval import ScoredChunk


class BaseReranker(ABC):
    """Abstract interface for passage reranking."""

    @abstractmethod
    def rerank(self, query: str, candidates: List[ScoredChunk], top_k: int = 5) -> List[ScoredChunk]:
        """Rerank candidates and return the top_k most relevant chunks."""
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder Reranker implementing Section 13 of the Architecture Blueprint.
    Re-scores dense+sparse candidate passages with fine-grained query-document attention.
    """

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=self.model_name)
        logger.info(f"Initialized CrossEncoderReranker with {self.model_name}")

    def _sigmoid(self, x: float) -> float:
        """Convert cross-encoder logit into normalized [0, 1] probability."""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    def rerank(self, query: str, candidates: List[ScoredChunk], top_k: int = 5) -> List[ScoredChunk]:
        if not candidates:
            return []

        doc_texts = [c.chunk.text for c in candidates]
        try:
            raw_scores = list(self._model.rerank(query, doc_texts))
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to initial retrieval scores.")
            return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]

        reranked: List[ScoredChunk] = []
        for candidate, raw_score in zip(candidates, raw_scores):
            norm_score = self._sigmoid(float(raw_score))
            # Store both original and rerank scores
            reranked.append(
                ScoredChunk(
                    chunk=candidate.chunk,
                    score=norm_score,
                    dense_score=candidate.dense_score,
                    sparse_score=candidate.sparse_score,
                    rerank_score=norm_score
                )
            )

        # Sort descending by normalized reranker relevance score
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]
