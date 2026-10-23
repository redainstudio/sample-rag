from typing import List, Optional
from fastembed import SparseTextEmbedding, TextEmbedding
from app.core.logging import logger
from app.embeddings.base import BaseEmbeddingProvider, BaseSparseEmbeddingProvider, SparseVectorData


class FastEmbedDenseProvider(BaseEmbeddingProvider):
    """Local high-throughput dense embedding provider using ONNX/FastEmbed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=self.model_name)
        # Determine dimensionality dynamically from test sample
        sample = list(self._model.embed(["test"]))
        self._dimension = len(sample[0])
        logger.info(f"Initialized FastEmbedDenseProvider with model={self.model_name}, dim={self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = list(self._model.embed(texts, batch_size=self.batch_size))
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embedding = list(self._model.query_embed(text))[0]
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


class FastEmbedSparseProvider(BaseSparseEmbeddingProvider):
    """Local BM25 sparse vector provider using FastEmbed for hybrid retrieval."""

    def __init__(self, model_name: str = "Qdrant/bm25", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SparseTextEmbedding(model_name=self.model_name)
        logger.info(f"Initialized FastEmbedSparseProvider with model={self.model_name}")

    def embed_sparse_documents(self, texts: List[str]) -> List[SparseVectorData]:
        if not texts:
            return []
        embeddings = list(self._model.passage_embed(texts, batch_size=self.batch_size))
        results = []
        for emb in embeddings:
            indices = emb.indices.tolist() if hasattr(emb.indices, "tolist") else list(emb.indices)
            values = emb.values.tolist() if hasattr(emb.values, "tolist") else list(emb.values)
            results.append(SparseVectorData(indices=indices, values=[float(v) for v in values]))
        return results

    def embed_sparse_query(self, text: str) -> SparseVectorData:
        emb = list(self._model.query_embed(text))[0]
        indices = emb.indices.tolist() if hasattr(emb.indices, "tolist") else list(emb.indices)
        values = emb.values.tolist() if hasattr(emb.values, "tolist") else list(emb.values)
        return SparseVectorData(indices=indices, values=[float(v) for v in values])
