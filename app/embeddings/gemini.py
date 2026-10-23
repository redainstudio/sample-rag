from typing import List, Optional
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.base import BaseEmbeddingProvider


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Google Gemini Embedding Provider implementing Section 9 of the Architecture.
    Utilizes gemini-embedding-001 / gemini-embedding-2 with batching and 768 dimensions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = settings.GEMINI_EMBEDDING_MODEL,
        batch_size: int = 64
    ):
        self._api_key = api_key or settings.GEMINI_API_KEY
        self.model = model
        self.batch_size = batch_size
        self._dimension = settings.DENSE_VECTOR_SIZE

        if not self._api_key:
            logger.warning("GEMINI_API_KEY is not set. GeminiEmbeddingProvider will fail if invoked.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self._api_key)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise ValueError("Gemini API key is not configured. Set GEMINI_API_KEY environment variable.")

        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        config = types.EmbedContentConfig(output_dimensionality=self._dimension)

        # Batch requests to maximize throughput per Section 25
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=config
            )
            if response.embeddings:
                for emb in response.embeddings:
                    all_embeddings.append(list(emb.values))
            else:
                raise RuntimeError(f"Empty embeddings response from Gemini for batch {i}")

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        if not self.client:
            raise ValueError("Gemini API key is not configured. Set GEMINI_API_KEY environment variable.")

        config = types.EmbedContentConfig(output_dimensionality=self._dimension)
        response = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config=config
        )
        if response.embeddings and len(response.embeddings) > 0:
            return list(response.embeddings[0].values)
        raise RuntimeError("No embedding returned for query from Gemini.")

