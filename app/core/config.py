from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    APP_NAME: str = "Botzi AI Service"
    APP_ENV: Literal["development", "staging", "production", "test"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # API
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Google Gemini AI
    GEMINI_API_KEY: str = ""
    GEMINI_LLM_MODEL: str = "gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"

    # Vector Database - Qdrant
    # If QDRANT_URL is empty or ":memory:", runs in-memory (great for tests/local)
    QDRANT_URL: str = ":memory:"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "botzi_knowledge"
    DENSE_VECTOR_SIZE: int = 768  # 768 for text-embedding-004 or BAAI/bge-base-en-v1.5

    # Reranking & Retrieval Defaults
    RETRIEVAL_TOP_K: int = 25  # Candidates from dense + sparse
    RERANK_TOP_K: int = 5      # Final chunks passed to LLM
    MIN_RELEVANCE_SCORE: float = 0.35  # Threshold below which bot abstains

    # Chunking Defaults
    CHUNK_SIZE_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 50


settings = Settings()
