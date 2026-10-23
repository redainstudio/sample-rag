from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from app.api.deps import Container
from app.api.router import api_router
from app.core.config import settings
from app.core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} in [{settings.APP_ENV}] mode...")
    # Pre-warm embedders and vector store connections
    try:
        Container.get_dense_embedder()
        Container.get_sparse_embedder()
        Container.get_reranker()
        Container.get_vector_store()
        logger.info("All AI pipelines, embedders, and vector stores pre-warmed successfully.")
    except Exception as e:
        logger.error(f"Error during startup pre-warming: {e}")

    yield

    logger.info(f"Shutting down {settings.APP_NAME}...")


def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="High-Speed Ingestion, Hybrid Retrieval & Grounded AI Service for Botzi SaaS",
        version="0.1.0",
        lifespan=lifespan
    )

    # Enable CORS for Next.js SaaS dashboard and external widget domains
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
    async def health_check():
        return {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "environment": settings.APP_ENV,
            "gemini_configured": bool(settings.GEMINI_API_KEY),
            "vector_store": "ready"
        }

    # Register API v1 routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
