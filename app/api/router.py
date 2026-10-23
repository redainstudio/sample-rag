from fastapi import APIRouter
from app.api.v1.chat import router as chat_router
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.search import router as search_router

api_router = APIRouter()
api_router.include_router(ingestion_router)
api_router.include_router(search_router)
api_router.include_router(chat_router)
