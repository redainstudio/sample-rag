from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_hybrid_engine
from app.retrieval.hybrid_engine import HybridRetrievalEngine
from app.schemas.retrieval import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["Retrieval"])


@router.post(
    "/hybrid",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute hybrid search (Dense + BM25 + Reranking) with multi-tenant filtering"
)
async def hybrid_search(
    request: SearchRequest,
    engine: HybridRetrievalEngine = Depends(get_hybrid_engine)
):
    try:
        response = engine.search(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failure: {e}"
        )
