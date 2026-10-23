import time
from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.api.deps import (
    get_dense_embedder,
    get_ingestion_service,
    get_sparse_embedder,
    get_vector_store,
)
from app.core.logging import logger
from app.embeddings.base import BaseEmbeddingProvider, BaseSparseEmbeddingProvider
from app.ingestion.service import DocumentIngestionService
from app.schemas.common import BaseResponse, DocumentStatus
from app.schemas.document import IngestionProgress, IngestionRequest
from app.vector_store.base import BaseVectorStore

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


@router.post(
    "/process",
    response_model=IngestionProgress,
    status_code=status.HTTP_200_OK,
    summary="Parse, chunk, embed, and index a document into Qdrant"
)
async def ingest_document(
    request: IngestionRequest,
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
    dense_embedder: BaseEmbeddingProvider = Depends(get_dense_embedder),
    sparse_embedder: BaseSparseEmbeddingProvider = Depends(get_sparse_embedder),
    vector_store: BaseVectorStore = Depends(get_vector_store)
):
    start_time = time.perf_counter()
    logger.info(
        f"API ingest request: org={request.organization_id}, bot={request.bot_id}, doc={request.document_id}, file={request.filename}"
    )

    # 1. Parse and chunk
    chunks, progress = ingestion_service.process_document(request)
    if progress.status == DocumentStatus.FAILED or not chunks:
        return progress

    try:
        # 2. Compute dense and sparse embeddings in high-throughput batches
        texts = [c.text for c in chunks]
        dense_vectors = dense_embedder.embed_documents(texts)
        sparse_vectors = sparse_embedder.embed_sparse_documents(texts)

        # 3. Upsert into Qdrant with tenant payload isolation
        vector_store.upsert_chunks(
            chunks=chunks,
            dense_vectors=dense_vectors,
            sparse_vectors=sparse_vectors
        )

        total_duration_ms = (time.perf_counter() - start_time) * 1000.0
        progress.duration_ms = total_duration_ms
        progress.status = DocumentStatus.READY
        return progress

    except Exception as e:
        logger.error(f"Failed to embed and index document {request.document_id}: {e}")
        progress.status = DocumentStatus.FAILED
        progress.error_message = str(e)
        progress.duration_ms = (time.perf_counter() - start_time) * 1000.0
        return progress


@router.delete(
    "/documents/{document_id}",
    response_model=BaseResponse,
    summary="Delete a document and all its indexed chunks from Qdrant"
)
async def delete_document(
    document_id: str,
    organization_id: str = Query(..., description="Organization ID"),
    bot_id: str = Query(..., description="Bot ID"),
    vector_store: BaseVectorStore = Depends(get_vector_store)
):
    try:
        vector_store.delete_document(
            organization_id=organization_id,
            bot_id=bot_id,
            document_id=document_id
        )
        return BaseResponse(message=f"Document {document_id} successfully deleted from vector index.")
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {e}"
        )
