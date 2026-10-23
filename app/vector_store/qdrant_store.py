import uuid
from typing import List, Optional
from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.base import SparseVectorData
from app.schemas.document import DocumentChunk
from app.schemas.retrieval import ScoredChunk
from app.vector_store.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant implementation supporting payload-based multi-tenancy (Section 11),
    named dense + sparse vectors (Section 12), and RRF hybrid retrieval (Section 10 & 53).
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
        dense_size: int = settings.DENSE_VECTOR_SIZE,
    ):
        self.url = url or settings.QDRANT_URL
        self.api_key = api_key or (settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None)
        self.collection_name = collection_name
        self.dense_size = dense_size

        if self.url == ":memory:" or not self.url:
            logger.info("Initializing in-memory Qdrant client")
            self.client = QdrantClient(":memory:")
        else:
            logger.info(f"Connecting to Qdrant cluster at {self.url}")
            self.client = QdrantClient(url=self.url, api_key=self.api_key)

        self.ensure_collection(self.dense_size)

    def ensure_collection(self, dimension: int) -> None:
        """Create collection with named dense and sparse vectors, plus tenant payload indexes."""
        self.dense_size = dimension
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            logger.info(f"Creating Qdrant collection: {self.collection_name} (dim={dimension})")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False
                        )
                    )
                }
            )

            # Build tenant payload indexes per Section 11
            for field in ["organization_id", "bot_id", "document_id"]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception as e:
                    logger.debug(f"Payload index creation for {field}: {e}")

    def _to_point_id(self, chunk_id: str) -> str:
        """Generate a deterministic UUID from chunk_id for idempotent storage."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        dense_vectors: List[List[float]],
        sparse_vectors: List[SparseVectorData],
        batch_size: int = 100
    ) -> int:
        """Batch upsert points into Qdrant."""
        if not chunks:
            return 0

        assert len(chunks) == len(dense_vectors) == len(sparse_vectors), (
            "Chunks, dense vectors, and sparse vectors must have identical lengths"
        )

        total_upserted = 0
        points: List[models.PointStruct] = []

        for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
            point_id = self._to_point_id(chunk.chunk_id)
            vector_dict = {
                "dense": dense_vec,
                "sparse": models.SparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values
                )
            }
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "organization_id": chunk.organization_id,
                "bot_id": chunk.bot_id,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata
            }
            points.append(models.PointStruct(id=point_id, vector=vector_dict, payload=payload))

            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.collection_name, points=points)
                total_upserted += len(points)
                points = []

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
            total_upserted += len(points)

        logger.info(f"Upserted {total_upserted} chunks into collection {self.collection_name}")
        return total_upserted

    def delete_document(
        self,
        organization_id: str,
        bot_id: str,
        document_id: str
    ) -> bool:
        """Delete all points belonging to a specific document under the tenant."""
        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(key="organization_id", match=models.MatchValue(value=organization_id)),
                models.FieldCondition(key="bot_id", match=models.MatchValue(value=bot_id)),
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            ]
        )
        self.client.delete(collection_name=self.collection_name, points_selector=tenant_filter)
        logger.info(f"Deleted document points: org={organization_id}, bot={bot_id}, doc={document_id}")
        return True

    def hybrid_search(
        self,
        organization_id: str,
        bot_id: str,
        dense_query: List[float],
        sparse_query: SparseVectorData,
        top_k: int = 25
    ) -> List[ScoredChunk]:
        """
        Hybrid search combining dense semantic search and BM25 sparse lexical search
        fused via RRF (Reciprocal Rank Fusion) with strict tenant isolation.
        """
        tenant_filter = models.Filter(
            must=[
                models.FieldCondition(key="organization_id", match=models.MatchValue(value=organization_id)),
                models.FieldCondition(key="bot_id", match=models.MatchValue(value=bot_id)),
            ]
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_query.indices,
                        values=sparse_query.values
                    ),
                    using="sparse",
                    filter=tenant_filter,
                    limit=top_k
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    filter=tenant_filter,
                    limit=top_k
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=tenant_filter,
            limit=top_k,
            with_payload=True
        )

        scored_chunks: List[ScoredChunk] = []
        for point in response.points:
            p = point.payload or {}
            chunk = DocumentChunk(
                chunk_id=p.get("chunk_id", str(point.id)),
                document_id=p.get("document_id", ""),
                organization_id=p.get("organization_id", organization_id),
                bot_id=p.get("bot_id", bot_id),
                text=p.get("text", ""),
                page_number=p.get("page_number"),
                section_title=p.get("section_title"),
                chunk_index=p.get("chunk_index", 0),
                token_count=p.get("token_count", 0),
                metadata=p.get("metadata", {})
            )
            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(point.score) if point.score is not None else 0.0
                )
            )

        return scored_chunks
