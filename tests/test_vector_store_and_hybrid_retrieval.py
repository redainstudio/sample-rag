import pytest
from app.embeddings.fastembed_provider import FastEmbedDenseProvider, FastEmbedSparseProvider
from app.retrieval.hybrid_engine import HybridRetrievalEngine
from app.retrieval.reranker import CrossEncoderReranker
from app.schemas.document import DocumentChunk
from app.schemas.retrieval import SearchRequest
from app.vector_store.qdrant_store import QdrantVectorStore


@pytest.fixture(scope="module")
def dense_embedder():
    return FastEmbedDenseProvider(model_name="BAAI/bge-small-en-v1.5")


@pytest.fixture(scope="module")
def sparse_embedder():
    return FastEmbedSparseProvider(model_name="Qdrant/bm25")


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker(model_name="Xenova/ms-marco-MiniLM-L-6-v2")


@pytest.fixture(scope="module")
def vector_store(dense_embedder):
    store = QdrantVectorStore(url=":memory:", dense_size=dense_embedder.dimension)
    return store


def test_hybrid_search_and_tenant_isolation(vector_store, dense_embedder, sparse_embedder, reranker):
    # Setup test chunks for Bot A
    chunks_bot_a = [
        DocumentChunk(
            chunk_id="chk_a_001",
            document_id="doc_shipping",
            organization_id="org_1",
            bot_id="bot_a",
            text="Standard delivery takes 2 to 4 business days throughout Kerala.",
            page_number=1,
            section_title="Shipping Policy"
        ),
        DocumentChunk(
            chunk_id="chk_a_002",
            document_id="doc_catalog",
            organization_id="org_1",
            bot_id="bot_a",
            text="The flagship product SKU-BOTZI-99 costs 49 USD.",
            page_number=2,
            section_title="Catalog"
        )
    ]

    # Setup test chunks for Bot B (same org, different bot)
    chunks_bot_b = [
        DocumentChunk(
            chunk_id="chk_b_001",
            document_id="doc_secret",
            organization_id="org_1",
            bot_id="bot_b",
            text="Internal confidential memo regarding bot B quarterly goals.",
            page_number=1,
            section_title="Internal"
        )
    ]

    # Upsert Bot A
    dense_vecs_a = dense_embedder.embed_documents([c.text for c in chunks_bot_a])
    sparse_vecs_a = sparse_embedder.embed_sparse_documents([c.text for c in chunks_bot_a])
    vector_store.upsert_chunks(chunks_bot_a, dense_vecs_a, sparse_vecs_a)

    # Upsert Bot B
    dense_vecs_b = dense_embedder.embed_documents([c.text for c in chunks_bot_b])
    sparse_vecs_b = sparse_embedder.embed_sparse_documents([c.text for c in chunks_bot_b])
    vector_store.upsert_chunks(chunks_bot_b, dense_vecs_b, sparse_vecs_b)

    engine = HybridRetrievalEngine(
        vector_store=vector_store,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        reranker=reranker
    )

    # 1. Test Semantic Search: "How long does it take for something to reach Kerala?"
    req_semantic = SearchRequest(
        organization_id="org_1",
        bot_id="bot_a",
        query="How long does it take for something to reach Kerala?",
        top_k=10,
        rerank_top_k=2
    )
    res_semantic = engine.search(req_semantic)
    assert len(res_semantic.results) > 0
    top_result = res_semantic.results[0]
    assert top_result.chunk.chunk_id == "chk_a_001"
    assert "2 to 4 business days" in top_result.chunk.text
    assert top_result.score > 0.5

    # 2. Test Exact Keyword/SKU Search: "SKU-BOTZI-99"
    req_sku = SearchRequest(
        organization_id="org_1",
        bot_id="bot_a",
        query="What is the price of SKU-BOTZI-99?",
        top_k=10,
        rerank_top_k=2
    )
    res_sku = engine.search(req_sku)
    assert len(res_sku.results) > 0
    assert res_sku.results[0].chunk.chunk_id == "chk_a_002"
    assert "49 USD" in res_sku.results[0].chunk.text

    # 3. Test Strict Multi-Tenant Isolation
    # Query bot_a for text only present in bot_b
    req_isolation = SearchRequest(
        organization_id="org_1",
        bot_id="bot_a",
        query="confidential memo quarterly goals",
        top_k=10,
        rerank_top_k=2
    )
    res_isolation = engine.search(req_isolation)
    # Must NOT contain bot_b chunks
    for r in res_isolation.results:
        assert r.chunk.bot_id == "bot_a"
        assert r.chunk.chunk_id != "chk_b_001"
