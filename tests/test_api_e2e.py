import pytest
from fastapi.testclient import TestClient
from app.api.deps import Container
from app.generation.base import BaseLLMProvider
from app.main import app
from app.schemas.chat import ChatMessage


class StubLLMProvider(BaseLLMProvider):
    def generate(self, system_instruction, user_message, history=None, temperature=0.1):
        return "Orders are delivered within 2 to 4 business days throughout Kerala [Source 1]."

    def generate_stream(self, system_instruction, user_message, history=None, temperature=0.1):
        yield "Orders "
        yield "are "
        yield "delivered "
        yield "within "
        yield "2-4 "
        yield "days."


@pytest.fixture(autouse=True)
def override_llm():
    Container._llm_provider = StubLLMProvider()
    Container._chat_service = None  # Reset so it picks up the stub


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["vector_store"] == "ready"


def test_e2e_ingestion_search_chat_and_delete(client):
    org_id = "org_production_test"
    bot_id = "bot_production_test"
    doc_id = "doc_shipping_policy_01"

    faq_content = """# Company Delivery & Shipping Guidelines
Standard shipping to Kerala takes between 2 to 4 business days.
Expedited express delivery takes 24 hours.

## Refund and Replacement
Products can be returned within 30 days of purchase in original packaging.
"""

    # 1. Ingest Document
    ingest_payload = {
        "organization_id": org_id,
        "bot_id": bot_id,
        "document_id": doc_id,
        "filename": "shipping_policy.md",
        "content_text": faq_content,
        "mime_type": "text/markdown"
    }
    ingest_res = client.post("/api/v1/ingestion/process", json=ingest_payload)
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["status"] == "READY"
    assert ingest_data["chunks_created"] >= 2

    # 2. Hybrid Search
    search_payload = {
        "organization_id": org_id,
        "bot_id": bot_id,
        "query": "How many days does standard shipping to Kerala take?",
        "top_k": 5,
        "rerank_top_k": 2
    }
    search_res = client.post("/api/v1/search/hybrid", json=search_payload)
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert len(search_data["results"]) > 0
    top_chunk = search_data["results"][0]
    assert "2 to 4 business days" in top_chunk["chunk"]["text"]
    assert top_chunk["score"] > 0.5

    # 3. Synchronous Chat
    chat_payload = {
        "organization_id": org_id,
        "bot_id": bot_id,
        "message": "What is the delivery time for Kerala?",
        "stream": False
    }
    chat_res = client.post("/api/v1/chat/completions", json=chat_payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert not chat_data["abstained"]
    assert "2 to 4 business days" in chat_data["answer"]
    assert len(chat_data["citations"]) >= 1
    assert chat_data["citations"][0]["document_id"] == doc_id

    # 4. Streaming Chat (SSE)
    chat_stream_payload = {
        "organization_id": org_id,
        "bot_id": bot_id,
        "message": "What is the delivery time for Kerala?",
        "stream": True
    }
    stream_res = client.post("/api/v1/chat/completions", json=chat_stream_payload)
    assert stream_res.status_code == 200
    stream_body = stream_res.text
    assert "event: citation" in stream_body
    assert "event: token" in stream_body
    assert "event: done" in stream_body

    # 5. Delete Document
    del_res = client.delete(
        f"/api/v1/ingestion/documents/{doc_id}?organization_id={org_id}&bot_id={bot_id}"
    )
    assert del_res.status_code == 200

    # 6. Verify Deletion
    verify_search_res = client.post("/api/v1/search/hybrid", json=search_payload)
    assert verify_search_res.status_code == 200
    assert len(verify_search_res.json()["results"]) == 0
