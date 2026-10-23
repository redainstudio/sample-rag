from typing import Generator, List, Optional
import pytest
from app.generation.answerability import AnswerabilityClassifier
from app.generation.base import BaseLLMProvider
from app.generation.prompt_builder import PromptBuilder
from app.generation.service import GroundedChatService
from app.schemas.chat import ChatMessage, ChatRequest, ChatRole, StreamEventType
from app.schemas.document import DocumentChunk
from app.schemas.retrieval import ScoredChunk, SearchRequest, SearchResponse


class MockLLMProvider(BaseLLMProvider):
    """Mock provider for unit testing generation pipeline."""

    def generate(
        self,
        system_instruction: str,
        user_message: str,
        history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.1
    ) -> str:
        assert "<EVIDENCE>" in user_message
        assert "PROMPT INJECTION DEFENSE" in system_instruction
        return "Standard delivery takes 2 to 4 business days throughout Kerala [Source 1]."

    def generate_stream(
        self,
        system_instruction: str,
        user_message: str,
        history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.1
    ) -> Generator[str, None, None]:
        assert "<EVIDENCE>" in user_message
        tokens = ["Standard ", "delivery ", "takes ", "2-4 ", "days."]
        for t in tokens:
            yield t


class MockRetrievalEngine:
    def __init__(self, return_chunks: bool = True):
        self.return_chunks = return_chunks

    def search(self, request: SearchRequest) -> SearchResponse:
        if not self.return_chunks:
            return SearchResponse(query=request.query, results=[], total_candidates=0, duration_ms=1.0)

        chunk = DocumentChunk(
            chunk_id="chk_01",
            document_id="doc_shipping",
            organization_id=request.organization_id,
            bot_id=request.bot_id,
            text="Standard delivery takes 2 to 4 business days throughout Kerala.",
            page_number=4,
            section_title="Delivery Timelines"
        )
        return SearchResponse(
            query=request.query,
            results=[ScoredChunk(chunk=chunk, score=0.92, rerank_score=0.92)],
            total_candidates=1,
            duration_ms=5.0
        )


def test_prompt_builder():
    scored_chunk = ScoredChunk(
        chunk=DocumentChunk(
            chunk_id="chk_01",
            document_id="doc_policy",
            organization_id="org_1",
            bot_id="bot_1",
            text="Refunds are provided within 30 days.",
            page_number=2,
            section_title="Returns"
        ),
        score=0.88
    )

    sys_inst, user_prompt, citations = PromptBuilder.build_prompt(
        user_message="Can I get a refund?",
        scored_chunks=[scored_chunk],
        custom_instructions="Be extra polite."
    )

    assert "PROMPT INJECTION DEFENSE" in sys_inst
    assert "Be extra polite" in sys_inst
    assert "<EVIDENCE>" in user_prompt
    assert "[Source 1]" in user_prompt
    assert "Refunds are provided within 30 days." in user_prompt
    assert len(citations) == 1
    assert citations[0].document_id == "doc_policy"
    assert citations[0].page_number == 2


def test_answerability_classifier():
    classifier = AnswerabilityClassifier(min_threshold=0.35)

    # Empty chunks -> False
    ok, _ = classifier.is_answerable([])
    assert not ok

    # Low score chunk -> False
    low_chunk = ScoredChunk(
        chunk=DocumentChunk(
            chunk_id="c1", document_id="d1", organization_id="o1", bot_id="b1", text="irrelevant"
        ),
        score=0.15
    )
    ok_low, _ = classifier.is_answerable([low_chunk])
    assert not ok_low

    # High score chunk -> True
    high_chunk = ScoredChunk(
        chunk=DocumentChunk(
            chunk_id="c2", document_id="d1", organization_id="o1", bot_id="b1", text="relevant text"
        ),
        score=0.75
    )
    ok_high, _ = classifier.is_answerable([high_chunk])
    assert ok_high


def test_grounded_chat_service_success():
    retrieval = MockRetrievalEngine(return_chunks=True)
    llm = MockLLMProvider()
    service = GroundedChatService(retrieval_engine=retrieval, llm_provider=llm)

    req = ChatRequest(
        organization_id="org_1",
        bot_id="bot_1",
        message="How long does delivery take?",
        stream=False
    )
    res = service.process_chat(req)

    assert not res.abstained
    assert "Standard delivery takes 2 to 4 business days" in res.answer
    assert len(res.citations) == 1
    assert res.citations[0].document_id == "doc_shipping"


def test_grounded_chat_service_abstention():
    retrieval = MockRetrievalEngine(return_chunks=False)
    llm = MockLLMProvider()
    service = GroundedChatService(retrieval_engine=retrieval, llm_provider=llm)

    req = ChatRequest(
        organization_id="org_1",
        bot_id="bot_1",
        message="What is the weather on Mars?",
        stream=False
    )
    res = service.process_chat(req)

    assert res.abstained
    assert "do not have enough information" in res.answer
    assert len(res.citations) == 0


def test_grounded_chat_service_streaming():
    retrieval = MockRetrievalEngine(return_chunks=True)
    llm = MockLLMProvider()
    service = GroundedChatService(retrieval_engine=retrieval, llm_provider=llm)

    req = ChatRequest(
        organization_id="org_1",
        bot_id="bot_1",
        message="Delivery timeline",
        stream=True
    )
    events = list(service.process_chat_stream(req))

    event_types = [e.event for e in events]
    assert StreamEventType.CITATION in event_types
    assert StreamEventType.TOKEN in event_types
    assert StreamEventType.DONE in event_types

    # Ensure citations came before tokens
    citation_idx = event_types.index(StreamEventType.CITATION)
    token_idx = event_types.index(StreamEventType.TOKEN)
    assert citation_idx < token_idx
