import time
from typing import Generator, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.generation.answerability import AnswerabilityClassifier
from app.generation.base import BaseLLMProvider
from app.generation.gemini import GeminiLLMProvider
from app.generation.prompt_builder import PromptBuilder
from app.retrieval.hybrid_engine import HybridRetrievalEngine
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamPayload,
    Citation,
    StreamEventType,
)
from app.schemas.retrieval import SearchRequest


class GroundedChatService:
    """
    End-to-end grounded chat service enforcing Sections 16, 20, 37, 38, and 51 of the Architecture.
    Retrieves evidence -> checks answerability -> constructs grounded context -> streams tokens + citations.
    """

    def __init__(
        self,
        retrieval_engine: HybridRetrievalEngine,
        llm_provider: Optional[BaseLLMProvider] = None,
        answerability: Optional[AnswerabilityClassifier] = None,
    ):
        self.retrieval_engine = retrieval_engine
        self.llm_provider = llm_provider or GeminiLLMProvider()
        self.answerability = answerability or AnswerabilityClassifier()

    def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Synchronous chat generation."""
        start_time = time.perf_counter()

        # 1. Retrieve candidates via hybrid engine
        search_req = SearchRequest(
            organization_id=request.organization_id,
            bot_id=request.bot_id,
            query=request.message,
            top_k=settings.RETRIEVAL_TOP_K,
            rerank_top_k=settings.RERANK_TOP_K
        )
        search_res = self.retrieval_engine.search(search_req)

        # 2. Check answerability
        is_answerable, reason = self.answerability.is_answerable(search_res.results)
        if not is_answerable:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ChatResponse(
                answer=self.answerability.get_abstention_response(),
                citations=[],
                abstained=True,
                duration_ms=duration_ms
            )

        # 3. Build grounded prompt and citation index
        system_instruction, user_prompt, citations = PromptBuilder.build_prompt(
            user_message=request.message,
            scored_chunks=search_res.results,
            custom_instructions=request.custom_system_prompt or ""
        )

        # 4. Generate answer via LLM
        answer = self.llm_provider.generate(
            system_instruction=system_instruction,
            user_message=user_prompt,
            history=request.history,
            temperature=0.1
        )

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return ChatResponse(
            answer=answer,
            citations=citations,
            abstained=False,
            duration_ms=duration_ms
        )

    def process_chat_stream(
        self, request: ChatRequest
    ) -> Generator[ChatStreamPayload, None, None]:
        """Streaming chat generation emitting tokens, citations, and completion events."""
        start_time = time.perf_counter()

        # 1. Retrieve candidates
        search_req = SearchRequest(
            organization_id=request.organization_id,
            bot_id=request.bot_id,
            query=request.message,
            top_k=settings.RETRIEVAL_TOP_K,
            rerank_top_k=settings.RERANK_TOP_K
        )
        search_res = self.retrieval_engine.search(search_req)

        # 2. Evaluate answerability
        is_answerable, _ = self.answerability.is_answerable(search_res.results)
        if not is_answerable:
            abstention_msg = self.answerability.get_abstention_response()
            # Yield token for abstention
            yield ChatStreamPayload(
                event=StreamEventType.TOKEN,
                data=abstention_msg
            )
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            yield ChatStreamPayload(
                event=StreamEventType.DONE,
                data={"abstained": True, "duration_ms": duration_ms}
            )
            return

        # 3. Context & citations
        system_instruction, user_prompt, citations = PromptBuilder.build_prompt(
            user_message=request.message,
            scored_chunks=search_res.results,
            custom_instructions=request.custom_system_prompt or ""
        )

        # Yield citations first or progressively so UI widget can prepare sources
        for citation in citations:
            yield ChatStreamPayload(
                event=StreamEventType.CITATION,
                data=citation.model_dump()
            )

        # 4. Stream response tokens
        try:
            token_stream = self.llm_provider.generate_stream(
                system_instruction=system_instruction,
                user_message=user_prompt,
                history=request.history,
                temperature=0.1
            )
            for token in token_stream:
                yield ChatStreamPayload(
                    event=StreamEventType.TOKEN,
                    data=token
                )
        except Exception as e:
            logger.error(f"Streaming LLM failure: {e}")
            yield ChatStreamPayload(
                event=StreamEventType.ERROR,
                data=str(e)
            )
            return

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        yield ChatStreamPayload(
            event=StreamEventType.DONE,
            data={"abstained": False, "duration_ms": duration_ms}
        )
