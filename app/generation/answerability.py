from typing import List, Tuple
from app.core.config import settings
from app.core.logging import logger
from app.schemas.retrieval import ScoredChunk


class AnswerabilityClassifier:
    """
    Evaluates evidence sufficiency before generation per Section 16 of the Architecture Blueprint.
    Prevents hallucination by triggering early safe abstention if retrieval confidence is low.
    """

    def __init__(self, min_threshold: float = settings.MIN_RELEVANCE_SCORE):
        self.min_threshold = min_threshold

    def is_answerable(self, chunks: List[ScoredChunk]) -> Tuple[bool, str]:
        """
        Determines whether the retrieved evidence is sufficient to formulate an accurate answer.
        Returns (is_answerable, reason).
        """
        if not chunks:
            logger.info("Answerability check: REJECTED (Zero candidates retrieved)")
            return False, "No knowledge chunks were retrieved."

        top_score = max(c.score for c in chunks)
        if top_score < self.min_threshold:
            logger.info(
                f"Answerability check: REJECTED (Top score {top_score:.3f} < threshold {self.min_threshold})"
            )
            return False, f"Evidence confidence score ({top_score:.3f}) is below reliability threshold ({self.min_threshold})."

        logger.info(f"Answerability check: PASSED (Top score: {top_score:.3f})")
        return True, "Evidence is sufficient."

    @staticmethod
    def get_abstention_response() -> str:
        """Standard safe abstention text when knowledge is absent or insufficient."""
        return (
            "I apologize, but I do not have enough information in the provided knowledge base "
            "to answer this question accurately. Please contact support or provide additional documentation."
        )
