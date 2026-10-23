from abc import ABC, abstractmethod
from typing import Generator, List, Optional
from app.schemas.chat import ChatMessage


class BaseLLMProvider(ABC):
    """Abstract interface for LLM generation with streaming capability."""

    @abstractmethod
    def generate(
        self,
        system_instruction: str,
        user_message: str,
        history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.1
    ) -> str:
        """Synchronously generate complete completion string."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        system_instruction: str,
        user_message: str,
        history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.1
    ) -> Generator[str, None, None]:
        """Yield token strings as they arrive from provider."""
        pass
