from typing import Generator, List, Optional
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.logging import logger
from app.generation.base import BaseLLMProvider
from app.schemas.chat import ChatMessage


class GeminiLLMProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider supporting fast streaming and evidence grounding (Section 19 & 20).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = settings.GEMINI_LLM_MODEL
    ):
        self._api_key = api_key or settings.GEMINI_API_KEY
        self.model = model

        if not self._api_key:
            logger.warning("GEMINI_API_KEY is not set. GeminiLLMProvider will fail if invoked.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self._api_key)

    def _build_contents(
        self,
        user_message: str,
        history: Optional[List[ChatMessage]] = None
    ) -> List[types.Content]:
        contents: List[types.Content] = []

        if history:
            for msg in history:
                role = "user" if msg.role.value == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg.content)]
                    )
                )

        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)]
            )
        )
        return contents

    def generate(
        self,
        system_instruction: str,
        user_message: str,
        history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.1
    ) -> str:
        if not self.client:
            raise ValueError("GEMINI_API_KEY is not configured.")

        contents = self._build_contents(user_message, history)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
        )
        return response.text or ""

    def generate_stream(
        self,
        system_instruction: str,
        user_message: str,
        history: Optional[List[ChatMessage]] = None,
        temperature: float = 0.1
    ) -> Generator[str, None, None]:
        if not self.client:
            raise ValueError("GEMINI_API_KEY is not configured.")

        contents = self._build_contents(user_message, history)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature
        )

        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config
        )

        for chunk in stream:
            if chunk.text:
                yield chunk.text
