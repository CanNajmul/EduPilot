
"""LLM access: Groq primary, Gemini fallback."""

from __future__ import annotations

import time
from typing import Type, TypeVar

from pydantic import BaseModel
from config import settings

T = TypeVar("T", bound=BaseModel)


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    return str(content)


class LLMService:
    def __init__(self, primary=None, fallback=None):
        self._primary = primary
        self._fallback = fallback
        self._built = primary is not None

    def _build(self) -> None:
        if self._built:
            return

        # Groq Primary
        if settings.GROQ_API_KEY:
            from langchain_groq import ChatGroq

            self._primary = ChatGroq(
                model=settings.FALLBACK_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
            )

        # Gemini Fallback
        if settings.GOOGLE_API_KEY:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._fallback = ChatGoogleGenerativeAI(
                model=settings.PRIMARY_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=settings.LLM_TEMPERATURE,
            )

        if self._primary is None and self._fallback is not None:
            self._primary, self._fallback = self._fallback, None

        if self._primary is None:
            raise RuntimeError(
                "No LLM configured: set GOOGLE_API_KEY and/or GROQ_API_KEY in .env"
            )

        self._built = True

    @staticmethod
    def _messages(system: str, user: str):
        return [("system", system), ("human", user)]

    def text(self, system: str, user: str) -> str:
        self._build()
        start = time.perf_counter()

        try:
            chain = self._primary

            if self._fallback is not None:
                chain = chain.with_fallbacks([self._fallback])

            result = chain.invoke(self._messages(system, user))
            return _content_to_text(result.content).strip()

        finally:
            elapsed = time.perf_counter() - start
            print(
                f"[LLM TIMING] text(): {elapsed:.2f} seconds",
                flush=True,
            )

    def structured(self, schema: Type[T], system: str, user: str) -> T:
        self._build()
        start = time.perf_counter()

        try:
            chain = self._primary.with_structured_output(schema)

            if self._fallback is not None:
                chain = chain.with_fallbacks([
                    self._fallback.with_structured_output(schema)
                ])

            last = None

            for _ in range(settings.MAX_LLM_RETRIES + 1):
                try:
                    result = chain.invoke(self._messages(system, user))

                    if result is None:
                        raise ValueError("Empty structured output")

                    if isinstance(result, schema):
                        return result

                    if isinstance(result, BaseModel):
                        result = result.model_dump()

                    if isinstance(result, dict):
                        return schema.model_validate(result)

                    raise TypeError(
                        f"Unexpected structured output type: {type(result).__name__}"
                    )

                except Exception as exc:
                    last = exc

            raise RuntimeError(
                f"Structured output failed after retries: {last}"
            )

        finally:
            elapsed = time.perf_counter() - start
            print(
                f"[LLM TIMING] structured(): {elapsed:.2f} seconds",
                flush=True,
            )


def get_llm() -> LLMService:
    return LLMService()
