"""Shared helpers: LLM factory, citation formatting, confidence scoring."""
from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import Settings
from .telemetry import TELEMETRY_CALLBACK

MEDICAL_DISCLAIMER = (
    "Educational reference only. Not a substitute for professional clinical "
    "judgement. Verify against primary sources before making patient-care decisions."
)


def build_chat_llm(settings: Settings, temperature: float = 0.1) -> ChatOpenAI:
    """Chat LLM factory pointed at NVIDIA's OpenAI-compatible endpoint.

    Every LLM built here carries the telemetry callback, so any ``.invoke()``
    from any LangGraph node is counted against the currently-active request.
    """
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
        temperature=temperature,
        max_tokens=4096,
        callbacks=[TELEMETRY_CALLBACK],
    )


def format_context(documents: Iterable[Document]) -> str:
    """Render retrieved chunks with numbered citation markers."""
    lines: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_name") or doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        header = f"[{i}] Source: {source}"
        if page is not None:
            header += f" (page {page})"
        lines.append(f"{header}\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(lines)


def format_citations(documents: Iterable[Document]) -> list[dict]:
    citations: list[dict] = []
    seen: set[tuple] = set()
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_name") or doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "id": i,
                "source": source,
                "page": page,
                "snippet": doc.page_content.strip()[:280],
            }
        )
    return citations


_SCORE_RE = re.compile(r"(?:score|confidence)\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_confidence(text: str) -> float:
    """Extract a 0-1 confidence value from a grader LLM response.

    Accepts either a bare float, a percentage, or a 'score: X' style line.
    Falls back to 0.5 if nothing sensible is found.
    """
    text = (text or "").strip()
    if not text:
        return 0.5
    match = _SCORE_RE.search(text)
    value: float | None = None
    if match:
        value = float(match.group(1))
    else:
        try:
            value = float(text.split()[0].rstrip("%,."))
        except (ValueError, IndexError):
            value = None
    if value is None:
        return 0.5
    if value > 1:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def label_confidence(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    if score >= 0.55:
        return "MODERATE"
    if score >= 0.3:
        return "LOW"
    return "VERY LOW"
