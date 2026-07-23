"""Standard Retrieval-Augmented Generation agent.

Streaming flow (single async pass):
    retrieve → topic gate (retrieval-score) → stream(generate) → meta

Cost per query:
    - happy path : 1 LLM call (streamed answer)
    - off-topic  : 0 LLM calls (refused from retrieval score alone)

Confidence is derived from the top cosine-similarity of the retrieved chunks,
not from a separate LLM audit — that removes an extra round-trip while giving
a more calibrated signal (the LLM was not a good judge of its own output).
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from langchain_core.prompts import ChatPromptTemplate

from .config import Settings, get_settings
from .telemetry import TELEMETRY
from .utils import (
    MEDICAL_DISCLAIMER,
    OFF_TOPIC_REFUSAL,
    build_chat_llm,
    format_citations,
    format_context,
    label_confidence,
)
from .vector_store import load_vector_store

logger = logging.getLogger("hkn.rag")


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Healthcare Knowledge Navigator assisting clinicians.\n"
            "Answer strictly from the provided context of clinical guidelines, "
            "research papers, and treatment protocols.\n"
            "Rules:\n"
            "- Cite claims inline with bracketed numbers like [1], [2] matching the "
            "numbered context passages.\n"
            "- If the context is insufficient, say so explicitly rather than guess.\n"
            "- Do not provide direct patient-care instructions; frame guidance as "
            "evidence summaries for a healthcare professional.\n"
            "- Prefer specificity: drug names, dosages, thresholds, patient populations, "
            "recommendation strength.\n"
            f"- Always end with the disclaimer: '{MEDICAL_DISCLAIMER}'",
        ),
        (
            "human",
            "Clinical question:\n{question}\n\n"
            "Context passages:\n{context}\n\n"
            "Write the evidence-based answer with inline citations.",
        ),
    ]
)


def _score_to_confidence(top_score: float) -> float:
    """Map top retrieval cosine similarity to a 0-1 confidence value."""
    return max(0.0, min(1.0, top_score * 1.2))


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = load_vector_store(self.settings)
        self.llm = build_chat_llm(self.settings, temperature=0.1)

    async def stream(self, question: str) -> AsyncIterator[dict]:
        """Yield events describing the pipeline's progress in real time.

        Event shapes:
            {"type": "trace",    "message": "..."}
            {"type": "token",    "text":    "..."}    (only for streamed answer)
            {"type": "meta",     "citations":[...], "confidence":..., "confidence_label":..., "disclaimer": "..."}
            {"type": "error",    "message": "..."}
        """
        logger.info("[RAG] ▶ stream: %r", question[:80])
        TELEMETRY.record_node("retrieve")
        try:
            pairs = self.vector_store.similarity_search_with_score(
                question, k=self.settings.top_k
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("retrieval failed")
            yield {"type": "error", "message": f"retrieval failed: {exc}"}
            return

        threshold = self.settings.topic_score_threshold
        if not pairs:
            yield {"type": "trace", "message": "retrieved 0 passages"}
            async for ev in self._emit_no_context(question):
                yield ev
            return

        docs = [doc for doc, _ in pairs]
        top_score = max(score for _, score in pairs)
        yield {
            "type": "trace",
            "message": f"retrieved {len(docs)} passages (top cosine {top_score:.2f})",
        }

        if top_score < threshold:
            yield {
                "type": "trace",
                "message": f"topic gate: {top_score:.2f} < {threshold} → refuse",
            }
            # Emit refusal answer as one token so the frontend still gets a stream.
            yield {"type": "token", "text": OFF_TOPIC_REFUSAL}
            yield {
                "type": "meta",
                "citations": [],
                "confidence": 0.0,
                "confidence_label": "OUT OF SCOPE",
                "disclaimer": MEDICAL_DISCLAIMER,
            }
            return

        # Stream the generation.
        TELEMETRY.record_node("generate")
        chain = ANSWER_PROMPT | self.llm
        try:
            async for chunk in chain.astream(
                {"question": question, "context": format_context(docs)}
            ):
                text = getattr(chunk, "content", "") or ""
                if text:
                    yield {"type": "token", "text": text}
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            yield {"type": "error", "message": f"generation failed: {exc}"}
            return

        confidence = _score_to_confidence(top_score)
        yield {
            "type": "meta",
            "citations": format_citations(docs),
            "confidence": confidence,
            "confidence_label": label_confidence(confidence),
            "disclaimer": MEDICAL_DISCLAIMER,
        }
        logger.info("[RAG] ✓ done → END")

    async def _emit_no_context(self, question: str) -> AsyncIterator[dict]:
        text = (
            "I could not find relevant information in the indexed clinical "
            f"sources to answer this question. {MEDICAL_DISCLAIMER}"
        )
        yield {"type": "token", "text": text}
        yield {
            "type": "meta",
            "citations": [],
            "confidence": 0.0,
            "confidence_label": label_confidence(0.0),
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    # ---------- async collector (kept for /api/ask non-streaming compat) ----------
    async def ask(self, question: str) -> dict:
        result: dict = {
            "question": question,
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "confidence_label": "VERY LOW",
            "trace": [],
        }
        async for event in self.stream(question):
            t = event.get("type")
            if t == "trace":
                result["trace"].append(event["message"])
            elif t == "token":
                result["answer"] += event["text"]
            elif t == "meta":
                for k in ("citations", "confidence", "confidence_label"):
                    if k in event:
                        result[k] = event[k]
            elif t == "error":
                raise RuntimeError(event["message"])
        return result
