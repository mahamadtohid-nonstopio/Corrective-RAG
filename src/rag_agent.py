"""Standard Retrieval-Augmented Generation agent built on LangGraph.

Graph:
    retrieve → (topic gate: score < TOPIC_SCORE_THRESHOLD) → refuse → END
             → generate (answer + self-scored confidence in one call) → END

Cost per query:
    - happy path : 1 LLM call
    - off-topic  : 0 LLM calls (refused from retrieval score alone)
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

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


class RAGState(TypedDict, total=False):
    question: str
    documents: list[Document]
    top_score: float
    off_topic: bool
    answer: str
    citations: list[dict]
    confidence: float
    confidence_label: str
    trace: list[str]


class AnsweredResponse(BaseModel):
    """Structured output for the answer LLM — returns answer + confidence together."""

    answer: str = Field(
        description=(
            "Evidence-based answer with inline bracket citations like [1], [2] "
            "referring to the numbered context passages. End with the disclaimer."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your honest self-assessment on a 0-1 scale of how directly the "
            "context passages support the answer. 1.0 = every claim is verbatim "
            "supported. 0.0 = the context does not support the answer."
        ),
    )


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Healthcare Knowledge Navigator assisting clinicians.\n"
            "Answer strictly from the provided context of clinical guidelines, "
            "research papers, and treatment protocols.\n"
            "Rules for the 'answer' field:\n"
            "- Cite claims inline with bracketed numbers like [1], [2] matching the "
            "numbered context passages.\n"
            "- If the context is insufficient, say so explicitly rather than guess.\n"
            "- Do not provide direct patient-care instructions; frame guidance as "
            "evidence summaries for a healthcare professional.\n"
            "- Prefer specificity: drug names, dosages, thresholds, patient populations, "
            "recommendation strength.\n"
            f"- End the answer with the disclaimer: '{MEDICAL_DISCLAIMER}'\n"
            "Rules for the 'confidence' field:\n"
            "- Score 0.0-1.0 for how directly the context supports every claim.\n"
            "- Be strict: partial support ≈ 0.5, unsupported claims lower.",
        ),
        (
            "human",
            "Clinical question:\n{question}\n\n"
            "Context passages:\n{context}\n\n"
            "Return the answer and confidence.",
        ),
    ]
)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = load_vector_store(self.settings)
        self.llm = build_chat_llm(self.settings, temperature=0.1)
        self.graph = self._build_graph()

    # ---------- nodes ----------
    def _retrieve(self, state: RAGState) -> RAGState:
        pairs = self.vector_store.similarity_search_with_score(
            state["question"], k=self.settings.top_k
        )
        threshold = self.settings.topic_score_threshold
        trace = state.get("trace", [])
        if not pairs:
            trace = trace + ["retrieved 0 passages → off-topic"]
            return {"documents": [], "top_score": 0.0, "off_topic": True, "trace": trace}
        docs = [doc for doc, _ in pairs]
        top_score = max(score for _, score in pairs)
        off_topic = top_score < threshold
        trace = trace + [
            f"retrieved {len(docs)} passages (top cosine {top_score:.2f})"
        ]
        if off_topic:
            trace = trace + [
                f"topic gate: {top_score:.2f} < {threshold} → refuse"
            ]
        return {
            "documents": docs,
            "top_score": top_score,
            "off_topic": off_topic,
            "trace": trace,
        }

    def _refuse(self, state: RAGState) -> RAGState:
        return {
            "answer": OFF_TOPIC_REFUSAL,
            "citations": [],
            "confidence": 0.0,
            "confidence_label": "OUT OF SCOPE",
        }

    def _generate(self, state: RAGState) -> RAGState:
        docs = state.get("documents") or []
        if not docs:
            return {
                "answer": (
                    "I could not find relevant information in the indexed clinical "
                    "sources to answer this question. "
                    f"{MEDICAL_DISCLAIMER}"
                ),
                "citations": [],
                "confidence": 0.0,
                "confidence_label": label_confidence(0.0),
            }
        answerer = self.llm.with_structured_output(AnsweredResponse)
        chain = ANSWER_PROMPT | answerer
        try:
            result: AnsweredResponse = chain.invoke(
                {"question": state["question"], "context": format_context(docs)}
            )
            answer = result.answer.strip()
            confidence = float(result.confidence)
        except Exception:  # noqa: BLE001 — degrade to a plain-text answer call
            plain = ANSWER_PROMPT | self.llm
            response = plain.invoke(
                {"question": state["question"], "context": format_context(docs)}
            )
            answer = response.content.strip()
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        return {
            "answer": answer,
            "citations": format_citations(docs),
            "confidence": confidence,
            "confidence_label": label_confidence(confidence),
        }

    # ---------- routing ----------
    @staticmethod
    def _route_after_retrieve(state: RAGState) -> str:
        return "refuse" if state.get("off_topic") else "generate"

    # ---------- graph ----------
    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("refuse", self._refuse)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"refuse": "refuse", "generate": "generate"},
        )
        graph.add_edge("refuse", END)
        graph.add_edge("generate", END)
        return graph.compile()

    def ask(self, question: str) -> dict:
        logger.info("\n[RAG] ▶ starting: %r", question[:80])
        final: dict = {"question": question}
        prev = "START"
        for step in self.graph.stream(
            {"question": question, "trace": []}, stream_mode="updates"
        ):
            for node_name, update in step.items():
                logger.info("[RAG]     %s → %s", prev, node_name)
                TELEMETRY.record_node(node_name)
                prev = node_name
                if isinstance(update, dict):
                    final.update(update)
        current = TELEMETRY.current()
        calls = current.llm_calls if current else 0
        logger.info("[RAG] ✓ done → END  (%d LLM calls)\n", calls)
        return final
