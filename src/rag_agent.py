"""Standard Retrieval-Augmented Generation agent built on LangGraph.

Graph:
    retrieve → generate → score_confidence → END
"""
from __future__ import annotations

import logging
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .config import Settings, get_settings
from .telemetry import TELEMETRY
from .utils import (
    MEDICAL_DISCLAIMER,
    build_chat_llm,
    format_citations,
    format_context,
    label_confidence,
    parse_confidence,
)
from .vector_store import load_vector_store

logger = logging.getLogger("hkn.rag")


class RAGState(TypedDict, total=False):
    question: str
    documents: list[Document]
    answer: str
    citations: list[dict]
    confidence: float
    confidence_label: str


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Healthcare Knowledge Navigator assisting clinicians.\n"
            "Answer strictly from the provided context of clinical guidelines, "
            "research papers, and treatment protocols.\n"
            "Rules:\n"
            "- Cite claims inline with bracketed numbers like [1], [2] that match the "
            "context passages.\n"
            "- If the context is insufficient, say so explicitly rather than guess.\n"
            "- Do not provide direct patient care instructions; frame guidance as "
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

CONFIDENCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You audit medical answers. Return ONLY a number between 0 and 1 "
            "indicating how well the answer is grounded in the provided context, "
            "where 1 means every claim is directly supported and 0 means the context "
            "does not support the answer.",
        ),
        (
            "human",
            "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\n"
            "Confidence score (0-1):",
        ),
    ]
)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = load_vector_store(self.settings)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.settings.top_k}
        )
        self.llm = build_chat_llm(self.settings, temperature=0.1)
        self.grader = build_chat_llm(self.settings, temperature=0.0)
        self.graph = self._build_graph()

    # ---------- nodes ----------
    def _retrieve(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state["question"])
        return {"documents": docs}

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
            }
        chain = ANSWER_PROMPT | self.llm
        response = chain.invoke(
            {"question": state["question"], "context": format_context(docs)}
        )
        return {
            "answer": response.content.strip(),
            "citations": format_citations(docs),
        }

    def _score_confidence(self, state: RAGState) -> RAGState:
        docs = state.get("documents") or []
        if not docs or not state.get("answer"):
            return {"confidence": 0.0, "confidence_label": label_confidence(0.0)}
        chain = CONFIDENCE_PROMPT | self.grader
        raw = chain.invoke(
            {
                "question": state["question"],
                "context": format_context(docs),
                "answer": state["answer"],
            }
        )
        score = parse_confidence(raw.content)
        return {"confidence": score, "confidence_label": label_confidence(score)}

    # ---------- graph ----------
    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("score_confidence", self._score_confidence)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "score_confidence")
        graph.add_edge("score_confidence", END)
        return graph.compile()

    def ask(self, question: str) -> dict:
        logger.info("\n[RAG] ▶ starting: %r", question[:80])
        final: dict = {"question": question}
        prev = "START"
        for step in self.graph.stream({"question": question}, stream_mode="updates"):
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
