"""Corrective RAG (CRAG) agent implemented as a LangGraph.

Pipeline (Yan et al., 2024, "Corrective Retrieval Augmented Generation"), with
two cost optimisations:

    retrieve (with cosine scores) → topic gate
        - top score < TOPIC_SCORE_THRESHOLD → refuse (no LLM calls)
    → grade each doc → decide action
        - all relevant       → generate
        - partially relevant → rewrite → web search → merge → generate
        - all irrelevant     → rewrite → web search only → generate
    → END

The generator uses structured output and returns (answer, confidence) in a
single LLM call — the separate audit call is folded away.

Cost per query:
    - happy path      : 2 LLM calls (grade + answer/confidence)
    - correction path : 3 LLM calls (grade + rewrite + answer/confidence)
    - off-topic       : 0 LLM calls (refused from retrieval score alone)
"""
from __future__ import annotations

import logging
from typing import Literal, TypedDict

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

logger = logging.getLogger("hkn.crag")


class GradeItem(BaseModel):
    """Relevance judgment for a single retrieved passage inside a batch."""

    id: int = Field(description="1-based index of the passage as shown in the prompt.")
    relevant: Literal["yes", "no"] = Field(
        description="'yes' if the passage helps answer the clinical question."
    )


class GradeBatch(BaseModel):
    """All relevance judgments returned by a single LLM call."""

    grades: list[GradeItem] = Field(
        description="One entry per passage, in the same order as the prompt."
    )


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
            "Self-assessment 0-1 for how well the context supports the answer. "
            "1.0 = every claim directly supported; 0.0 = unsupported."
        ),
    )


class CRAGState(TypedDict, total=False):
    question: str
    original_question: str
    documents: list[Document]
    web_documents: list[Document]
    top_score: float
    off_topic: bool
    grades: list[str]
    action: Literal["generate", "supplement", "web_only", "refuse"]
    answer: str
    citations: list[dict]
    confidence: float
    confidence_label: str
    trace: list[str]


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You judge whether each retrieved passage is relevant to a clinical "
            "question. For every passage return a JSON entry with its id and "
            "'yes' or 'no'. Return one entry per passage, in the same order as "
            "the input. Do NOT skip any.",
        ),
        (
            "human",
            "Clinical question:\n{question}\n\nPassages:\n{passages}\n\n"
            "Return the batched relevance judgments now.",
        ),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite clinical questions to improve retrieval. "
            "Produce a single, self-contained query optimised for search over "
            "medical literature. Preserve clinical intent, expand acronyms, and "
            "add MeSH-style synonyms where useful. Return only the rewritten query.",
        ),
        ("human", "Original question:\n{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Healthcare Knowledge Navigator assisting clinicians. "
            "Answer strictly from the provided context.\n"
            "Rules for the 'answer' field:\n"
            "- Cite each claim inline with bracketed markers [1], [2] matching the "
            "numbered passages.\n"
            "- If some context comes from web sources, note this and lower confidence.\n"
            "- If context is insufficient, say so explicitly.\n"
            "- Frame guidance as evidence summaries, not direct patient-care orders.\n"
            f"- End the answer with the disclaimer: '{MEDICAL_DISCLAIMER}'\n"
            "Rules for the 'confidence' field:\n"
            "- Score 0.0-1.0 for how directly the context supports every claim.\n"
            "- Be strict: partial support ≈ 0.5, unsupported claims lower.",
        ),
        (
            "human",
            "Clinical question:\n{question}\n\nContext passages:\n{context}\n\n"
            "Return the answer and confidence.",
        ),
    ]
)


class CRAGPipeline:
    RELEVANCE_THRESHOLD = 0.5  # fraction of relevant docs to skip correction
    WEB_ONLY_CONF_CAP = 0.6    # penalise unvetted web-only answers

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = load_vector_store(self.settings)
        self.llm = build_chat_llm(self.settings, temperature=0.1)
        self.grader = build_chat_llm(self.settings, temperature=0.0)
        self.rewriter = build_chat_llm(self.settings, temperature=0.2)
        self.web_search = self._build_web_search()
        self.graph = self._build_graph()

    # ---------- optional web search tool ----------
    def _build_web_search(self):
        if not self.settings.tavily_api_key:
            return None
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            return TavilySearchResults(
                max_results=4,
                tavily_api_key=self.settings.tavily_api_key,
                search_depth="advanced",
                include_answer=False,
            )
        except Exception:  # noqa: BLE001 - degrade gracefully at runtime
            return None

    # ---------- graph nodes ----------
    def _retrieve(self, state: CRAGState) -> CRAGState:
        pairs = self.vector_store.similarity_search_with_score(
            state["question"], k=self.settings.top_k
        )
        threshold = self.settings.topic_score_threshold
        trace = state.get("trace", [])
        if not pairs:
            return {
                "documents": [],
                "top_score": 0.0,
                "off_topic": True,
                "trace": trace + ["retrieved 0 passages → off-topic"],
            }
        docs = [doc for doc, _ in pairs]
        top_score = max(score for _, score in pairs)
        off_topic = top_score < threshold
        trace = trace + [f"retrieved {len(docs)} passages (top cosine {top_score:.2f})"]
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

    def _refuse(self, state: CRAGState) -> CRAGState:
        return {
            "answer": OFF_TOPIC_REFUSAL,
            "citations": [],
            "confidence": 0.0,
            "confidence_label": "OUT OF SCOPE",
        }

    def _grade(self, state: CRAGState) -> CRAGState:
        docs = state.get("documents") or []
        if not docs:
            return {"documents": [], "grades": [],
                    "trace": state.get("trace", []) + ["graded relevance: 0/0"]}

        # Batch all passages into a single structured-output LLM call.
        passages_block = "\n\n".join(
            f"[{i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )
        grader = self.grader.with_structured_output(GradeBatch)
        chain = GRADE_PROMPT | grader
        grades: list[str] = ["yes"] * len(docs)  # fail-open default
        try:
            result: GradeBatch = chain.invoke(
                {"question": state["question"], "passages": passages_block}
            )
            for item in result.grades:
                idx = item.id - 1
                if 0 <= idx < len(docs):
                    grades[idx] = item.relevant
        except Exception:  # noqa: BLE001
            pass  # keep the fail-open defaults so recall isn't destroyed

        kept = [doc for doc, g in zip(docs, grades) if g == "yes"]
        trace = state.get("trace", []) + [
            f"graded relevance: {grades.count('yes')}/{len(grades)} relevant (1 batched LLM call)"
        ]
        return {"documents": kept, "grades": grades, "trace": trace}

    def _decide(self, state: CRAGState) -> CRAGState:
        grades = state.get("grades") or []
        if not grades:
            action = "web_only"
        else:
            ratio = grades.count("yes") / len(grades)
            if ratio == 0:
                action = "web_only"
            elif ratio < self.RELEVANCE_THRESHOLD:
                action = "supplement"
            else:
                action = "generate"
        trace = state.get("trace", []) + [f"decision: {action}"]
        return {"action": action, "trace": trace}

    def _rewrite(self, state: CRAGState) -> CRAGState:
        chain = REWRITE_PROMPT | self.rewriter
        response = chain.invoke({"question": state["question"]})
        rewritten = response.content.strip().strip('"')
        trace = state.get("trace", []) + [f"rewrote query → {rewritten!r}"]
        return {
            "original_question": state.get("original_question") or state["question"],
            "question": rewritten,
            "trace": trace,
        }

    def _web_search(self, state: CRAGState) -> CRAGState:
        trace = state.get("trace", [])
        if self.web_search is None:
            trace = trace + ["web search skipped (no TAVILY_API_KEY)"]
            return {"web_documents": [], "trace": trace}
        try:
            results = self.web_search.invoke({"query": state["question"]})
        except Exception as exc:  # noqa: BLE001
            trace = trace + [f"web search failed: {exc}"]
            return {"web_documents": [], "trace": trace}
        web_docs: list[Document] = []
        for item in results or []:
            content = item.get("content") if isinstance(item, dict) else str(item)
            url = item.get("url") if isinstance(item, dict) else "web"
            web_docs.append(
                Document(
                    page_content=content or "",
                    metadata={"source": url, "source_name": url, "kind": "web"},
                )
            )
        trace = trace + [f"web search retrieved {len(web_docs)} results"]
        return {"web_documents": web_docs, "trace": trace}

    def _generate(self, state: CRAGState) -> CRAGState:
        docs = (state.get("documents") or []) + (state.get("web_documents") or [])
        question = state.get("original_question") or state["question"]
        if not docs:
            return {
                "answer": (
                    "I could not retrieve relevant evidence for this question from "
                    "the indexed clinical sources or the web. "
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
                {"question": question, "context": format_context(docs)}
            )
            answer = result.answer.strip()
            confidence = float(result.confidence)
        except Exception:  # noqa: BLE001 — degrade to a plain-text answer call
            plain = ANSWER_PROMPT | self.llm
            response = plain.invoke(
                {"question": question, "context": format_context(docs)}
            )
            answer = response.content.strip()
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        # Cap confidence when we only had web sources (no vetted local docs).
        if not state.get("documents") and state.get("web_documents"):
            confidence = min(confidence, self.WEB_ONLY_CONF_CAP)
        return {
            "answer": answer,
            "citations": format_citations(docs),
            "confidence": confidence,
            "confidence_label": label_confidence(confidence),
        }

    # ---------- routing ----------
    @staticmethod
    def _route_after_retrieve(state: CRAGState) -> str:
        return "refuse" if state.get("off_topic") else "grade"

    @staticmethod
    def _route_from_decision(state: CRAGState) -> str:
        action = state.get("action", "generate")
        if action == "generate":
            return "generate"
        return "rewrite"  # supplement + web_only both need query rewriting

    # ---------- graph wiring ----------
    def _build_graph(self):
        graph = StateGraph(CRAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("refuse", self._refuse)
        graph.add_node("grade", self._grade)
        graph.add_node("decide", self._decide)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("web_search", self._web_search)
        graph.add_node("generate", self._generate)

        graph.set_entry_point("retrieve")
        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {"refuse": "refuse", "grade": "grade"},
        )
        graph.add_edge("refuse", END)
        graph.add_edge("grade", "decide")
        graph.add_conditional_edges(
            "decide",
            self._route_from_decision,
            {"generate": "generate", "rewrite": "rewrite"},
        )
        graph.add_edge("rewrite", "web_search")
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def ask(self, question: str) -> dict:
        logger.info("\n[CRAG] ▶ starting: %r", question[:80])
        final: dict = {"question": question, "original_question": question}
        prev = "START"
        for step in self.graph.stream(
            {"question": question, "original_question": question, "trace": []},
            stream_mode="updates",
        ):
            for node_name, update in step.items():
                extra = ""
                if isinstance(update, dict) and "action" in update:
                    extra = f" (action={update['action']})"
                logger.info("[CRAG]    %s → %s%s", prev, node_name, extra)
                TELEMETRY.record_node(node_name)
                prev = node_name
                if isinstance(update, dict):
                    final.update(update)
        current = TELEMETRY.current()
        calls = current.llm_calls if current else 0
        logger.info("[CRAG] ✓ done → END  (%d LLM calls)\n", calls)
        return final
