"""Corrective RAG (CRAG) agent (Yan et al., 2024).

Streaming flow (single async pass):
    retrieve → topic gate → grade → decide
        - generate      → stream(generate)
        - supplement    → rewrite → web search → stream(generate on local+web)
        - web_only      → rewrite → web search → stream(generate on web)
    → meta

Cost per query:
    - happy path      : 2 LLM calls (grade + streamed generate)
    - correction path : 3 LLM calls (grade + rewrite + streamed generate)
    - off-topic       : 0 LLM calls (refused from retrieval score alone)

Confidence is derived from the top cosine-similarity of the retrieved chunks
plus a cap when we had to fall back to unvetted web sources.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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
    id: int = Field(description="1-based index of the passage as shown in the prompt.")
    relevant: Literal["yes", "no"] = Field(
        description="'yes' if the passage helps answer the clinical question."
    )


class GradeBatch(BaseModel):
    grades: list[GradeItem] = Field(
        description="One entry per passage, in the same order as the prompt."
    )


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
            "Rules:\n"
            "- Cite each claim inline with bracketed markers [1], [2] matching the "
            "numbered passages.\n"
            "- If some context comes from web sources, note this and flag lower "
            "confidence.\n"
            "- If context is insufficient, say so explicitly.\n"
            "- Frame guidance as evidence summaries, not direct patient-care orders.\n"
            f"- Always end with the disclaimer: '{MEDICAL_DISCLAIMER}'",
        ),
        (
            "human",
            "Clinical question:\n{question}\n\nContext passages:\n{context}\n\n"
            "Write the evidence-based answer with inline citations.",
        ),
    ]
)


def _score_to_confidence(top_score: float) -> float:
    return max(0.0, min(1.0, top_score * 1.2))


class CRAGPipeline:
    RELEVANCE_THRESHOLD = 0.5  # fraction of relevant docs to skip correction
    WEB_ONLY_CONF_CAP = 0.6

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = load_vector_store(self.settings)
        self.llm = build_chat_llm(self.settings, temperature=0.1)
        self.grader = build_chat_llm(self.settings, temperature=0.0)
        self.rewriter = build_chat_llm(self.settings, temperature=0.2)
        self.web_search = self._build_web_search()

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
        except Exception:  # noqa: BLE001
            return None

    # ---------- sub-steps ----------
    def _grade_docs(self, question: str, docs: list[Document]) -> list[str]:
        if not docs:
            return []
        passages_block = "\n\n".join(
            f"[{i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )
        grader = self.grader.with_structured_output(GradeBatch)
        chain = GRADE_PROMPT | grader
        grades: list[str] = ["yes"] * len(docs)  # fail-open default
        try:
            result: GradeBatch = chain.invoke(
                {"question": question, "passages": passages_block}
            )
            for item in result.grades:
                idx = item.id - 1
                if 0 <= idx < len(docs):
                    grades[idx] = item.relevant
        except Exception:  # noqa: BLE001
            pass
        return grades

    def _decide_action(self, grades: list[str]) -> Literal["generate", "supplement", "web_only"]:
        if not grades:
            return "web_only"
        ratio = grades.count("yes") / len(grades)
        if ratio == 0:
            return "web_only"
        if ratio < self.RELEVANCE_THRESHOLD:
            return "supplement"
        return "generate"

    def _rewrite_query(self, question: str) -> str:
        chain = REWRITE_PROMPT | self.rewriter
        response = chain.invoke({"question": question})
        return response.content.strip().strip('"')

    def _web_lookup(self, query: str) -> list[Document]:
        if self.web_search is None:
            return []
        try:
            results = self.web_search.invoke({"query": query})
        except Exception:  # noqa: BLE001
            return []
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
        return web_docs

    # ---------- streaming pipeline ----------
    async def stream(self, question: str) -> AsyncIterator[dict]:
        logger.info("[CRAG] ▶ stream: %r", question[:80])
        original_question = question

        # 1. Retrieve
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
            async for ev in self._emit_no_context():
                yield ev
            return

        docs = [doc for doc, _ in pairs]
        top_score = max(score for _, score in pairs)
        yield {
            "type": "trace",
            "message": f"retrieved {len(docs)} passages (top cosine {top_score:.2f})",
        }

        # 2. Topic gate
        if top_score < threshold:
            yield {
                "type": "trace",
                "message": f"topic gate: {top_score:.2f} < {threshold} → refuse",
            }
            yield {"type": "token", "text": OFF_TOPIC_REFUSAL}
            yield {
                "type": "meta",
                "citations": [],
                "confidence": 0.0,
                "confidence_label": "OUT OF SCOPE",
                "disclaimer": MEDICAL_DISCLAIMER,
            }
            return

        # 3. Grade
        TELEMETRY.record_node("grade")
        grades = self._grade_docs(question, docs)
        kept = [doc for doc, g in zip(docs, grades) if g == "yes"]
        yield {
            "type": "trace",
            "message": f"graded relevance: {grades.count('yes')}/{len(grades)} relevant",
        }

        # 4. Decide
        action = self._decide_action(grades)
        yield {"type": "trace", "message": f"decision: {action}"}

        # 5. Correction: rewrite + web search
        web_docs: list[Document] = []
        if action != "generate":
            TELEMETRY.record_node("rewrite")
            rewritten = self._rewrite_query(question)
            yield {"type": "trace", "message": f"rewrote query → {rewritten!r}"}
            TELEMETRY.record_node("web_search")
            if self.web_search is None:
                yield {"type": "trace", "message": "web search skipped (no TAVILY_API_KEY)"}
            else:
                web_docs = self._web_lookup(rewritten)
                yield {
                    "type": "trace",
                    "message": f"web search retrieved {len(web_docs)} results",
                }

        # 6. Generate (streamed)
        merged = kept + web_docs
        if not merged:
            async for ev in self._emit_no_context():
                yield ev
            return

        TELEMETRY.record_node("generate")
        chain = ANSWER_PROMPT | self.llm
        try:
            async for chunk in chain.astream(
                {"question": original_question, "context": format_context(merged)}
            ):
                text = getattr(chunk, "content", "") or ""
                if text:
                    yield {"type": "token", "text": text}
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed")
            yield {"type": "error", "message": f"generation failed: {exc}"}
            return

        # 7. Meta with heuristic confidence
        confidence = _score_to_confidence(top_score)
        if not kept and web_docs:
            # web-only answer: cap confidence
            confidence = min(confidence, self.WEB_ONLY_CONF_CAP)
        yield {
            "type": "meta",
            "citations": format_citations(merged),
            "confidence": confidence,
            "confidence_label": label_confidence(confidence),
            "disclaimer": MEDICAL_DISCLAIMER,
        }
        logger.info("[CRAG] ✓ done → END")

    async def _emit_no_context(self) -> AsyncIterator[dict]:
        text = (
            "I could not retrieve relevant evidence for this question from "
            f"the indexed clinical sources or the web. {MEDICAL_DISCLAIMER}"
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
