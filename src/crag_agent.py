"""Corrective RAG (CRAG) agent (Yan et al., 2024).

Streaming flow (single async pass):
    retrieve → topic gate → grade → decide
        - generate      → stream(generate)
        - supplement    → rewrite → web search → stream(generate on local+web)
        - web_only      → rewrite → web search → stream(generate on web)
    → meta

The topic gate is two-stage: a retrieval score at or above
``Settings.topic_score_threshold`` is on-topic outright, and only a weaker
score costs a cheap medical-domain classification call. A medical question the
local corpus does not cover goes straight to the correction path — grading
passages already known to be weak would only buy an answer we are about to
replace — while a non-medical question is refused. Without Tavily there is no
correction to route to, so the gate refuses as it did before.

Cost per query:
    - happy path      : 2 LLM calls (grade + streamed generate)
    - correction path : 3 LLM calls (grade + rewrite + streamed generate)
    - weak evidence   : 3 LLM calls (classify + rewrite + streamed generate)
    - off-topic       : 1 LLM call (domain classification, then refusal)

Confidence is derived from the top cosine-similarity of the retrieved chunks
plus a cap when we had to fall back to unvetted web sources.

Concurrency: LLM steps and the Tavily tool use ``ainvoke``/``astream``
natively; blocking I/O (Pinecone retrieval) runs via ``asyncio.to_thread`` so
the event loop stays free while RAG and CRAG stream side by side.

Every LLM-backed step and the web search carry a LangSmith run name from
:func:`src.utils.run_identity`, so a run reads as a labelled waterfall.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .telemetry import TELEMETRY
from .utils import (
    DOMAIN_CLASSIFIER_NODE,
    MEDICAL_DISCLAIMER,
    PIPELINE_CRAG,
    RUN_NAME_CRAG_GENERATE,
    RUN_NAME_GRADE_PASSAGES,
    RUN_NAME_REWRITE_QUERY,
    RUN_NAME_WEB_SEARCH,
    build_chat_llm,
    build_domain_classifier,
    build_structured_llm,
    context_texts,
    format_citations,
    format_context,
    is_medical_question,
    label_confidence,
    off_topic_events,
    run_identity,
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


class RewrittenQuery(BaseModel):
    """Search query produced by the correction step — never an answer."""

    query: str = Field(
        description=(
            "Keyword-style web search query for medical literature: clinical "
            "terms, expanded acronyms and MeSH-style synonyms. Never a sentence "
            "that answers the question, and never ends with a full stop."
        )
    )


REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite clinical questions into search queries for medical "
            "literature. Produce a single, self-contained keyword query: "
            "preserve clinical intent, expand acronyms, and add MeSH-style "
            "synonyms where useful.\n"
            "Never answer the question, never define the subject, and never "
            "write a declarative sentence — you are building a search string, "
            "not replying to the user.\n"
            "Examples:\n"
            "Question: What is sepsis\n"
            "Query: sepsis septic shock definition pathophysiology diagnostic "
            "criteria SOFA score management guidelines\n"
            "Question: Is lecanemab effective in early Alzheimer disease?\n"
            "Query: lecanemab early Alzheimer disease efficacy randomized "
            "controlled trial amyloid cognitive decline outcomes",
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


def _web_grounded_confidence(web_doc_count: int, cap: float) -> float:
    """Confidence for an answer whose evidence came from the web, not the corpus.

    The local cosine score describes passages a web-corrected answer deliberately
    discarded, so reusing it would report how poorly the corpus matched rather
    than how well the answer is now grounded — penalising correction exactly when
    it worked. Scale with the number of web sources instead, and keep the cap:
    these sources are unvetted, so a web answer never outranks a corpus one.
    """
    if web_doc_count <= 0:
        return 0.0
    earned = WEB_BASE_CONFIDENCE + WEB_CONFIDENCE_PER_SOURCE * (web_doc_count - 1)
    return min(earned, cap)


# Signals that a "rewritten query" is actually an answer sentence: a search
# query is a bag of clinical terms, so it neither closes with sentence
# punctuation nor runs on for a paragraph. The word cap is deliberately
# generous — a long synonym list is still a usable query, and only a runaway
# explanation should trip it.
SENTENCE_TERMINATORS = (".", "!")
MAX_QUERY_WORDS = 60

# A web-only answer starts at MODERATE on a single source and climbs with
# corroboration, bounded by WEB_ONLY_CONF_CAP.
WEB_BASE_CONFIDENCE = 0.55
WEB_CONFIDENCE_PER_SOURCE = 0.02

REWRITE_FALLBACK_NOTE = " (kept the original question)"

WEB_SEARCH_MAX_RESULTS = 4
WEB_SEARCH_DEPTH = "advanced"
WEB_RESULTS_KEY = "results"
WEB_ERROR_KEY = "error"
WEB_RESULT_URL_KEY = "url"
WEB_RESULT_CONTENT_KEY = "content"
WEB_SOURCE_KIND = "web"
UNKNOWN_WEB_SOURCE = "web"


def _web_results(query: str, payload: object) -> list[dict]:
    """Pull the result rows out of a ``TavilySearch`` response.

    The tool answers with a mapping carrying a ``results`` list. A failed
    search comes back as an ``error`` mapping, and an empty one as the handled
    ``ToolException`` message (a plain string) — neither is a usable result
    set, so both are logged and treated as "no web evidence".
    """
    if not isinstance(payload, dict):
        logger.warning(
            "web search returned no usable results for %r: %s",
            query[:80],
            str(payload)[:200],
        )
        return []
    error = payload.get(WEB_ERROR_KEY)
    if error:
        logger.warning("web search errored for %r: %s", query[:80], error)
        return []
    return [
        result
        for result in payload.get(WEB_RESULTS_KEY) or []
        if isinstance(result, dict)
    ]


def _to_web_document(result: dict) -> Document:
    """One Tavily result as a Document with the citation metadata the UI needs."""
    url = result.get(WEB_RESULT_URL_KEY) or UNKNOWN_WEB_SOURCE
    return Document(
        page_content=result.get(WEB_RESULT_CONTENT_KEY) or "",
        metadata={"source": url, "source_name": url, "kind": WEB_SOURCE_KIND},
    )


def _is_prose_not_query(candidate: str) -> bool:
    """Whether the rewrite looks like an answer sentence instead of a query."""
    if not candidate:
        return True
    return candidate.endswith(SENTENCE_TERMINATORS) or (
        len(candidate.split()) > MAX_QUERY_WORDS
    )


class CRAGPipeline:
    RELEVANCE_THRESHOLD = 0.5  # fraction of relevant docs to skip correction
    WEB_ONLY_CONF_CAP = 0.6

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = load_vector_store(self.settings)
        self.llm = build_chat_llm(
            self.settings,
            temperature=0.1,
            identity=run_identity(RUN_NAME_CRAG_GENERATE, PIPELINE_CRAG),
        )
        self.grader = build_structured_llm(
            self.settings,
            GradeBatch,
            temperature=0.0,
            identity=run_identity(RUN_NAME_GRADE_PASSAGES, PIPELINE_CRAG),
        )
        self.rewriter = build_structured_llm(
            self.settings,
            RewrittenQuery,
            temperature=0.2,
            identity=run_identity(RUN_NAME_REWRITE_QUERY, PIPELINE_CRAG),
        )
        self.domain_classifier = build_domain_classifier(self.settings, PIPELINE_CRAG)
        self.web_search = self._build_web_search()

    # ---------- optional web search tool ----------
    def _build_web_search(self):
        if not self.settings.tavily_api_key:
            return None
        try:
            from langchain_tavily import TavilySearch

            return TavilySearch(
                max_results=WEB_SEARCH_MAX_RESULTS,
                tavily_api_key=self.settings.tavily_api_key,
                search_depth=WEB_SEARCH_DEPTH,
                include_answer=False,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Tavily web search unavailable; CRAG will run without web fallback",
                exc_info=True,
            )
            return None

    # ---------- sub-steps ----------
    async def _grade_docs(self, question: str, docs: list[Document]) -> list[str]:
        if not docs:
            return []
        passages_block = "\n\n".join(
            f"[{i + 1}]\n{doc.page_content}" for i, doc in enumerate(docs)
        )
        chain = GRADE_PROMPT | self.grader
        grades: list[str] = ["yes"] * len(docs)  # fail-open default
        try:
            result: GradeBatch = await chain.ainvoke(
                {"question": question, "passages": passages_block}
            )
            for item in result.grades:
                idx = item.id - 1
                if 0 <= idx < len(docs):
                    grades[idx] = item.relevant
        except Exception:  # noqa: BLE001
            logger.warning(
                "relevance grading failed; failing open (all %d passages kept)",
                len(docs),
                exc_info=True,
            )
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

    async def _rewrite_query(self, question: str) -> str:
        """Turn the question into a web search query, or keep the original.

        Structured output forces the model to emit a ``query`` field instead of
        free prose, and the prose guard catches the residual failure mode where
        it answers the question inside that field — searching on an answer
        poisons the web results, so the original question is the safer input.
        """
        chain = REWRITE_PROMPT | self.rewriter
        try:
            rewritten: RewrittenQuery = await chain.ainvoke({"question": question})
        except Exception:  # noqa: BLE001
            logger.warning(
                "query rewrite failed for %r; searching the original question",
                question[:80],
                exc_info=True,
            )
            return question
        candidate = rewritten.query.strip().strip('"')
        if _is_prose_not_query(candidate):
            logger.warning(
                "query rewrite returned prose rather than a search query (%r); "
                "searching the original question instead",
                candidate[:160],
            )
            return question
        return candidate

    async def _web_lookup(self, query: str) -> list[Document]:
        if self.web_search is None:
            return []
        try:
            payload = await self.web_search.ainvoke(
                {"query": query},
                run_identity(RUN_NAME_WEB_SEARCH, PIPELINE_CRAG),
            )
        except Exception:  # noqa: BLE001
            logger.warning("web search failed for query %r", query[:80], exc_info=True)
            return []
        return [_to_web_document(result) for result in _web_results(query, payload)]

    # ---------- streaming pipeline ----------
    async def stream(self, question: str) -> AsyncIterator[dict]:
        logger.info("[CRAG] ▶ stream: %r", question[:80])
        original_question = question

        # 1. Retrieve
        TELEMETRY.record_node("retrieve")
        try:
            pairs = await asyncio.to_thread(
                self.vector_store.similarity_search_with_score,
                question,
                k=self.settings.top_k,
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

        # 2. Topic gate — weak retrieval only costs a domain classification call
        kept: list[Document] = []
        if top_score < threshold:
            yield {
                "type": "trace",
                "message": (
                    f"topic gate: {top_score:.2f} < {threshold} → "
                    "classifying medical domain"
                ),
            }
            TELEMETRY.record_node(DOMAIN_CLASSIFIER_NODE)
            if not await is_medical_question(self.domain_classifier, question):
                yield {"type": "trace", "message": "not a medical question → refuse"}
                for event in off_topic_events():
                    yield event
                return
            if self.web_search is None:
                yield {
                    "type": "trace",
                    "message": (
                        "question is medical but below threshold, and no "
                        "TAVILY_API_KEY → no correction available → refuse"
                    ),
                }
                for event in off_topic_events():
                    yield event
                return
            action: Literal["generate", "supplement", "web_only"] = "web_only"
            yield {
                "type": "trace",
                "message": (
                    "question is medical but not in the local corpus → skipping "
                    f"grading, correcting via web (decision: {action})"
                ),
            }
        else:
            # 3. Grade
            TELEMETRY.record_node("grade")
            grades = await self._grade_docs(question, docs)
            kept = [doc for doc, g in zip(docs, grades) if g == "yes"]
            yield {
                "type": "trace",
                "message": (
                    f"graded relevance: {grades.count('yes')}/{len(grades)} relevant"
                ),
            }

            # 4. Decide
            action = self._decide_action(grades)
            yield {"type": "trace", "message": f"decision: {action}"}

        # 5. Correction: rewrite + web search
        web_docs: list[Document] = []
        if action != "generate":
            TELEMETRY.record_node("rewrite")
            rewritten = await self._rewrite_query(question)
            fallback_note = "" if rewritten != question else REWRITE_FALLBACK_NOTE
            yield {
                "type": "trace",
                "message": f"rewrote query → {rewritten!r}{fallback_note}",
            }
            TELEMETRY.record_node("web_search")
            if self.web_search is None:
                yield {"type": "trace", "message": "web search skipped (no TAVILY_API_KEY)"}
            else:
                web_docs = await self._web_lookup(rewritten)
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
        if not kept and web_docs:
            confidence = _web_grounded_confidence(len(web_docs), self.WEB_ONLY_CONF_CAP)
        else:
            confidence = _score_to_confidence(top_score)
        yield {
            "type": "meta",
            "citations": format_citations(merged),
            "contexts": context_texts(merged),
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
            "contexts": [],
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
