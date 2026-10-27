"""Shared helpers: LLM factory, topic gate, citation formatting, confidence scoring."""
from __future__ import annotations

import logging
from typing import Iterable, Literal, Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import Settings
from .telemetry import TELEMETRY_CALLBACK

logger = logging.getLogger("hkn.gate")

CHAT_MAX_TOKENS = 4096

# NVIDIA NIM hybrid-reasoning models (nemotron-3.5-*) default to emitting a long
# hidden reasoning pass before the answer. That pass can consume the entire token
# budget — leaving an empty response and breaking structured output — and adds
# tens of seconds of latency. This template flag turns it off; see
# Settings.model_thinking_enabled.
THINKING_DISABLED_BODY = {"chat_template_kwargs": {"thinking": False}}

MEDICAL_DISCLAIMER = (
    "Educational reference only. Not a substitute for professional clinical "
    "judgement. Verify against primary sources before making patient-care decisions."
)

OFF_TOPIC_REFUSAL = (
    "This question is outside the scope of the Healthcare Knowledge Navigator, "
    "which answers evidence-cited clinical questions from indexed medical sources. "
    "Please rephrase as a question about diagnosis, treatment, pharmacology, "
    "clinical guidelines, or another medical topic."
)

OUT_OF_SCOPE_LABEL = "OUT OF SCOPE"


DOMAIN_CLASSIFIER_NODE = "classify_domain"

PIPELINE_RAG = "rag"
PIPELINE_CRAG = "crag"

# Stable, greppable LangSmith run names — one per LLM-backed step, so a run
# reads as a labelled waterfall instead of a stack of "ChatOpenAI" rows.
RUN_NAME_CLASSIFY_DOMAIN = "gate.classify_domain"
RUN_NAME_GRADE_PASSAGES = "crag.grade_passages"
RUN_NAME_REWRITE_QUERY = "crag.rewrite_query"
RUN_NAME_WEB_SEARCH = "crag.web_search"
RUN_NAME_CRAG_GENERATE = "crag.generate"
RUN_NAME_RAG_GENERATE = "rag.generate"

PIPELINE_MODE_METADATA_KEY = "pipeline_mode"
STEP_METADATA_KEY = "step"
_RUN_NAME_SEPARATOR = "."


def run_identity(run_name: str, pipeline: str) -> RunnableConfig:
    """LangSmith identity for one step: run name, filter tags and pipeline mode.

    The step tag is the part of the run name after the prefix, so
    ``crag.grade_passages`` is filterable in the LangSmith UI as both ``crag``
    and ``grade_passages``.
    """
    step = run_name.rsplit(_RUN_NAME_SEPARATOR, 1)[-1]
    return {
        "run_name": run_name,
        "tags": [pipeline, step],
        "metadata": {PIPELINE_MODE_METADATA_KEY: pipeline, STEP_METADATA_KEY: step},
    }


def _apply_run_identity(
    runnable: Runnable, identity: RunnableConfig | None
) -> Runnable:
    """Attach ``identity`` where it lands on the model's own LangSmith run.

    ``RunnableSequence`` deletes ``run_name`` from the config it hands to its
    steps, so an identity set on a composed chain labels only the outer run and
    the nested chat-model run stays ``ChatOpenAI``. Binding it to the model
    itself — or to the model step of the sequence ``with_structured_output``
    builds — makes the identity part of the model's own merged config, so it
    survives both composition in ``__init__`` and invocation per request.
    """
    if identity is None:
        return runnable
    if isinstance(runnable, RunnableSequence):
        model_step, *downstream_steps = runnable.steps
        named: Runnable = model_step.with_config(identity)
        for step in downstream_steps:
            named = named | step
        return named
    return runnable.with_config(identity)


class DomainVerdict(BaseModel):
    """Structured answer of the medical-domain classifier."""

    medical: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the question is about medicine, health or clinical care; "
            "'no' for any other subject."
        )
    )


DOMAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You decide whether a question belongs to the medical domain. "
            "Answer 'yes' for anything about human health: diagnosis, treatment, "
            "pharmacology, drugs and dosing, clinical guidelines, diseases, "
            "procedures, laboratory results, medical devices or public health — "
            "even if the specific topic is rare, very recent, or you do not know "
            "the answer. Answer 'no' for every other subject, such as cooking, "
            "travel, sport, finance or general programming. "
            "Judge only the subject area, never whether you can answer.",
        ),
        ("human", "Question:\n{question}\n\nIs this a medical question?"),
    ]
)


def build_domain_classifier(settings: Settings, pipeline: str) -> Runnable:
    """Chain that labels a question as medical or not, built once per pipeline.

    Returns ``DomainVerdict`` instances, so the answer is parseable without
    string sniffing, and carries the telemetry callback like every other chain
    built by the factories in this module. ``pipeline`` is the calling mode
    (``rag``/``crag``), recorded on the LangSmith run so the shared gate can be
    told apart per pipeline.
    """
    return DOMAIN_PROMPT | build_structured_llm(
        settings,
        DomainVerdict,
        identity=run_identity(RUN_NAME_CLASSIFY_DOMAIN, pipeline),
    )


async def is_medical_question(classifier: Runnable, question: str) -> bool:
    """Return whether ``question`` is a medical/clinical question.

    Used only as the second stage of the topic gate, when retrieval similarity
    alone is too weak to tell an off-topic question from an on-topic one that
    simply is not covered by the indexed corpus. Costs one cheap structured
    LLM call, counted by telemetry.

    On classifier failure the question is treated as non-medical, which keeps
    the pre-existing refusal behaviour as the safe default rather than letting
    an unclassifiable question through to generation.
    """
    try:
        verdict: DomainVerdict = await classifier.ainvoke({"question": question})
    except Exception:  # noqa: BLE001
        logger.warning(
            "medical-domain classification failed for %r; treating as off-topic",
            question[:80],
            exc_info=True,
        )
        return False
    return verdict.medical == "yes"


def off_topic_events() -> list[dict]:
    """The refusal token + meta pair emitted when the topic gate rejects."""
    return [
        {"type": "token", "text": OFF_TOPIC_REFUSAL},
        {
            "type": "meta",
            "citations": [],
            "contexts": [],
            "confidence": 0.0,
            "confidence_label": OUT_OF_SCOPE_LABEL,
            "disclaimer": MEDICAL_DISCLAIMER,
        },
    ]


def _build_chat_model(settings: Settings, model: str, temperature: float) -> ChatOpenAI:
    """One ChatOpenAI client with timeout, retries and the telemetry callback."""
    return ChatOpenAI(
        model=model,
        base_url=settings.nvidia_base_url,
        api_key=settings.nvidia_api_key,
        temperature=temperature,
        max_tokens=CHAT_MAX_TOKENS,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        callbacks=[TELEMETRY_CALLBACK],
        extra_body=None if settings.model_thinking_enabled else THINKING_DISABLED_BODY,
    )


def _build_chat_models(settings: Settings, temperature: float) -> list[ChatOpenAI]:
    """Primary model first, then the optional fallback model."""
    models = [_build_chat_model(settings, settings.chat_model, temperature)]
    if settings.chat_model_fallback:
        models.append(
            _build_chat_model(settings, settings.chat_model_fallback, temperature)
        )
    return models


def _combine_with_fallbacks(runnables: Sequence[Runnable]) -> Runnable:
    """Chain ``runnables`` so each later one is tried if the previous one fails."""
    primary, *fallbacks = runnables
    return primary.with_fallbacks(fallbacks) if fallbacks else primary


def build_chat_llm(
    settings: Settings,
    temperature: float = 0.1,
    identity: RunnableConfig | None = None,
) -> Runnable:
    """Chat LLM factory pointed at NVIDIA's OpenAI-compatible endpoint.

    Every model built here carries the telemetry callback, so any ``.invoke()``
    from any pipeline step is counted against the currently-active request
    (a call that falls over to ``CHAT_MODEL_FALLBACK`` counts once per model
    actually attempted). When no fallback is configured the plain
    ``ChatOpenAI`` is returned.

    ``identity`` (see :func:`run_identity`) is bound to every candidate model
    before the fallback chain is assembled, so the primary and the fallback
    both report the same step name to LangSmith.
    """
    named = [
        _apply_run_identity(model, identity)
        for model in _build_chat_models(settings, temperature)
    ]
    return _combine_with_fallbacks(named)


def build_structured_llm(
    settings: Settings,
    schema: type[BaseModel],
    temperature: float = 0.0,
    identity: RunnableConfig | None = None,
) -> Runnable:
    """Chat LLM that returns ``schema`` instances, with the same fallback policy.

    ``with_structured_output`` is applied to each model *before* combining
    them, rather than relying on ``RunnableWithFallbacks`` to proxy the call
    to its members — that proxying is an implementation detail of
    ``langchain_core`` and not part of its typed interface. The LangSmith
    ``identity`` is then bound to the model step of the resulting chain, which
    is where the traced LLM run is created.
    """
    structured = [
        _apply_run_identity(model.with_structured_output(schema), identity)
        for model in _build_chat_models(settings, temperature)
    ]
    return _combine_with_fallbacks(structured)


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


def context_texts(documents: Iterable[Document]) -> list[str]:
    """Full text of the passages an answer was actually generated from.

    Citations are deduplicated by source and truncated to a UI-sized snippet,
    so they describe where the evidence came from rather than what it said.
    Offline evaluation (``evaluate.py``) needs the evidence itself to judge
    whether an answer is grounded in it, which is what this returns.
    """
    return [doc.page_content.strip() for doc in documents]


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


def label_confidence(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    if score >= 0.55:
        return "MODERATE"
    if score >= 0.3:
        return "LOW"
    return "VERY LOW"
