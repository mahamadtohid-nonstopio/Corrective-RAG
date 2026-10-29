"""FastAPI backend for the Healthcare Knowledge Navigator.

Serves the HTML/JS frontend and exposes JSON endpoints that run either the
standard RAG pipeline or the Corrective RAG (CRAG) pipeline.

Run with:
    uvicorn server:app --reload
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import Settings, get_settings
from src.crag_agent import CRAGPipeline
from src.rag_agent import RAGPipeline
from src.telemetry import TELEMETRY
from src.utils import MEDICAL_DISCLAIMER

logger = logging.getLogger("hkn.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# Make sure the pipeline loggers propagate to the root handler (uvicorn --reload
# sometimes overrides root config; force propagation here).
for name in ("hkn.rag", "hkn.crag", "hkn.gate", "hkn.server", "hkn.live_eval"):
    _lg = logging.getLogger(name)
    _lg.setLevel(logging.INFO)
    _lg.propagate = True

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EVALUATION_DIR = BASE_DIR / "RAG-Evaluation"

# Lazily-populated pipeline singletons.
_state: dict[str, object] = {}

WARMUP_TIMEOUT_SECONDS = 45
WARMUP_PROMPT = "ping"
WARMUP_MAX_TOKENS = 8


def _load_pipelines() -> None:
    """Instantiate the RAG and CRAG pipelines once at startup.

    Kept out of module scope so an import doesn't require GOOGLE_API_KEY.
    """
    settings: Settings = get_settings()
    _state["settings"] = settings
    logger.info("Loading RAG pipeline …")
    _state["rag"] = RAGPipeline(settings)
    logger.info("Loading CRAG pipeline …")
    _state["crag"] = CRAGPipeline(settings)
    logger.info("Pipelines ready.")


def _load_live_evaluator(settings: Settings) -> None:
    """Enable per-question RAGAS scoring of frontend answers when possible.

    ragas lives in requirements-eval.txt, not the runtime image, so a missing
    install (or a judge that cannot be built) disables the feature with a
    warning instead of failing startup.
    """
    if not settings.live_evaluation_enabled:
        logger.info("Live evaluation disabled (LIVE_EVALUATION=false).")
        return
    sys.path.insert(0, str(EVALUATION_DIR))
    try:
        from live_evaluation import LiveEvaluator

        _state["live_evaluator"] = LiveEvaluator(
            settings, {"rag": _state["rag"], "crag": _state["crag"]}
        )
    except ImportError:
        logger.warning(
            "Live evaluation unavailable: install requirements-eval.txt to enable it.",
            exc_info=True,
        )
        return
    except Exception:  # noqa: BLE001 — optional feature, startup must not fail
        logger.warning("Live evaluation could not be initialised.", exc_info=True)
        return
    logger.info("Live evaluation enabled (judge: %s).", settings.eval_model)


def _live_answer_started(mode: str, question: str) -> None:
    evaluator = _state.get("live_evaluator")
    if evaluator is not None:
        evaluator.answer_started(mode, question)


def _live_answer_finished(
    mode: str, question: str, events: list[dict] | None, duration_seconds: float
) -> None:
    evaluator = _state.get("live_evaluator")
    if evaluator is not None:
        evaluator.answer_finished(mode, question, events, duration_seconds)


def _warm_up_embeddings(rag: RAGPipeline) -> None:
    rag.vector_store.embeddings.embed_query(WARMUP_PROMPT)


def _warm_up_chat(rag: RAGPipeline) -> None:
    rag.llm.bind(max_tokens=WARMUP_MAX_TOKENS).invoke(WARMUP_PROMPT)


WARMUP_STEPS: tuple[tuple[str, Callable[[RAGPipeline], None]], ...] = (
    ("embedding", _warm_up_embeddings),
    ("chat", _warm_up_chat),
)


def _warm_up_models(rag: RAGPipeline) -> None:
    """Prime the embedding and chat endpoints with one tiny call each.

    Steps are independent: one failing is logged and the next still runs.
    Runs outside any request, so no telemetry trace is active and the LLM
    call's telemetry callback is a no-op — it never shows in /api/traces.
    """
    for name, warm_up in WARMUP_STEPS:
        try:
            warm_up(rag)
            logger.info("Warm-up: %s endpoint ready.", name)
        except Exception:  # noqa: BLE001 — warm-up is optional
            logger.warning("Warm-up: %s endpoint failed.", name, exc_info=True)


async def _warm_up_best_effort() -> None:
    """Warm up off the event loop, bounded by a timeout; never raises.

    On timeout the worker thread is abandoned (threads can't be cancelled)
    and finishes in the background; startup proceeds regardless.
    """
    settings: Settings = _state["settings"]  # type: ignore[assignment]
    if not settings.warmup_on_startup:
        logger.info("Warm-up skipped (WARMUP_ON_STARTUP=false).")
        return
    rag: RAGPipeline = _state["rag"]  # type: ignore[assignment]
    try:
        await asyncio.wait_for(
            asyncio.to_thread(_warm_up_models, rag), timeout=WARMUP_TIMEOUT_SECONDS
        )
        logger.info("Warm-up finished.")
    except asyncio.TimeoutError:
        logger.warning("Warm-up timed out after %ss; continuing startup.", WARMUP_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001 — warm-up is optional, startup must not fail
        logger.warning("Warm-up aborted; continuing startup.", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_pipelines()
    except Exception as exc:  # noqa: BLE001 — surface reason at /api/health
        logger.exception("Failed to initialise pipelines: %s", exc)
        _state["startup_error"] = str(exc)
    else:
        _load_live_evaluator(_state["settings"])  # type: ignore[arg-type]
        await _warm_up_best_effort()
    yield


app = FastAPI(
    title="Healthcare Knowledge Navigator",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- schemas ----------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    mode: Literal["rag", "crag"] = "crag"


class Citation(BaseModel):
    id: int
    source: str
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    mode: Literal["rag", "crag"]
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    confidence_label: str
    trace: list[str] = []
    disclaimer: str = MEDICAL_DISCLAIMER


# ---------- helpers ----------
def _get_pipeline(mode: str):
    if _state.get("startup_error"):
        raise HTTPException(
            status_code=503,
            detail=f"Pipelines unavailable: {_state['startup_error']}",
        )
    pipeline = _state.get(mode)
    if pipeline is None:
        raise HTTPException(status_code=503, detail=f"{mode.upper()} pipeline not ready.")
    return pipeline


# ---------- routes ----------
@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/dashboard", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(STATIC_DIR / "dashboard.html")


@app.get("/api/health")
async def health() -> dict:
    ok = "startup_error" not in _state
    settings: Settings | None = _state.get("settings")  # type: ignore[assignment]
    ls_tracing = (os.getenv("LANGSMITH_TRACING", "").lower() == "true")
    ls_key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    ls_active = ls_tracing and bool(ls_key) and not ls_key.startswith("lsv2_pt_your_")
    return {
        "status": "ok" if ok else "error",
        "error": _state.get("startup_error"),
        "provider": "nvidia",
        "chat_model": settings.chat_model if settings else None,
        "embedding_model": settings.embedding_model if settings else None,
        "vector_store": "pinecone",
        "pinecone_index": settings.pinecone_index if settings else None,
        "pinecone_namespace": settings.pinecone_namespace if settings else None,
        "top_k": settings.top_k if settings else None,
        "web_fallback_enabled": bool(settings and settings.tavily_api_key),
        "langsmith_enabled": ls_active,
        "langsmith_project": os.getenv("LANGSMITH_PROJECT") if ls_active else None,
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    pipeline = _get_pipeline(payload.mode)
    TELEMETRY.start_request(payload.mode, payload.question)
    try:
        result = await pipeline.ask(payload.question)
    except Exception as exc:  # noqa: BLE001
        TELEMETRY.end_request(ok=False, error=str(exc))
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    TELEMETRY.end_request(ok=True)

    return AskResponse(
        mode=payload.mode,
        question=payload.question,
        answer=result.get("answer", ""),
        citations=[Citation(**c) for c in (result.get("citations") or [])],
        confidence=float(result.get("confidence", 0.0)),
        confidence_label=result.get("confidence_label", "VERY LOW"),
        trace=list(result.get("trace") or []),
    )


@app.post("/api/ask/stream")
async def ask_stream(payload: AskRequest):
    """Stream pipeline events as NDJSON so the frontend can render tokens live.

    Response body is a series of newline-delimited JSON events, one per line:
        {"type": "trace", "message": "..."}
        {"type": "token", "text": "..."}
        {"type": "meta",  "citations": [...], "confidence": ..., ...}
        {"type": "error", "message": "..."}
        {"type": "done"}
    """
    pipeline = _get_pipeline(payload.mode)
    TELEMETRY.start_request(payload.mode, payload.question)

    async def emit():
        errored = False
        answered = False
        events: list[dict] = []
        started = time.perf_counter()
        _live_answer_started(payload.mode, payload.question)
        try:
            async for event in pipeline.stream(payload.question):
                events.append(event)
                yield json.dumps(event) + "\n"
                if event.get("type") == "error":
                    errored = True
                elif event.get("type") == "meta":
                    answered = True
        except Exception as exc:  # noqa: BLE001
            errored = True
            logger.exception("Streaming pipeline error")
            yield json.dumps({"type": "error", "message": str(exc)}) + "\n"
        finally:
            TELEMETRY.end_request(ok=not errored)
            _live_answer_finished(
                payload.mode,
                payload.question,
                events if answered and not errored else None,
                time.perf_counter() - started,
            )
            yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(
        emit(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/traces")
async def traces() -> dict:
    """Aggregate LLM-call telemetry + per-request history."""
    return TELEMETRY.snapshot()


@app.post("/api/traces/reset")
async def traces_reset() -> dict:
    TELEMETRY.reset()
    return {"status": "ok", "message": "trace history cleared"}
