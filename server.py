"""FastAPI backend for the Healthcare Knowledge Navigator.

Serves the HTML/JS frontend and exposes JSON endpoints that run either the
standard RAG pipeline or the Corrective RAG (CRAG) pipeline.

Run with:
    uvicorn server:app --reload
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
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
for name in ("hkn.rag", "hkn.crag", "hkn.server"):
    _lg = logging.getLogger(name)
    _lg.setLevel(logging.INFO)
    _lg.propagate = True

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Lazily-populated pipeline singletons.
_state: dict[str, object] = {}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_pipelines()
    except Exception as exc:  # noqa: BLE001 — surface reason at /api/health
        logger.exception("Failed to initialise pipelines: %s", exc)
        _state["startup_error"] = str(exc)
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
        result = pipeline.ask(payload.question)
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


@app.get("/api/traces")
async def traces() -> dict:
    """Aggregate LLM-call telemetry + per-request history."""
    return TELEMETRY.snapshot()


@app.post("/api/traces/reset")
async def traces_reset() -> dict:
    TELEMETRY.reset()
    return {"status": "ok", "message": "trace history cleared"}
