"""LLM-call telemetry.

Uses a LangChain ``BaseCallbackHandler`` attached to every chat LLM built by
:func:`src.utils.build_chat_llm`, so every ``.invoke()`` (from any pipeline
node) increments the current request's counter automatically.

The counter is stored in a :class:`contextvars.ContextVar` so concurrent
requests never mix, even under uvicorn's async event loop.
"""
from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

_current_trace: ContextVar["RequestTrace | None"] = ContextVar(
    "hkn_current_trace", default=None
)


@dataclass
class RequestTrace:
    mode: str  # "rag" or "crag"
    question: str
    started_at: str  # ISO 8601 UTC
    finished_at: str | None = None
    duration_ms: int = 0
    llm_calls: int = 0
    nodes_visited: list[str] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    # Not serialised — used internally to compute duration.
    _start_time: float = field(default_factory=time.time, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_start_time", None)
        return d


class Telemetry:
    """In-memory ring buffer of recent request traces."""

    def __init__(self, max_history: int = 200) -> None:
        self._lock = threading.Lock()
        self._history: list[RequestTrace] = []
        self._max_history = max_history

    def start_request(self, mode: str, question: str) -> RequestTrace:
        trace = RequestTrace(
            mode=mode,
            question=question,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        _current_trace.set(trace)
        return trace

    def end_request(self, ok: bool = True, error: str | None = None) -> RequestTrace | None:
        trace = _current_trace.get()
        if trace is None:
            return None
        trace.finished_at = datetime.now(timezone.utc).isoformat()
        trace.duration_ms = int((time.time() - trace._start_time) * 1000)
        trace.ok = ok
        trace.error = error
        with self._lock:
            self._history.append(trace)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
        _current_trace.set(None)
        return trace

    def increment(self) -> None:
        trace = _current_trace.get()
        if trace is not None:
            trace.llm_calls += 1

    def record_node(self, node_name: str) -> None:
        trace = _current_trace.get()
        if trace is not None:
            trace.nodes_visited.append(node_name)

    def current(self) -> RequestTrace | None:
        return _current_trace.get()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            history = list(self._history)
        rag = [t for t in history if t.mode == "rag"]
        crag = [t for t in history if t.mode == "crag"]
        return {
            "totals": {
                "requests": len(history),
                "rag_requests": len(rag),
                "crag_requests": len(crag),
                "llm_calls": sum(t.llm_calls for t in history),
                "rag_llm_calls": sum(t.llm_calls for t in rag),
                "crag_llm_calls": sum(t.llm_calls for t in crag),
                "avg_rag_llm_calls_per_request": (
                    round(sum(t.llm_calls for t in rag) / len(rag), 2) if rag else 0
                ),
                "avg_crag_llm_calls_per_request": (
                    round(sum(t.llm_calls for t in crag) / len(crag), 2) if crag else 0
                ),
            },
            "history": [t.to_dict() for t in reversed(history)],
        }

    def reset(self) -> None:
        with self._lock:
            self._history.clear()


class TelemetryCallback(BaseCallbackHandler):
    """Increments the current request's LLM-call counter."""

    def __init__(self, telemetry: "Telemetry") -> None:
        self._telemetry = telemetry

    # Regular completion LLMs
    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        self._telemetry.increment()

    # Chat models — ChatOpenAI fires this one
    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        self._telemetry.increment()


TELEMETRY = Telemetry()
TELEMETRY_CALLBACK = TelemetryCallback(TELEMETRY)
