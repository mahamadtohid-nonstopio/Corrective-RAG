"""RAGAS access layer: import shim, evaluator LLM factory, NVIDIA metric exports.

Two things have to happen before any ``ragas`` symbol can be used in this
project, and both are isolated here so ``evaluate.py`` can simply import from
this module:

1. **The import shim.** ``ragas`` 0.4.3 still ships a deprecated LangChain LLM
   wrapper whose module-level imports reach for
   ``langchain_community.chat_models.vertexai``. That submodule was removed in
   the ``langchain-community`` 0.4 line this app is pinned to, so merely
   ``import ragas`` raises ``ModuleNotFoundError``. The symbol is used only for
   an ``isinstance`` check in a legacy code path the collections API never
   touches, so a stub class satisfies it. Downgrading ``langchain-community``
   instead would break the app's own floors.

2. **The evaluator client.** ``llm_factory`` builds its own OpenAI client, so
   the NVIDIA endpoint, the key and — for hybrid-reasoning models — the
   thinking-disabled template flag all have to be handed to it explicitly.
"""
from __future__ import annotations

import sys
import types

_VERTEXAI_MODULE_NAME = "langchain_community.chat_models.vertexai"
_VERTEXAI_CLASS_NAME = "ChatVertexAI"


def _install_legacy_vertexai_stub() -> None:
    """Register a stub for the ``langchain_community`` module ragas expects."""
    if _VERTEXAI_MODULE_NAME in sys.modules:
        return
    stub = types.ModuleType(_VERTEXAI_MODULE_NAME)
    setattr(stub, _VERTEXAI_CLASS_NAME, type(_VERTEXAI_CLASS_NAME, (), {}))
    sys.modules[_VERTEXAI_MODULE_NAME] = stub


_install_legacy_vertexai_stub()

from openai import AsyncOpenAI  # noqa: E402  (must follow the stub registration)
from ragas.llms import llm_factory  # noqa: E402
from ragas.llms.base import InstructorBaseRagasLLM  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
)

from src.config import Settings  # noqa: E402
from src.utils import THINKING_DISABLED_BODY  # noqa: E402

__all__ = [
    "AnswerAccuracy",
    "ContextRelevance",
    "ResponseGroundedness",
    "build_evaluator_llm",
]

# NVIDIA's hybrid-reasoning families emit a hidden reasoning pass by default
# that consumes the whole token budget and returns an empty message, which
# RAGAS then scores as NaN. Verified on this key: the same model with the
# thinking flag off scores normally. See src.utils.THINKING_DISABLED_BODY.
_HYBRID_REASONING_MODEL_MARKERS = ("nemotron-3.5",)

_OPENAI_PROVIDER = "openai"


def _needs_thinking_disabled(model: str) -> bool:
    return any(marker in model for marker in _HYBRID_REASONING_MODEL_MARKERS)


def _evaluator_model_options(model: str) -> dict:
    """Extra completion options required to get usable output from ``model``."""
    if not _needs_thinking_disabled(model):
        return {}
    return {"extra_body": THINKING_DISABLED_BODY}


def build_evaluator_llm(settings: Settings) -> InstructorBaseRagasLLM:
    """The LLM-as-judge used by every metric, pointed at the NVIDIA endpoint.

    Deliberately driven by ``Settings.eval_model`` rather than
    ``Settings.chat_model``: scoring an answer with the model that wrote it
    makes the numbers self-congratulatory, so the judge is configured
    independently of the app's generator.
    """
    client = AsyncOpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
    )
    return llm_factory(
        settings.eval_model,
        provider=_OPENAI_PROVIDER,
        client=client,
        **_evaluator_model_options(settings.eval_model),
    )
