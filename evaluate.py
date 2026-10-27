"""Offline RAG-vs-CRAG evaluation harness built on the RAGAS NVIDIA metrics.

Runs every question in ``eval/golden_set.json`` through both pipelines
in-process (no server needed), scores each answer with three LLM-as-judge
metrics, prints a per-category and a per-question table, and writes a
timestamped JSON result file under ``eval/`` so runs stay comparable.

Metrics (all from ``ragas.metrics.collections``):
    ContextRelevance      - did retrieval return passages that bear on the
                            question? Needs no ground truth.
    ResponseGroundedness  - is the answer supported by the passages it was
                            actually generated from? Needs no ground truth.
    AnswerAccuracy        - does the answer agree with the reference answer?
                            Skipped for golden-set rows whose reference is
                            null, which is the normal case for questions the
                            corpus cannot cover.

The judge is ``Settings.eval_model``, configured independently of the app's
``CHAT_MODEL`` so the generator never grades its own homework.

Usage:
    python evaluate.py                      # full set, both pipelines
    python evaluate.py --limit 4            # first 4 questions
    python evaluate.py --mode crag          # one pipeline only
    python evaluate.py --category in-corpus # one category only
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ragas_compat import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
    build_evaluator_llm,
)
from src.config import Settings, get_settings
from src.crag_agent import CRAGPipeline
from src.rag_agent import RAGPipeline

logger = logging.getLogger("hkn.eval")

ROOT_DIR = Path(__file__).resolve().parent
EVAL_DIR = ROOT_DIR / "eval"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"
RESULTS_FILENAME_TEMPLATE = "results-{timestamp}.json"
RESULTS_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

MODE_RAG = "rag"
MODE_CRAG = "crag"
MODE_BOTH = "both"
ALL_MODES = (MODE_RAG, MODE_CRAG)

METRIC_CONTEXT_RELEVANCE = "context_relevance"
METRIC_RESPONSE_GROUNDEDNESS = "response_groundedness"
METRIC_ANSWER_ACCURACY = "answer_accuracy"
METRIC_NAMES = (
    METRIC_CONTEXT_RELEVANCE,
    METRIC_RESPONSE_GROUNDEDNESS,
    METRIC_ANSWER_ACCURACY,
)

# One judge call can stall on a slow NVIDIA endpoint; cap it so a single row
# cannot hold up the whole run.
METRIC_TIMEOUT_SECONDS = 300.0
PIPELINE_TIMEOUT_SECONDS = 240.0

STATUS_OK = "ok"
STATUS_FAILED = "failed"
SKIPPED_NO_REFERENCE = "skipped: no reference"
SKIPPED_NO_CONTEXT = "skipped: no retrieved context"
NOT_APPLICABLE = "n/a"
METRIC_ERROR_MARKER = "ERR"

# Routing paths, recovered from the pipelines' own trace events.
PATH_REFUSED = "refused"
PATH_NO_CONTEXT = "no-context"
PATH_WEAK_LOCAL = "weak-local"
PATH_DIRECT = "direct"
PATH_UNKNOWN = "unknown"

_CRAG_DECISION_PATTERN = re.compile(r"decision: (\w+)")
_REFUSAL_TRACE_MARKER = "not a medical question"
_NO_CONTEXT_TRACE_MARKER = "retrieved 0 passages"
_WEAK_RETRIEVAL_TRACE_MARKER = "topic gate:"

_CITATION_SNIPPET_KEY = "snippet"
_META_CONTEXTS_KEY = "contexts"

_SUMMARY_COLUMN_WIDTH = 24
_CATEGORY_COLUMN_WIDTH = 16
_SCORE_FORMAT = "{:.3f}"
_OVERALL_ROW_LABEL = "ALL"


# ---------------------------------------------------------------- golden set


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    category: str
    question: str
    reference: str | None


def load_golden_set(path: Path) -> list[GoldenQuestion]:
    """Read the golden set, failing loudly on a malformed or missing file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read golden set at {path}: {exc}") from exc
    try:
        return [
            GoldenQuestion(
                id=row["id"],
                category=row["category"],
                question=row["question"],
                reference=row.get("reference"),
            )
            for row in payload["questions"]
        ]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"golden set at {path} is missing required fields: {exc}") from exc


def select_questions(
    questions: Sequence[GoldenQuestion], category: str | None, limit: int | None
) -> list[GoldenQuestion]:
    """Apply the --category and --limit filters, in that order."""
    selected = [q for q in questions if category is None or q.category == category]
    if not selected and category is not None:
        raise RuntimeError(f"no golden-set questions in category {category!r}")
    return selected[:limit] if limit else selected


# ------------------------------------------------------------- pipeline runs


@dataclass
class PipelineRun:
    """What one pipeline produced for one question."""

    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    confidence_label: str = ""
    path: str = PATH_UNKNOWN
    trace: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None


def _contexts_from_meta(meta: dict) -> list[str]:
    """The passages the answer was generated from.

    Both pipelines publish the merged, post-grading context set on their meta
    event (for CRAG that is kept-local + web, never the discarded pre-grading
    retrieval). Citation snippets are the fallback for an older pipeline that
    does not emit the full passages — they are deduplicated and truncated, so
    they understate groundedness rather than overstating it.
    """
    contexts = meta.get(_META_CONTEXTS_KEY)
    if contexts is not None:
        return list(contexts)
    return [
        citation[_CITATION_SNIPPET_KEY]
        for citation in meta.get("citations", [])
        if citation.get(_CITATION_SNIPPET_KEY)
    ]


def _routing_path(mode: str, trace: Sequence[str]) -> str:
    """Recover which branch the pipeline took from its own trace messages."""
    joined = " | ".join(trace)
    if _REFUSAL_TRACE_MARKER in joined:
        return PATH_REFUSED
    if _NO_CONTEXT_TRACE_MARKER in joined:
        return PATH_NO_CONTEXT
    if mode == MODE_CRAG:
        decision = _CRAG_DECISION_PATTERN.search(joined)
        return decision.group(1) if decision else PATH_UNKNOWN
    if _WEAK_RETRIEVAL_TRACE_MARKER in joined:
        return PATH_WEAK_LOCAL
    return PATH_DIRECT


def _apply_event(run: PipelineRun, event: dict) -> None:
    """Fold one streamed pipeline event into the accumulating run."""
    kind = event.get("type")
    if kind == "trace":
        run.trace.append(event["message"])
    elif kind == "token":
        run.answer += event["text"]
    elif kind == "meta":
        run.contexts = _contexts_from_meta(event)
        run.citations = event.get("citations", [])
        run.confidence = event.get("confidence", 0.0)
        run.confidence_label = event.get("confidence_label", "")
    elif kind == "error":
        raise RuntimeError(event["message"])


async def run_pipeline(pipeline, mode: str, question: str) -> PipelineRun:
    """Stream one question through one pipeline, capturing everything it emits.

    A pipeline failure is recorded on the run rather than raised: one bad
    question must not abort the sweep.
    """
    run = PipelineRun()
    started = time.perf_counter()
    try:
        await asyncio.wait_for(
            _drain_stream(pipeline, question, run), timeout=PIPELINE_TIMEOUT_SECONDS
        )
    except Exception as exc:  # noqa: BLE001 - recorded, reported and surfaced below
        run.error = f"{type(exc).__name__}: {exc}"
        logger.warning("%s pipeline failed for %r", mode, question[:70], exc_info=True)
    run.duration_seconds = time.perf_counter() - started
    run.path = _routing_path(mode, run.trace)
    return run


async def _drain_stream(pipeline, question: str, run: PipelineRun) -> None:
    async for event in pipeline.stream(question):
        _apply_event(run, event)


# ------------------------------------------------------------------ scoring


@dataclass
class MetricOutcome:
    """One metric's verdict for one answer: a score, a skip, or a failure."""

    value: float | None = None
    status: str = STATUS_OK
    error: str | None = None

    @property
    def scored(self) -> bool:
        return self.value is not None


def _skipped(reason: str) -> MetricOutcome:
    return MetricOutcome(status=reason)


async def _score_one(name: str, score: Callable[[], Awaitable]) -> MetricOutcome:
    """Await one metric call, converting any failure into a recorded outcome."""
    try:
        result = await asyncio.wait_for(score(), timeout=METRIC_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - recorded, reported and surfaced below
        logger.warning("metric %s failed", name, exc_info=True)
        return MetricOutcome(status=STATUS_FAILED, error=f"{type(exc).__name__}: {exc}")
    value = float(result.value)
    if value != value:  # NaN: the judge returned nothing parseable.
        return MetricOutcome(status=STATUS_FAILED, error="judge returned an unparseable score")
    return MetricOutcome(value=value)


class MetricSuite:
    """The three NVIDIA metrics, sharing one judge model."""

    def __init__(self, settings: Settings) -> None:
        llm = build_evaluator_llm(settings)
        self.model = settings.eval_model
        self._context_relevance = ContextRelevance(llm=llm)
        self._response_groundedness = ResponseGroundedness(llm=llm)
        self._answer_accuracy = AnswerAccuracy(llm=llm)

    async def score(
        self, question: GoldenQuestion, run: PipelineRun
    ) -> dict[str, MetricOutcome]:
        """Score one answer, running every applicable metric concurrently."""
        if run.error:
            return {name: _skipped(STATUS_FAILED) for name in METRIC_NAMES}
        skips, applicable = self._partition_metrics(question, run)
        scored = await asyncio.gather(
            *(_score_one(name, call) for name, call in applicable.items())
        )
        return {**skips, **dict(zip(applicable, scored))}

    def _partition_metrics(
        self, question: GoldenQuestion, run: PipelineRun
    ) -> tuple[dict[str, MetricOutcome], dict[str, Callable[[], Awaitable]]]:
        """Split the suite into metrics this row skips and metrics it can score.

        A metric is skipped — explicitly, never by crashing — when its inputs
        do not exist: no retrieved context for the two reference-free metrics,
        or no asserted reference answer for AnswerAccuracy.
        """
        skips: dict[str, MetricOutcome] = {}
        applicable: dict[str, Callable[[], Awaitable]] = {}
        if run.contexts:
            applicable[METRIC_CONTEXT_RELEVANCE] = lambda: self._context_relevance.ascore(
                user_input=question.question, retrieved_contexts=run.contexts
            )
            applicable[METRIC_RESPONSE_GROUNDEDNESS] = lambda: self._response_groundedness.ascore(
                response=run.answer, retrieved_contexts=run.contexts
            )
        else:
            skips[METRIC_CONTEXT_RELEVANCE] = _skipped(SKIPPED_NO_CONTEXT)
            skips[METRIC_RESPONSE_GROUNDEDNESS] = _skipped(SKIPPED_NO_CONTEXT)
        if question.reference is None:
            skips[METRIC_ANSWER_ACCURACY] = _skipped(SKIPPED_NO_REFERENCE)
        else:
            applicable[METRIC_ANSWER_ACCURACY] = lambda: self._answer_accuracy.ascore(
                user_input=question.question,
                response=run.answer,
                reference=question.reference,
            )
        return skips, applicable


# ------------------------------------------------------------------ results


@dataclass
class EvaluatedRow:
    """One golden-set question scored for one pipeline."""

    id: str
    category: str
    question: str
    mode: str
    run: PipelineRun
    metrics: dict[str, MetricOutcome]

    @property
    def failed(self) -> bool:
        return self.run.error is not None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "mode": self.mode,
            "answer": self.run.answer,
            "contexts": self.run.contexts,
            "citations": self.run.citations,
            "confidence": self.run.confidence,
            "confidence_label": self.run.confidence_label,
            "path": self.run.path,
            "trace": self.run.trace,
            "duration_seconds": round(self.run.duration_seconds, 2),
            "error": self.run.error,
            "metrics": {name: asdict(outcome) for name, outcome in self.metrics.items()},
        }


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    return sum(collected) / len(collected) if collected else None


def _metric_mean(rows: Sequence[EvaluatedRow], metric: str) -> float | None:
    return _mean(
        row.metrics[metric].value
        for row in rows
        if metric in row.metrics and row.metrics[metric].scored
    )


def aggregate(rows: Sequence[EvaluatedRow]) -> dict[str, dict[str, dict]]:
    """Mean per metric, grouped by category then mode, plus an overall bucket."""
    categories = sorted({row.category for row in rows})
    summary: dict[str, dict[str, dict]] = {}
    for category in [*categories, _OVERALL_ROW_LABEL]:
        bucket = rows if category == _OVERALL_ROW_LABEL else [
            row for row in rows if row.category == category
        ]
        summary[category] = {
            mode: _mode_summary([row for row in bucket if row.mode == mode])
            for mode in ALL_MODES
            if any(row.mode == mode for row in bucket)
        }
    return summary


def _mode_summary(rows: Sequence[EvaluatedRow]) -> dict:
    return {
        "n": len(rows),
        "failures": sum(1 for row in rows if row.failed),
        "mean_confidence": _mean(row.run.confidence for row in rows),
        **{metric: _metric_mean(rows, metric) for metric in METRIC_NAMES},
    }


# ------------------------------------------------------------------ printing


def _format_score(value: float | None) -> str:
    return _SCORE_FORMAT.format(value) if value is not None else NOT_APPLICABLE


def _format_outcome(outcome: MetricOutcome) -> str:
    """A score, ``n/a`` where the metric did not apply, or ``ERR`` where it broke."""
    if outcome.scored:
        return _SCORE_FORMAT.format(outcome.value)
    return METRIC_ERROR_MARKER if outcome.status == STATUS_FAILED else NOT_APPLICABLE


def print_summary_table(summary: dict[str, dict[str, dict]]) -> None:
    header = (
        f"{'category':<{_CATEGORY_COLUMN_WIDTH}}{'mode':<6}{'n':>3}  "
        + "".join(f"{name:>{_SUMMARY_COLUMN_WIDTH}}" for name in METRIC_NAMES)
        + f"{'mean_confidence':>18}"
    )
    print("\nPER-CATEGORY SUMMARY (mean score; n/a = metric never applied)")
    print(header)
    print("-" * len(header))
    for category, modes in summary.items():
        for mode, stats in modes.items():
            print(
                f"{category:<{_CATEGORY_COLUMN_WIDTH}}{mode:<6}{stats['n']:>3}  "
                + "".join(
                    f"{_format_score(stats[name]):>{_SUMMARY_COLUMN_WIDTH}}"
                    for name in METRIC_NAMES
                )
                + f"{_format_score(stats['mean_confidence']):>18}"
            )
        print()


def print_detail_table(rows: Sequence[EvaluatedRow]) -> None:
    header = (
        f"{'id':<16}{'mode':<6}{'path':<12}{'conf':>6}  "
        f"{'ctx_rel':>8}{'grounded':>10}{'accuracy':>10}  {'secs':>6}  status"
    )
    print("\nPER-QUESTION DETAIL")
    print(header)
    print("-" * len(header))
    for row in rows:
        metrics = "".join(
            f"{_format_outcome(row.metrics[name]):>{width}}"
            for name, width in zip(METRIC_NAMES, (8, 10, 10))
        )
        status = row.run.error or STATUS_OK
        print(
            f"{row.id:<16}{row.mode:<6}{row.run.path:<12}"
            f"{row.run.confidence:>6.2f}  {metrics}  "
            f"{row.run.duration_seconds:>6.1f}  {status[:60]}"
        )


# --------------------------------------------------------------------- main


def _modes_for(mode_argument: str) -> tuple[str, ...]:
    return ALL_MODES if mode_argument == MODE_BOTH else (mode_argument,)


def _build_pipelines(modes: Sequence[str], settings: Settings) -> dict[str, object]:
    builders = {MODE_RAG: RAGPipeline, MODE_CRAG: CRAGPipeline}
    return {mode: builders[mode](settings) for mode in modes}


async def evaluate_all(
    questions: Sequence[GoldenQuestion],
    pipelines: dict[str, object],
    metrics: MetricSuite,
) -> list[EvaluatedRow]:
    """Run and score every question against every selected pipeline."""
    rows: list[EvaluatedRow] = []
    total = len(questions) * len(pipelines)
    for question in questions:
        for mode, pipeline in pipelines.items():
            print(
                f"[{len(rows) + 1}/{total}] {mode:<4} {question.id} ...",
                end="",
                flush=True,
            )
            run = await run_pipeline(pipeline, mode, question.question)
            scored = await metrics.score(question, run)
            rows.append(
                EvaluatedRow(
                    id=question.id,
                    category=question.category,
                    question=question.question,
                    mode=mode,
                    run=run,
                    metrics=scored,
                )
            )
            print(f" {run.path} ({run.duration_seconds:.1f}s)", flush=True)
    return rows


def write_results(
    rows: Sequence[EvaluatedRow],
    summary: dict,
    settings: Settings,
    arguments: argparse.Namespace,
    elapsed_seconds: float,
) -> Path:
    """Persist the full run so later runs can be diffed against this one."""
    timestamp = datetime.now(timezone.utc)
    path = EVAL_DIR / RESULTS_FILENAME_TEMPLATE.format(
        timestamp=timestamp.strftime(RESULTS_TIMESTAMP_FORMAT)
    )
    payload = {
        "run": {
            "timestamp_utc": timestamp.isoformat(),
            "elapsed_seconds": round(elapsed_seconds, 1),
            "generator_model": settings.chat_model,
            "evaluator_model": settings.eval_model,
            "embedding_model": settings.embedding_model,
            "top_k": settings.top_k,
            "topic_score_threshold": settings.topic_score_threshold,
            "arguments": vars(arguments),
            "question_count": len({row.id for row in rows}),
            "row_count": len(rows),
            "failed_rows": sum(1 for row in rows if row.failed),
        },
        "summary": summary,
        "rows": [row.to_dict() for row in rows],
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N questions"
    )
    parser.add_argument(
        "--mode",
        choices=(MODE_RAG, MODE_CRAG, MODE_BOTH),
        default=MODE_BOTH,
        help="which pipeline(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--category", default=None, help="evaluate only one golden-set category"
    )
    return parser.parse_args(argv)


async def main_async(arguments: argparse.Namespace) -> int:
    settings = get_settings()
    questions = select_questions(
        load_golden_set(GOLDEN_SET_PATH), arguments.category, arguments.limit
    )
    modes = _modes_for(arguments.mode)
    print(
        f"Evaluating {len(questions)} question(s) x {len(modes)} pipeline(s)\n"
        f"  generator : {settings.chat_model}\n"
        f"  evaluator : {settings.eval_model}"
    )

    started = time.perf_counter()
    rows = await evaluate_all(
        questions, _build_pipelines(modes, settings), MetricSuite(settings)
    )
    elapsed = time.perf_counter() - started

    summary = aggregate(rows)
    print_summary_table(summary)
    print_detail_table(rows)

    failures = sum(1 for row in rows if row.failed)
    results_path = write_results(rows, summary, settings, arguments, elapsed)
    print(
        f"\n{len(rows)} row(s) in {elapsed:.0f}s, {failures} pipeline failure(s)."
        f"\nResults written to {results_path}"
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return asyncio.run(main_async(parse_arguments()))
    except RuntimeError as exc:
        logger.error("evaluation aborted: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
