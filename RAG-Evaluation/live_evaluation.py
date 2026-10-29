"""Per-question RAG-vs-CRAG RAGAS evaluation of questions asked from the frontend.

``server.py`` reports every ``/api/ask/stream`` request to
:class:`LiveEvaluator` when it starts and when it finishes. Once no pipeline is
still answering a question, the evaluator scores the RAG and the CRAG answer
side by side in the background — the user's answer is never delayed — and
writes one Markdown file per question to ``RAG-Evaluation/evaluations/``,
named ``<date>_<time>_<question>.md`` so the folder sorts chronologically.

In Compare mode the frontend asks both pipelines at once, so both answers come
from the frontend. In RAG-only or CRAG-only mode the evaluator runs the other
pipeline itself, so every file is a comparison.

A live question has no reference answer, so only the reference-free RAGAS
metrics apply: context_relevance, faithfulness, response_groundedness and
answer_relevancy. Accuracy against ground truth needs the golden-set run
(``evaluation.py``).
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from evaluate import ALL_MODES, PipelineRun, _apply_event, _routing_path, run_pipeline
from evaluation import (
    METRIC_ERROR_MARKER,
    SCORE_FORMAT,
    STATUS_FAILED,
    GoldenQuestion,
    MetricOutcome,
    MetricSpec,
    Scorer,
    build_metric_specs,
    score_band,
)
from src.config import Settings

logger = logging.getLogger("hkn.live_eval")

EVALUATIONS_DIR = Path(__file__).resolve().parent / "evaluations"
FILENAME_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
DISPLAY_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_SLUG_LENGTH = 60
_NON_SLUG_CHARACTERS = re.compile(r"[^a-z0-9]+")

LIVE_QUESTION_ID = "live"
LIVE_CATEGORY = "live"
SNIPPET_PREVIEW_LENGTH = 240

SOURCE_FRONTEND = "asked from the frontend"
SOURCE_EVALUATOR = "run by the evaluator for comparison"
TIE_MARGIN = 0.01
TIE_LABEL = "tie"
NOT_SCORED = "not scored"


def question_slug(question: str) -> str:
    slug = _NON_SLUG_CHARACTERS.sub("-", question.lower()).strip("-")
    return slug[:MAX_SLUG_LENGTH].rstrip("-") or "question"


def _question_key(question: str) -> str:
    return " ".join(question.lower().split())


def run_from_events(mode: str, events: Sequence[dict], duration_seconds: float) -> PipelineRun:
    """Rebuild what the pipeline produced from the events the server streamed."""
    run = PipelineRun(duration_seconds=duration_seconds)
    for event in events:
        _apply_event(run, event)
    run.path = _routing_path(mode, run.trace)
    return run


# ------------------------------------------------------------------ sessions


@dataclass
class QuestionSession:
    """Every answer to one question, gathered until no pipeline is still answering."""

    question: str
    in_flight: set[str] = field(default_factory=set)
    runs: dict[str, PipelineRun] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)


@dataclass
class ScoredAnswer:
    mode: str
    source: str
    run: PipelineRun
    metrics: dict[str, MetricOutcome]

    @property
    def overall(self) -> float | None:
        values = [outcome.value for outcome in self.metrics.values() if outcome.scored]
        return sum(values) / len(values) if values else None


class LiveEvaluator:
    """Pairs the RAG and CRAG answers to a frontend question and scores both."""

    def __init__(self, settings: Settings, pipelines: Mapping[str, object]) -> None:
        self._settings = settings
        self._pipelines = dict(pipelines)
        self._scorer = Scorer(
            [spec for spec in build_metric_specs(settings) if not spec.needs_reference]
        )
        self._sessions: dict[str, QuestionSession] = {}
        # The event loop holds only weak references to tasks; keep strong ones
        # so an evaluation cannot be garbage-collected mid-run.
        self._tasks: set[asyncio.Task] = set()

    def answer_started(self, mode: str, question: str) -> None:
        session = self._sessions.setdefault(_question_key(question), QuestionSession(question))
        session.in_flight.add(mode)

    def answer_finished(
        self, mode: str, question: str, events: Sequence[dict] | None, duration_seconds: float
    ) -> None:
        """Record one finished answer; ``events`` is ``None`` when it failed or was aborted.

        When it was the last pipeline still answering, the evaluation is
        handed to a background task and the session is closed.
        """
        key = _question_key(question)
        session = self._sessions.get(key)
        if session is None:
            return
        session.in_flight.discard(mode)
        if events is not None:
            session.runs[mode] = run_from_events(mode, events, duration_seconds)
            session.sources[mode] = SOURCE_FRONTEND
        if session.in_flight:
            return
        del self._sessions[key]
        if not session.runs:
            return
        task = asyncio.create_task(self._evaluate_and_save(session))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _evaluate_and_save(self, session: QuestionSession) -> Path | None:
        """Fill in the missing pipeline, score both answers, write the report.

        Runs as a detached background task, so failures are logged with
        context rather than raised into an unobserved task.
        """
        answered_at = datetime.now()
        try:
            await self._run_missing_pipelines(session)
            golden = GoldenQuestion(
                id=LIVE_QUESTION_ID, category=LIVE_CATEGORY, question=session.question, reference=None
            )
            scored = await asyncio.gather(
                *(self._score(golden, mode, session) for mode in ALL_MODES)
            )
            return self._write_report(session.question, list(scored), answered_at)
        except Exception:  # noqa: BLE001 - background task: log, never crash the server
            logger.exception("live evaluation failed for %r", session.question[:80])
            return None

    async def _run_missing_pipelines(self, session: QuestionSession) -> None:
        missing = [mode for mode in ALL_MODES if mode not in session.runs]
        runs = await asyncio.gather(
            *(run_pipeline(self._pipelines[mode], mode, session.question) for mode in missing)
        )
        for mode, run in zip(missing, runs):
            session.runs[mode] = run
            session.sources[mode] = SOURCE_EVALUATOR

    async def _score(self, golden: GoldenQuestion, mode: str, session: QuestionSession) -> ScoredAnswer:
        run = session.runs[mode]
        metrics = await self._scorer.score(golden, run)
        return ScoredAnswer(mode=mode, source=session.sources[mode], run=run, metrics=metrics)

    def _write_report(self, question: str, answers: list[ScoredAnswer], answered_at: datetime) -> Path:
        EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{answered_at.strftime(FILENAME_TIMESTAMP_FORMAT)}_{question_slug(question)}.md"
        path = EVALUATIONS_DIR / filename
        path.write_text(
            render_live_report(question, answers, self._scorer.specs, self._settings, answered_at),
            encoding="utf-8",
        )
        logger.info("live evaluation saved → %s", path)
        return path


# ------------------------------------------------------------------- report


def render_live_report(
    question: str,
    answers: Sequence[ScoredAnswer],
    specs: Sequence[MetricSpec],
    settings: Settings,
    answered_at: datetime,
) -> str:
    return "\n".join(
        [
            *_header_section(question, answers, settings, answered_at),
            *_comparison_section(answers, specs),
            *_skip_notes_section(answers, specs),
            *(line for answer in answers for line in _answer_section(answer)),
        ]
    )


def _format_score(value: float | None) -> str:
    return SCORE_FORMAT.format(value) if value is not None else NOT_SCORED


def _winner(answers: Sequence[ScoredAnswer], values: Sequence[float | None]) -> str:
    """The pipeline with the higher score, ``tie`` when within the margin."""
    scored = [(value, answer.mode) for answer, value in zip(answers, values) if value is not None]
    if len(scored) < len(answers):
        return ""
    scored.sort(reverse=True)
    if scored[0][0] - scored[-1][0] < TIE_MARGIN:
        return TIE_LABEL
    return f"**{scored[0][1].upper()}**"


def _header_section(
    question: str, answers: Sequence[ScoredAnswer], settings: Settings, answered_at: datetime
) -> list[str]:
    return [
        "# RAG vs CRAG answer evaluation",
        "",
        f"> **Question:** {question}",
        "",
        f"- **Answered:** {answered_at.strftime(DISPLAY_TIMESTAMP_FORMAT)}",
        *(f"- **{answer.mode.upper()}:** {answer.source}" for answer in answers),
        f"- **Generator:** `{settings.chat_model}` · **Judge:** `{settings.eval_model}`",
        "",
    ]


def _cell(outcome: MetricOutcome) -> str:
    if outcome.scored:
        return SCORE_FORMAT.format(outcome.value)
    return METRIC_ERROR_MARKER if outcome.status == STATUS_FAILED else "n/a"


def _comparison_section(answers: Sequence[ScoredAnswer], specs: Sequence[MetricSpec]) -> list[str]:
    modes = [answer.mode.upper() for answer in answers]
    overalls = [answer.overall for answer in answers]
    lines = [
        "## Side-by-side scores",
        "",
        "| Metric | " + " | ".join(modes) + " | Better | What it measures |",
        "|---|" + "---:|" * len(modes) + "---|---|",
        "| **Overall RAGAS score** | "
        + " | ".join(f"**{_format_score(value)}** ({score_band(value)})" for value in overalls)
        + f" | {_winner(answers, overalls)} | Mean of the metrics below |",
    ]
    for spec in specs:
        outcomes = [answer.metrics[spec.name] for answer in answers]
        values = [outcome.value for outcome in outcomes]
        lines.append(
            f"| `{spec.name}` | " + " | ".join(_cell(outcome) for outcome in outcomes)
            + f" | {_winner(answers, values)} | {spec.description} |"
        )
    lines += [
        "| Route taken | " + " | ".join(answer.run.path for answer in answers) + " | | Pipeline branch |",
        "| Pipeline confidence | "
        + " | ".join(f"{answer.run.confidence:.2f} ({answer.run.confidence_label or 'n/a'})" for answer in answers)
        + " | | Pipeline's own heuristic |",
        "| Latency | " + " | ".join(f"{answer.run.duration_seconds:.1f} s" for answer in answers) + " | | |",
    ]
    return [
        *lines,
        "",
        "_Live questions have no reference answer, so only reference-free metrics apply. "
        "Scores are 0-1, higher is better; the medical disclaimer is excluded from scoring._",
        "",
    ]


def _skip_notes_section(answers: Sequence[ScoredAnswer], specs: Sequence[MetricSpec]) -> list[str]:
    """Why any metric was not scored — refusals, missing context, judge errors."""
    notes = [
        f"- {answer.mode.upper()} `{spec.name}`: "
        f"{answer.metrics[spec.name].error or answer.metrics[spec.name].status}"
        for answer in answers
        for spec in specs
        if not answer.metrics[spec.name].scored
    ]
    if not notes:
        return []
    return ["### Not scored", "", *notes, ""]


def _answer_body(run: PipelineRun) -> str:
    if run.error:
        return f"_Pipeline failed: {run.error}_"
    return run.answer.strip() or "_(empty)_"


def _answer_section(answer: ScoredAnswer) -> list[str]:
    run = answer.run
    return [
        f"## {answer.mode.upper()} answer",
        "",
        _answer_body(run),
        "",
        f"### {answer.mode.upper()} sources",
        "",
        *_source_lines(run.citations),
        "",
        f"### {answer.mode.upper()} pipeline trace",
        "",
        *(f"- {message}" for message in run.trace),
        "",
    ]


def _source_lines(citations: Sequence[dict]) -> list[str]:
    if not citations:
        return ["_No sources were used._"]
    lines = []
    for citation in citations:
        page = f", p.{citation['page']}" if citation.get("page") is not None else ""
        snippet = " ".join(str(citation.get("snippet", "")).split())[:SNIPPET_PREVIEW_LENGTH]
        lines.append(f"{citation.get('id', '-')}. **{citation.get('source', 'unknown')}{page}** — {snippet}")
    return lines
