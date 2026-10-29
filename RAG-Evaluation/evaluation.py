"""Full RAGAS scorecard for the CRAG pipeline (src/crag_agent.py).

Where the root ``evaluate.py`` compares RAG against CRAG on three NVIDIA
metrics, this script measures the CRAG pipeline on its own across the whole
RAGAS RAG metric family, grouped by the stage of the pipeline each one judges:

    Retrieval quality
        context_precision      - are the relevant passages ranked first?       (reference)
        context_recall         - do the passages cover the reference answer?   (reference)
        context_entity_recall  - do the passages contain the reference's key
                                 clinical entities (drugs, doses, scores)?     (reference)
        context_relevance      - do the passages bear on the question at all?
    Generation quality
        faithfulness           - share of answer claims supported by context
        response_groundedness  - is the answer anchored in the context?
        answer_relevancy       - does the answer address the question asked?  (embeddings)
    Answer correctness
        answer_accuracy        - does the answer agree with the reference?     (reference)
        answer_correctness     - factual overlap + semantic similarity blend   (reference, embeddings)
        factual_correctness    - claim-level F1 against the reference          (reference)
        semantic_similarity    - embedding cosine against the reference        (reference, embeddings)

Golden-set rows with a null reference skip the reference-based metrics, and
questions the topic gate refuses skip every metric: a refusal is the correct
behaviour for an off-topic question, so it is reported as a refusal rate
rather than scored as an irrelevant answer.

The medical disclaimer appended to every answer is stripped before scoring —
it is boilerplate, not a clinical claim, and would otherwise count as an
unsupported statement against faithfulness on every single row.

The judge is ``Settings.eval_model`` (independent of ``CHAT_MODEL``) and the
embeddings are the app's own NVIDIA embedding model.

Outputs land in ``RAG-Evaluation/results/``:
    report-<timestamp>.md    - headline scorecard, per-category table, per-question table
    scores-<timestamp>.csv   - one row per question, one column per metric
    results-<timestamp>.json - everything, including answers, contexts and failures

Usage (from the repo root, with requirements-eval.txt installed):
    python RAG-Evaluation/evaluation.py
    python RAG-Evaluation/evaluation.py --limit 3
    python RAG-Evaluation/evaluation.py --category in-corpus
    python RAG-Evaluation/evaluation.py --metrics faithfulness answer_relevancy
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Sequence

EVALUATION_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVALUATION_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from ragas_compat import build_evaluator_llm  # noqa: E402  (installs the ragas import shim)
from ragas.embeddings.base import BaseRagasEmbedding  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerAccuracy,
    AnswerCorrectness,
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecisionWithReference,
    ContextRecall,
    ContextRelevance,
    FactualCorrectness,
    Faithfulness,
    ResponseGroundedness,
    SemanticSimilarity,
)

from evaluate import (  # noqa: E402
    GOLDEN_SET_PATH,
    MODE_CRAG,
    PATH_REFUSED,
    GoldenQuestion,
    PipelineRun,
    load_golden_set,
    run_pipeline,
    select_questions,
)
from src.config import Settings, get_settings  # noqa: E402
from src.crag_agent import CRAGPipeline  # noqa: E402
from src.utils import MEDICAL_DISCLAIMER  # noqa: E402
from src.vector_store import build_embeddings  # noqa: E402

logger = logging.getLogger("hkn.ragas_eval")

RESULTS_DIR = EVALUATION_DIR / "results"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
REPORT_FILENAME = "report-{timestamp}.md"
CSV_FILENAME = "scores-{timestamp}.csv"
JSON_FILENAME = "results-{timestamp}.json"

# A judge call can stall on a slow NVIDIA endpoint; cap each one, and cap how
# many run at once so a single question does not trip the endpoint's rate limit.
METRIC_TIMEOUT_SECONDS = 300.0
MAX_CONCURRENT_METRICS = 4

STAGE_RETRIEVAL = "Retrieval quality"
STAGE_GENERATION = "Generation quality"
STAGE_CORRECTNESS = "Answer correctness"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
SKIPPED_NO_REFERENCE = "skipped: no reference"
SKIPPED_NO_CONTEXT = "skipped: no retrieved context"
SKIPPED_REFUSED = "skipped: refused by topic gate"
SKIPPED_PIPELINE_FAILED = "skipped: pipeline failed"

NOT_APPLICABLE = "n/a"
METRIC_ERROR_MARKER = "ERR"
SCORE_FORMAT = "{:.3f}"
PERCENT_FORMAT = "{:.1%}"
OVERALL_LABEL = "ALL"
OFF_TOPIC_CATEGORY = "off-topic"

# Bands used only to label the headline numbers in the report.
SCORE_BANDS = ((0.85, "excellent"), (0.70, "good"), (0.50, "fair"), (0.0, "poor"))


# ---------------------------------------------------------------- embeddings


class NvidiaRagasEmbeddings(BaseRagasEmbedding):
    """The app's NVIDIA embeddings behind the interface RAGAS metrics expect.

    The NVIDIA retrieval models require an ``input_type`` that the generic
    OpenAI-compatible RAGAS client never sends; the LangChain wrapper the app
    already uses sets it, so RAGAS borrows that instead of a second client.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._embeddings = build_embeddings(settings)

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self._embeddings.embed_query(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await self._embeddings.aembed_query(text)


# ------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class ScoringInput:
    """The fields every metric draws its arguments from."""

    question: str
    response: str
    contexts: list[str]
    reference: str | None


@dataclass(frozen=True)
class MetricSpec:
    """One RAGAS metric: where it sits in the pipeline and what it needs."""

    name: str
    stage: str
    description: str
    needs_reference: bool
    needs_context: bool
    score: Callable[[ScoringInput], Awaitable]


def build_metric_specs(settings: Settings) -> list[MetricSpec]:
    """Every RAGAS RAG metric, wired to one judge LLM and one embedding model."""
    llm = build_evaluator_llm(settings)
    embeddings = NvidiaRagasEmbeddings(settings)

    context_precision = ContextPrecisionWithReference(llm=llm)
    context_recall = ContextRecall(llm=llm)
    context_entity_recall = ContextEntityRecall(llm=llm)
    context_relevance = ContextRelevance(llm=llm)
    faithfulness = Faithfulness(llm=llm)
    response_groundedness = ResponseGroundedness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    answer_accuracy = AnswerAccuracy(llm=llm)
    answer_correctness = AnswerCorrectness(llm=llm, embeddings=embeddings)
    factual_correctness = FactualCorrectness(llm=llm)
    semantic_similarity = SemanticSimilarity(embeddings=embeddings)

    return [
        MetricSpec(
            "context_precision", STAGE_RETRIEVAL,
            "Relevant passages are ranked above irrelevant ones",
            needs_reference=True, needs_context=True,
            score=lambda s: context_precision.ascore(
                user_input=s.question, reference=s.reference, retrieved_contexts=s.contexts
            ),
        ),
        MetricSpec(
            "context_recall", STAGE_RETRIEVAL,
            "Retrieved passages cover every statement in the reference answer",
            needs_reference=True, needs_context=True,
            score=lambda s: context_recall.ascore(
                user_input=s.question, retrieved_contexts=s.contexts, reference=s.reference
            ),
        ),
        MetricSpec(
            "context_entity_recall", STAGE_RETRIEVAL,
            "Retrieved passages contain the reference's key clinical entities",
            needs_reference=True, needs_context=True,
            score=lambda s: context_entity_recall.ascore(
                reference=s.reference, retrieved_contexts=s.contexts
            ),
        ),
        MetricSpec(
            "context_relevance", STAGE_RETRIEVAL,
            "Retrieved passages bear on the question",
            needs_reference=False, needs_context=True,
            score=lambda s: context_relevance.ascore(
                user_input=s.question, retrieved_contexts=s.contexts
            ),
        ),
        MetricSpec(
            "faithfulness", STAGE_GENERATION,
            "Share of answer claims supported by the retrieved passages",
            needs_reference=False, needs_context=True,
            score=lambda s: faithfulness.ascore(
                user_input=s.question, response=s.response, retrieved_contexts=s.contexts
            ),
        ),
        MetricSpec(
            "response_groundedness", STAGE_GENERATION,
            "Answer is anchored in the retrieved passages",
            needs_reference=False, needs_context=True,
            score=lambda s: response_groundedness.ascore(
                response=s.response, retrieved_contexts=s.contexts
            ),
        ),
        MetricSpec(
            "answer_relevancy", STAGE_GENERATION,
            "Answer addresses the question that was asked",
            needs_reference=False, needs_context=False,
            score=lambda s: answer_relevancy.ascore(user_input=s.question, response=s.response),
        ),
        MetricSpec(
            "answer_accuracy", STAGE_CORRECTNESS,
            "Answer agrees with the reference answer",
            needs_reference=True, needs_context=False,
            score=lambda s: answer_accuracy.ascore(
                user_input=s.question, response=s.response, reference=s.reference
            ),
        ),
        MetricSpec(
            "answer_correctness", STAGE_CORRECTNESS,
            "Factual overlap with the reference, blended with semantic similarity",
            needs_reference=True, needs_context=False,
            score=lambda s: answer_correctness.ascore(
                user_input=s.question, response=s.response, reference=s.reference
            ),
        ),
        MetricSpec(
            "factual_correctness", STAGE_CORRECTNESS,
            "Claim-level F1 between the answer and the reference",
            needs_reference=True, needs_context=False,
            score=lambda s: factual_correctness.ascore(response=s.response, reference=s.reference),
        ),
        MetricSpec(
            "semantic_similarity", STAGE_CORRECTNESS,
            "Embedding similarity between the answer and the reference",
            needs_reference=True, needs_context=False,
            score=lambda s: semantic_similarity.ascore(reference=s.reference, response=s.response),
        ),
    ]


def select_metric_specs(specs: Sequence[MetricSpec], names: Sequence[str] | None) -> list[MetricSpec]:
    if not names:
        return list(specs)
    known = {spec.name for spec in specs}
    unknown = sorted(set(names) - known)
    if unknown:
        raise RuntimeError(f"unknown metric(s) {unknown}; choose from {sorted(known)}")
    return [spec for spec in specs if spec.name in names]


# ------------------------------------------------------------------- scoring


@dataclass
class MetricOutcome:
    """One metric's verdict for one answer: a score, a skip, or a failure."""

    value: float | None = None
    status: str = STATUS_OK
    error: str | None = None

    @property
    def scored(self) -> bool:
        return self.value is not None


def strip_disclaimer(answer: str) -> str:
    return answer.replace(MEDICAL_DISCLAIMER, "").strip()


def _skip_reason(spec: MetricSpec, run: PipelineRun, sample: ScoringInput) -> str | None:
    """Why ``spec`` cannot be scored for this row, or ``None`` when it can."""
    if run.error:
        return SKIPPED_PIPELINE_FAILED
    if run.path == PATH_REFUSED:
        return SKIPPED_REFUSED
    if spec.needs_context and not sample.contexts:
        return SKIPPED_NO_CONTEXT
    if spec.needs_reference and sample.reference is None:
        return SKIPPED_NO_REFERENCE
    return None


class Scorer:
    """Runs the selected metrics for one answer, bounded and failure-tolerant."""

    def __init__(self, specs: Sequence[MetricSpec]) -> None:
        self.specs = list(specs)
        self._slots = asyncio.Semaphore(MAX_CONCURRENT_METRICS)

    async def score(self, question: GoldenQuestion, run: PipelineRun) -> dict[str, MetricOutcome]:
        sample = ScoringInput(
            question=question.question,
            response=strip_disclaimer(run.answer),
            contexts=run.contexts,
            reference=question.reference,
        )
        outcomes = await asyncio.gather(*(self._score_one(spec, run, sample) for spec in self.specs))
        return {spec.name: outcome for spec, outcome in zip(self.specs, outcomes)}

    async def _score_one(self, spec: MetricSpec, run: PipelineRun, sample: ScoringInput) -> MetricOutcome:
        skip_reason = _skip_reason(spec, run, sample)
        if skip_reason:
            return MetricOutcome(status=skip_reason)
        async with self._slots:
            try:
                result = await asyncio.wait_for(spec.score(sample), timeout=METRIC_TIMEOUT_SECONDS)
            except Exception as exc:  # noqa: BLE001 - recorded on the row and reported
                logger.warning("metric %s failed for %r", spec.name, sample.question[:70], exc_info=True)
                return MetricOutcome(status=STATUS_FAILED, error=f"{type(exc).__name__}: {exc}")
        value = float(result.value)
        if value != value:  # NaN: the judge returned nothing parseable.
            return MetricOutcome(status=STATUS_FAILED, error="judge returned an unparseable score")
        return MetricOutcome(value=value)


# ------------------------------------------------------------------- results


@dataclass
class EvaluatedQuestion:
    question: GoldenQuestion
    run: PipelineRun
    metrics: dict[str, MetricOutcome]

    def to_dict(self) -> dict:
        return {
            "id": self.question.id,
            "category": self.question.category,
            "question": self.question.question,
            "reference": self.question.reference,
            "answer": self.run.answer,
            "contexts": self.run.contexts,
            "citations": self.run.citations,
            "path": self.run.path,
            "confidence": self.run.confidence,
            "trace": self.run.trace,
            "duration_seconds": round(self.run.duration_seconds, 2),
            "error": self.run.error,
            "metrics": {name: asdict(outcome) for name, outcome in self.metrics.items()},
        }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _scored_values(rows: Sequence[EvaluatedQuestion], metric: str) -> list[float]:
    return [row.metrics[metric].value for row in rows if row.metrics[metric].scored]


def summarise(rows: Sequence[EvaluatedQuestion], specs: Sequence[MetricSpec]) -> dict:
    """Per-metric mean overall and per category, plus pipeline-level rates."""
    categories = sorted({row.question.category for row in rows})
    buckets = {OVERALL_LABEL: list(rows)} | {
        category: [row for row in rows if row.question.category == category]
        for category in categories
    }
    return {
        "metrics": {
            label: {
                spec.name: {
                    "mean": _mean(_scored_values(bucket, spec.name)),
                    "scored": len(_scored_values(bucket, spec.name)),
                    "failed": sum(1 for row in bucket if row.metrics[spec.name].status == STATUS_FAILED),
                }
                for spec in specs
            }
            for label, bucket in buckets.items()
        },
        "pipeline": _pipeline_stats(rows),
    }


def _pipeline_stats(rows: Sequence[EvaluatedQuestion]) -> dict:
    off_topic = [row for row in rows if row.question.category == OFF_TOPIC_CATEGORY]
    on_topic = [row for row in rows if row.question.category != OFF_TOPIC_CATEGORY]
    return {
        "questions": len(rows),
        "pipeline_failures": sum(1 for row in rows if row.run.error),
        "off_topic_refusal_rate": _mean([1.0 if row.run.path == PATH_REFUSED else 0.0 for row in off_topic]),
        "on_topic_answer_rate": _mean([0.0 if row.run.path == PATH_REFUSED or row.run.error else 1.0 for row in on_topic]),
        "mean_latency_seconds": _mean([row.run.duration_seconds for row in rows]),
        "routing": {path: sum(1 for row in rows if row.run.path == path) for path in sorted({row.run.path for row in rows})},
    }


def _format_score(value: float | None) -> str:
    return SCORE_FORMAT.format(value) if value is not None else NOT_APPLICABLE


def _format_outcome(outcome: MetricOutcome) -> str:
    if outcome.scored:
        return SCORE_FORMAT.format(outcome.value)
    return METRIC_ERROR_MARKER if outcome.status == STATUS_FAILED else NOT_APPLICABLE


def score_band(value: float | None) -> str:
    if value is None:
        return NOT_APPLICABLE
    return next(label for floor, label in SCORE_BANDS if value >= floor)


def _stage_mean(overall: dict, specs: Sequence[MetricSpec], stage: str) -> float | None:
    return _mean([overall[spec.name]["mean"] for spec in specs if spec.stage == stage and overall[spec.name]["mean"] is not None])


# ------------------------------------------------------------------- report


def render_report(
    rows: Sequence[EvaluatedQuestion],
    specs: Sequence[MetricSpec],
    summary: dict,
    settings: Settings,
    elapsed_seconds: float,
) -> str:
    return "\n".join(
        [
            *_report_header(rows, settings, elapsed_seconds),
            *_report_stage_scores(specs, summary),
            *_report_metric_table(specs, summary),
            *_report_pipeline_behaviour(summary["pipeline"]),
            *_report_category_table(specs, summary),
            *_report_question_table(rows, specs),
            *_report_notes(),
        ]
    )


def _judge_line(settings: Settings) -> str:
    if settings.eval_model == settings.chat_model:
        return (
            f"- **Judge:** `{settings.eval_model}` - **same model as the generator, "
            "so these scores are self-graded and likely inflated**"
        )
    return f"- **Judge (independent of generator):** `{settings.eval_model}`"


def _report_header(rows: Sequence[EvaluatedQuestion], settings: Settings, elapsed_seconds: float) -> list[str]:
    return [
        "# CRAG pipeline - RAGAS evaluation",
        "",
        f"- **Run:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ({elapsed_seconds / 60:.1f} min)",
        f"- **Questions:** {len(rows)} from `eval/golden_set.json`",
        f"- **Generator:** `{settings.chat_model}`",
        _judge_line(settings),
        f"- **Embeddings:** `{settings.embedding_model}` · top_k={settings.top_k}",
        "",
    ]


def _report_stage_scores(specs: Sequence[MetricSpec], summary: dict) -> list[str]:
    overall = summary["metrics"][OVERALL_LABEL]
    lines = ["## Headline", "", "| Stage | Mean score | Rating |", "|---|---:|---|"]
    for stage in (STAGE_RETRIEVAL, STAGE_GENERATION, STAGE_CORRECTNESS):
        value = _stage_mean(overall, specs, stage)
        lines.append(f"| {stage} | {_format_score(value)} | {score_band(value)} |")
    return [*lines, ""]


def _report_metric_table(specs: Sequence[MetricSpec], summary: dict) -> list[str]:
    overall = summary["metrics"][OVERALL_LABEL]
    lines = [
        "## RAGAS metrics (all questions)",
        "",
        "| Stage | Metric | Mean | Rating | Scored rows | Judge errors | What it measures |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for spec in specs:
        stats = overall[spec.name]
        lines.append(
            f"| {spec.stage} | `{spec.name}` | {_format_score(stats['mean'])} | {score_band(stats['mean'])} "
            f"| {stats['scored']} | {stats['failed']} | {spec.description} |"
        )
    return [*lines, ""]


def _report_pipeline_behaviour(pipeline: dict) -> list[str]:
    routing = ", ".join(f"{path}: {count}" for path, count in pipeline["routing"].items())
    refusal = pipeline["off_topic_refusal_rate"]
    answered = pipeline["on_topic_answer_rate"]
    return [
        "## Pipeline behaviour",
        "",
        f"- **Off-topic questions correctly refused:** {PERCENT_FORMAT.format(refusal) if refusal is not None else NOT_APPLICABLE}",
        f"- **Medical questions answered (not refused):** {PERCENT_FORMAT.format(answered) if answered is not None else NOT_APPLICABLE}",
        f"- **Pipeline failures:** {pipeline['pipeline_failures']} / {pipeline['questions']}",
        f"- **Mean latency per question:** {_format_score(pipeline['mean_latency_seconds'])} s",
        f"- **CRAG routing decisions:** {routing}",
        "",
    ]


def _report_category_table(specs: Sequence[MetricSpec], summary: dict) -> list[str]:
    categories = [label for label in summary["metrics"] if label != OVERALL_LABEL]
    lines = [
        "## By golden-set category",
        "",
        "| Metric | " + " | ".join(categories) + " |",
        "|---|" + "---:|" * len(categories),
    ]
    for spec in specs:
        cells = " | ".join(_format_score(summary["metrics"][category][spec.name]["mean"]) for category in categories)
        lines.append(f"| `{spec.name}` | {cells} |")
    return [*lines, ""]


def _report_question_table(rows: Sequence[EvaluatedQuestion], specs: Sequence[MetricSpec]) -> list[str]:
    lines = [
        "## Per question",
        "",
        "| id | path | " + " | ".join(spec.name for spec in specs) + " |",
        "|---|---|" + "---:|" * len(specs),
    ]
    for row in rows:
        cells = " | ".join(_format_outcome(row.metrics[spec.name]) for spec in specs)
        lines.append(f"| {row.question.id} | {row.run.path} | {cells} |")
    return [*lines, ""]


def _report_notes() -> list[str]:
    return [
        "## Reading these numbers",
        "",
        "- Scores are 0-1; higher is better. `n/a` means the metric did not apply to that row "
        "(no reference answer, no retrieved context, or the question was correctly refused); "
        f"`{METRIC_ERROR_MARKER}` means the judge call failed and the row was left out of the mean.",
        "- Reference-based metrics only cover golden-set rows that assert a reference answer.",
        "- The medical disclaimer is stripped from answers before scoring.",
        "",
    ]


# ------------------------------------------------------------------- writing


def write_outputs(
    rows: Sequence[EvaluatedQuestion],
    specs: Sequence[MetricSpec],
    summary: dict,
    report: str,
    settings: Settings,
    arguments: argparse.Namespace,
) -> list[Path]:
    timestamp = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / REPORT_FILENAME.format(timestamp=timestamp)
    csv_path = RESULTS_DIR / CSV_FILENAME.format(timestamp=timestamp)
    json_path = RESULTS_DIR / JSON_FILENAME.format(timestamp=timestamp)

    report_path.write_text(report, encoding="utf-8")
    _write_csv(csv_path, rows, specs)
    json_path.write_text(
        json.dumps(
            {
                "run": {
                    "generator_model": settings.chat_model,
                    "evaluator_model": settings.eval_model,
                    "embedding_model": settings.embedding_model,
                    "top_k": settings.top_k,
                    "topic_score_threshold": settings.topic_score_threshold,
                    "arguments": vars(arguments),
                },
                "summary": summary,
                "rows": [row.to_dict() for row in rows],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return [report_path, csv_path, json_path]


def _write_csv(path: Path, rows: Sequence[EvaluatedQuestion], specs: Sequence[MetricSpec]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "path", "duration_seconds", *(spec.name for spec in specs)])
        for row in rows:
            writer.writerow(
                [
                    row.question.id,
                    row.question.category,
                    row.run.path,
                    round(row.run.duration_seconds, 2),
                    *(row.metrics[spec.name].value if row.metrics[spec.name].scored else "" for spec in specs),
                ]
            )


# --------------------------------------------------------------------- main


async def evaluate_crag(
    questions: Sequence[GoldenQuestion], pipeline: CRAGPipeline, scorer: Scorer
) -> list[EvaluatedQuestion]:
    rows: list[EvaluatedQuestion] = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question.id} ...", end="", flush=True)
        run = await run_pipeline(pipeline, MODE_CRAG, question.question)
        metrics = await scorer.score(question, run)
        rows.append(EvaluatedQuestion(question=question, run=run, metrics=metrics))
        print(f" {run.path} ({run.duration_seconds:.1f}s)", flush=True)
    return rows


def print_scorecard(specs: Sequence[MetricSpec], summary: dict) -> None:
    overall = summary["metrics"][OVERALL_LABEL]
    print("\nCRAG RAGAS SCORECARD")
    print(f"{'stage':<20}{'metric':<24}{'mean':>8}{'scored':>8}  rating")
    print("-" * 70)
    for spec in specs:
        stats = overall[spec.name]
        print(
            f"{spec.stage:<20}{spec.name:<24}{_format_score(stats['mean']):>8}"
            f"{stats['scored']:>8}  {score_band(stats['mean'])}"
        )
    refusal = summary["pipeline"]["off_topic_refusal_rate"]
    if refusal is not None:
        print(f"\noff-topic refusal rate: {PERCENT_FORMAT.format(refusal)}")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--category", default=None, help="evaluate only one golden-set category")
    parser.add_argument("--metrics", nargs="+", default=None, help="score only these metrics")
    return parser.parse_args(argv)


async def main_async(arguments: argparse.Namespace) -> int:
    settings = get_settings()
    questions = select_questions(load_golden_set(GOLDEN_SET_PATH), arguments.category, arguments.limit)
    specs = select_metric_specs(build_metric_specs(settings), arguments.metrics)
    print(
        f"Evaluating CRAG on {len(questions)} question(s) with {len(specs)} RAGAS metric(s)\n"
        f"  generator : {settings.chat_model}\n"
        f"  judge     : {settings.eval_model}\n"
        f"  embeddings: {settings.embedding_model}"
    )

    started = time.perf_counter()
    rows = await evaluate_crag(questions, CRAGPipeline(settings), Scorer(specs))
    elapsed = time.perf_counter() - started

    summary = summarise(rows, specs)
    print_scorecard(specs, summary)
    report = render_report(rows, specs, summary, settings, elapsed)
    paths = write_outputs(rows, specs, summary, report, settings, arguments)
    print(f"\nDone in {elapsed / 60:.1f} min. Outputs:")
    for path in paths:
        print(f"  {path}")
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
