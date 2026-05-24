"""Run DeepEval over LightRAG's answers to the handcrafted Q&A set."""
import json
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from lightrag import QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag_config import build_lightrag, describe_config

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    GEval,
    PatternMatchMetric,
)
from deepeval_config import build_judge_model, describe_judge_model

load_dotenv()

QA_PATH = Path(os.getenv("QA_PATH", "tests/handcrafted_qa.json"))
RESULTS_PATH = Path(os.getenv("RESULTS_PATH", "results/handcrafted_scores.json"))
STORAGE_DIR = Path("rag_storage")
QUERY_MODE = os.getenv("RAG_QUERY_MODE", "hybrid")
DEEPEVAL_MAX_CONCURRENT = int(os.getenv("DEEPEVAL_MAX_CONCURRENT", "1"))
DEEPEVAL_CASE_LIMIT = int(os.getenv("DEEPEVAL_CASE_LIMIT", "0"))
DEEPEVAL_CONTEXT_CHAR_LIMIT = int(os.getenv("DEEPEVAL_CONTEXT_CHAR_LIMIT", "0"))
DEEPEVAL_METRICS = os.getenv(
    "DEEPEVAL_METRICS",
    "answer_relevancy,faithfulness,contextual_relevancy,contextual_recall,refusal",
)
DEEPEVAL_INCLUDE_REASON = os.getenv("DEEPEVAL_INCLUDE_REASON", "true").lower() in {
    "1",
    "true",
    "yes",
}
DEEPEVAL_ASYNC_MODE = os.getenv("DEEPEVAL_ASYNC_MODE", "true").lower() in {
    "1",
    "true",
    "yes",
}
DEEPEVAL_IGNORE_ERRORS = os.getenv("DEEPEVAL_IGNORE_ERRORS", "false").lower() in {
    "1",
    "true",
    "yes",
}


def metric_name(metric) -> str:
    return getattr(metric, "name", getattr(metric, "__name__", type(metric).__name__))


def limit_context(contexts: list[str]) -> list[str]:
    if DEEPEVAL_CONTEXT_CHAR_LIMIT <= 0:
        return contexts
    return [context[:DEEPEVAL_CONTEXT_CHAR_LIMIT] for context in contexts]


async def run_queries() -> list[LLMTestCase]:
    print(f"Using {describe_config(STORAGE_DIR)}")
    rag = build_lightrag(STORAGE_DIR)
    await rag.initialize_storages()
    await initialize_pipeline_status()

    print(f"Using QA file {QA_PATH}")
    qa = json.loads(QA_PATH.read_text())
    if DEEPEVAL_CASE_LIMIT > 0:
        qa = qa[:DEEPEVAL_CASE_LIMIT]
        print(f"Limiting eval to first {len(qa)} case(s)")
    cases: list[LLMTestCase] = []

    for i, item in enumerate(qa, 1):
        question = item["question"]
        print(f"[{i}/{len(qa)}] {item['bucket']:11s} | {question[:60]}")
        answer = await rag.aquery(question, param=QueryParam(mode=QUERY_MODE))
        context_str = await rag.aquery(
            question, param=QueryParam(mode=QUERY_MODE, only_need_context=True)
        )
        # DeepEval expects retrieval_context as list[str]
        retrieval_context = [context_str] if isinstance(context_str, str) else list(context_str)
        retrieval_context = limit_context(retrieval_context)
        cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                expected_output=item["expected"],
                retrieval_context=retrieval_context,
            )
        )
    return cases


def main() -> None:
    print(f"Using query mode {QUERY_MODE}")
    cases = asyncio.run(run_queries())
    selected_metrics = [
        metric.strip().lower()
        for metric in DEEPEVAL_METRICS.split(",")
        if metric.strip()
    ]
    known_metrics = {
        "answer_relevancy",
        "faithfulness",
        "contextual_relevancy",
        "contextual_recall",
        "refusal",
        "refusal_pattern",
        "exact_match",
    }
    unknown_metrics = sorted(set(selected_metrics) - known_metrics)
    if unknown_metrics:
        raise ValueError(
            f"Unknown DEEPEVAL_METRICS: {', '.join(unknown_metrics)}. "
            f"Known metrics: {', '.join(sorted(known_metrics))}"
        )

    judge_model = None
    deterministic_metrics = {"exact_match", "refusal_pattern"}
    if any(metric not in deterministic_metrics for metric in selected_metrics):
        judge_model = build_judge_model()
        print(f"Using {describe_judge_model()}")

    metrics = []
    for metric in selected_metrics:
        if metric == "answer_relevancy":
            metrics.append(
                AnswerRelevancyMetric(
                    threshold=0.7,
                    model=judge_model,
                    include_reason=DEEPEVAL_INCLUDE_REASON,
                    async_mode=DEEPEVAL_ASYNC_MODE,
                )
            )
        elif metric == "faithfulness":
            metrics.append(
                FaithfulnessMetric(
                    threshold=0.7,
                    model=judge_model,
                    include_reason=DEEPEVAL_INCLUDE_REASON,
                    async_mode=DEEPEVAL_ASYNC_MODE,
                )
            )
        elif metric == "contextual_relevancy":
            metrics.append(
                ContextualRelevancyMetric(
                    threshold=0.7,
                    model=judge_model,
                    include_reason=DEEPEVAL_INCLUDE_REASON,
                    async_mode=DEEPEVAL_ASYNC_MODE,
                )
            )
        elif metric == "contextual_recall":
            metrics.append(
                ContextualRecallMetric(
                    threshold=0.7,
                    model=judge_model,
                    include_reason=DEEPEVAL_INCLUDE_REASON,
                    async_mode=DEEPEVAL_ASYNC_MODE,
                )
            )
        elif metric == "refusal":
            metrics.append(
                GEval(
                    name="RefusalCorrectness",
                    criteria=(
                        "If the retrieval context does not contain enough information to answer "
                        "the input, the actual_output should explicitly say so (e.g. 'I don't know', "
                        "'the documents don't cover this', 'no information available') rather than "
                        "making up an answer."
                    ),
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                        LLMTestCaseParams.RETRIEVAL_CONTEXT,
                    ],
                    model=judge_model,
                    async_mode=DEEPEVAL_ASYNC_MODE,
                )
            )
        elif metric == "refusal_pattern":
            metrics.append(
                PatternMatchMetric(
                    pattern=(
                        "(?s).*("
                        "not have enough information|not enough information|"
                        "does not contain|don't know|do not know|no information"
                        ").*"
                    ),
                    ignore_case=True,
                    threshold=1.0,
                )
            )
        elif metric == "exact_match":
            metrics.append(ExactMatchMetric(threshold=1.0))
    print(f"Using metrics: {', '.join(metric_name(metric) for metric in metrics)}")

    result = evaluate(
        test_cases=cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=DEEPEVAL_MAX_CONCURRENT),
        display_config=DisplayConfig(print_results=True),
        error_config=ErrorConfig(ignore_errors=DEEPEVAL_IGNORE_ERRORS),
    )

    # Persist a compact JSON summary for the article
    summary = []
    for tc, tr in zip(cases, result.test_results):
        summary.append({
            "input": tc.input,
            "expected": tc.expected_output,
            "actual": tc.actual_output,
            "scores": {metric_name(m): m.score for m in tr.metrics_data},
            "passed": {metric_name(m): m.success for m in tr.metrics_data},
        })
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
