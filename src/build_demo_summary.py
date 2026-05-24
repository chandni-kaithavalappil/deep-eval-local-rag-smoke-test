"""Build consolidated JSON and Markdown summaries from demo score files."""
from __future__ import annotations

import json
from pathlib import Path


SCORE_FILES = {
    "Answer Relevancy": Path("results/demo_answer_relevancy_scores.json"),
    "Contextual Relevancy": Path("results/demo_contextual_relevancy_scores.json"),
    "Faithfulness": Path("results/demo_faithfulness_scores.json"),
    "Refusal Pattern": Path("results/demo_refusal_pattern_scores.json"),
}


def main() -> None:
    summary = {}
    rows = []

    for label, path in SCORE_FILES.items():
        data = json.loads(path.read_text())
        metric_scores = []
        for item in data:
            for metric, score in item["scores"].items():
                passed = item["passed"][metric]
                metric_scores.append((item["input"], metric, score, passed))
                rows.append(
                    {
                        "suite": label,
                        "input": item["input"],
                        "metric": metric,
                        "score": score,
                        "passed": passed,
                        "actual": item["actual"],
                    }
                )

        scores = [score for _, _, score, _ in metric_scores if score is not None]
        summary[label] = {
            "cases": len(metric_scores),
            "average": round(sum(scores) / len(scores), 3) if scores else None,
            "passed": sum(1 for *_, passed in metric_scores if passed),
            "total": len(metric_scores),
        }

    output = {"summary": summary, "rows": rows}
    Path("results/demo_scores_summary.json").write_text(json.dumps(output, indent=2))
    Path("results/demo_scores_summary.md").write_text(to_markdown(summary, rows))

    print("\n=== DEEPEVAL SMOKE DEMO SUMMARY ===")
    for label, aggregate in summary.items():
        print(
            f"{label:22s} "
            f"avg={aggregate['average']:.3f} "
            f"pass={aggregate['passed']}/{aggregate['total']}"
        )
    print("\nArtifacts:")
    print("results/demo_scores_summary.json")
    print("results/demo_scores_summary.md")


def to_markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        "# DeepEval Smoke Demo Summary",
        "",
        "| Metric suite | Avg score | Pass rate | Cases |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, aggregate in summary.items():
        lines.append(
            f"| {label} | {aggregate['average']:.3f} | "
            f"{aggregate['passed']}/{aggregate['total']} | {aggregate['cases']} |"
        )

    lines.extend(
        [
            "",
            "## Case Scores",
            "",
            "| Metric | Case | Score | Passed |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for row in rows:
        short_input = row["input"][:72] + ("..." if len(row["input"]) > 72 else "")
        lines.append(
            f"| {row['metric']} | {short_input} | "
            f"{row['score']:.3f} | {row['passed']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
