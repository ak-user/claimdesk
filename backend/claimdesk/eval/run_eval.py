"""Scores retrieval and answers against a small labelled set.

    python -m claimdesk.eval.run_eval                  # print the report
    python -m claimdesk.eval.run_eval --mlflow         # also log params and metrics to MLflow
    python -m claimdesk.eval.run_eval --min-hit-rate 0.9 --min-answer-accuracy 0.9   # CI gate

Metrics:
- hit_rate_at_k: share of questions where a labelled section is in the top k search results.
- mrr: mean reciprocal rank of the first labelled section.
- answer_accuracy: share of answers that contain every expected phrase.
- cites_labelled_section: share of answers that cite at least one labelled section.
- grounded_rate: share of answers with no citation removed by the grounding check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..app_state import Components, build_components
from ..config import Settings

DATASET = Path(__file__).with_name("dataset.jsonl")


def load_dataset(path: Path = DATASET) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(components: Components, rows: list[dict[str, Any]], k: int = 3) -> dict[str, Any]:
    results = []
    for row in rows:
        labelled = set(row["sections"])
        ranked = [hit.chunk.section for hit in components.retriever.search(row["question"], k)]
        rank = next((i + 1 for i, section in enumerate(ranked) if section in labelled), None)
        answer = components.agent.answer(row["question"])
        text = answer.answer.lower()
        cited = {citation.section for citation in answer.citations}
        results.append(
            {
                "question": row["question"],
                "rank": rank,
                "correct": all(phrase.lower() in text for phrase in row["expect"]),
                "cites_labelled": bool(cited & labelled),
                "grounded": not any(f.startswith("removed_unsupported") for f in answer.flags),
                "answer": answer.answer,
            }
        )
    n = len(results)
    metrics = {
        "hit_rate_at_k": sum(r["rank"] is not None for r in results) / n,
        "mrr": sum(1 / r["rank"] for r in results if r["rank"]) / n,
        "answer_accuracy": sum(r["correct"] for r in results) / n,
        "cites_labelled_section": sum(r["cites_labelled"] for r in results) / n,
        "grounded_rate": sum(r["grounded"] for r in results) / n,
    }
    return {"metrics": {name: round(value, 3) for name, value in metrics.items()}, "rows": results}


def print_report(report: dict[str, Any], components: Components, k: int) -> None:
    print(f"provider={components.provider} retrieval={components.retriever.mode} k={k}\n")
    print(f"{'question':<58} {'rank':>4}  answer  cites")
    for row in report["rows"]:
        question = row["question"] if len(row["question"]) <= 56 else row["question"][:55] + "…"
        rank = str(row["rank"]) if row["rank"] else "-"
        answer = "ok" if row["correct"] else "MISS"
        cites = "ok" if row["cites_labelled"] else "MISS"
        print(f"{question:<58} {rank:>4}  {answer:<6}  {cites}")
    print()
    for name, value in report["metrics"].items():
        print(f"{name:<24} {value:.3f}")


def log_to_mlflow(report: dict[str, Any], components: Components, k: int) -> None:
    """Logs to MLFLOW_TRACKING_URI, e.g. "databricks" with DATABRICKS_HOST and DATABRICKS_TOKEN.

    Without it, runs go to a local ./mlruns folder. MLflow 3 keeps that file store only in
    maintenance mode, so this opts in explicitly; use a server or Databricks for anything shared.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("claimdesk-eval")
    with mlflow.start_run(run_name=f"{components.provider}-{components.retriever.mode}-k{k}"):
        mlflow.log_params(
            {
                "provider": components.provider,
                "retrieval": components.retriever.mode,
                "k": k,
                "questions": len(report["rows"]),
            }
        )
        mlflow.log_metrics(report["metrics"])
        mlflow.log_dict({"rows": report["rows"]}, "rows.json")
    print(f"\nLogged to MLflow at {mlflow.get_tracking_uri()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--mlflow", action="store_true", help="log the run to MLflow")
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-answer-accuracy", type=float, default=0.0)
    args = parser.parse_args(argv)

    components = build_components(Settings())
    report = evaluate(components, load_dataset(args.dataset), args.k)
    print_report(report, components, args.k)
    if args.mlflow:
        log_to_mlflow(report, components, args.k)

    metrics = report["metrics"]
    if metrics["hit_rate_at_k"] < args.min_hit_rate:
        print(f"FAIL: hit rate {metrics['hit_rate_at_k']} < {args.min_hit_rate}")
        return 1
    if metrics["answer_accuracy"] < args.min_answer_accuracy:
        print(f"FAIL: answer accuracy {metrics['answer_accuracy']} < {args.min_answer_accuracy}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
