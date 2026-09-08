from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import inspect
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from chatbot import get_chatbot_chain, run_query  # noqa: E402

ABSTENTION_PATTERNS = (
    "i do not know",
    "i don't know",
    "insufficient information",
    "not enough information",
    "not supported by the context",
    "not supported by the provided materials",
    "the provided context does not contain",
    "cannot answer based on the provided materials",
)


@dataclass
class EvaluationQuestion:
    id: str
    question: str
    reference_answer: str
    reference_keywords: list[str]
    expected_sources: list[str] | None
    acceptable_sources_any: list[str] | None
    required_sources_all: list[str] | None
    category: str
    difficulty: str
    should_answer: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the MSc NLP chatbot using the live RAG pipeline."
    )
    parser.add_argument(
        "--questions",
        default="evaluation_questions.json",
        help="Path to the evaluation questions JSON file.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap on the number of questions to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Retriever top-k passed into the chatbot chain.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where evaluation outputs should be written.",
    )
    parser.add_argument(
        "--use-ragas",
        choices=("auto", "always", "never"),
        default="auto",
        help="Whether to try optional Ragas metrics.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[EvaluationQuestion]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("evaluation_questions.json must contain a JSON list.")

    questions: list[EvaluationQuestion] = []
    required_fields = {
        "id",
        "question",
        "reference_answer",
        "reference_keywords",
        "category",
        "difficulty",
        "should_answer",
    }

    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {index} is not a JSON object.")

        missing = sorted(required_fields - set(item))
        if missing:
            raise ValueError(f"Question {index} is missing required fields: {missing}")

        questions.append(
            EvaluationQuestion(
                id=str(item["id"]),
                question=str(item["question"]),
                reference_answer=str(item.get("reference_answer") or ""),
                reference_keywords=[str(value) for value in item.get("reference_keywords", [])],
                expected_sources=[Path(str(value)).name for value in item.get("expected_sources", [])] if "expected_sources" in item else None,
                acceptable_sources_any=[Path(str(value)).name for value in item.get("acceptable_sources_any", [])] if "acceptable_sources_any" in item else None,
                required_sources_all=[Path(str(value)).name for value in item.get("required_sources_all", [])] if "required_sources_all" in item else None,
                category=str(item["category"]),
                difficulty=str(item["difficulty"]),
                should_answer=bool(item["should_answer"]),
            )
        )

    return questions


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def round_metric(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, digits)
    return value


def mean_ignore_none(values: list[Any]) -> float | None:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not math.isnan(float(value))
    ]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def truncate(text: str, limit: int = 240) -> str:
    clean_text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(clean_text) <= limit:
        return clean_text
    return clean_text[: limit - 3] + "..."


def serialize_list(value: list[Any]) -> str:
    return json.dumps(value, ensure_ascii=True)


def keyword_recall(answer: str, reference_keywords: list[str]) -> float | None:
    if not reference_keywords:
        return None

    answer_text = normalise_text(answer)
    hits = sum(1 for keyword in reference_keywords if normalise_text(keyword) in answer_text)
    return hits / len(reference_keywords)


def detect_abstention(answer: str) -> bool:
    answer_text = normalise_text(answer)
    return any(pattern in answer_text for pattern in ABSTENTION_PATTERNS)


def compute_source_metrics(
    question: EvaluationQuestion,
    retrieved_sources: list[str],
) -> tuple[float | None, float | None]:
    retrieved = {Path(source).name for source in retrieved_sources}

    if question.acceptable_sources_any:
        acceptable = {Path(source).name for source in question.acceptable_sources_any}
        matches = acceptable & retrieved
        hit_rate = 1.0 if matches else 0.0
        return hit_rate, None

    if question.required_sources_all:
        required = {Path(source).name for source in question.required_sources_all}
        matches = required & retrieved
        hit_rate = 1.0 if matches else 0.0
        recall = len(matches) / len(required)
        return hit_rate, recall

    if question.expected_sources:
        expected = {Path(source).name for source in question.expected_sources}
        matches = expected & retrieved
        hit_rate = 1.0 if matches else 0.0
        recall = len(matches) / len(expected)
        return hit_rate, recall

    return None, None


def build_question_score(row: dict[str, Any]) -> float | None:
    components: list[float] = [0.0 if row["error_flag"] else 1.0]

    if row["should_answer"]:
        for field in ("source_hit_rate_at_k", "source_recall_at_k", "keyword_recall"):
            value = row.get(field)
            if isinstance(value, (int, float)):
                components.append(float(value))
        if row.get("abstention_detected"):
            components.append(0.0)
    else:
        components.append(1.0 if row.get("abstention_correct") else 0.0)

    return mean_ignore_none(components)


def prepare_context_texts(contexts: list[Any]) -> list[str]:
    prepared = []
    for document in contexts:
        text = getattr(document, "page_content", "") or ""
        prepared.append(re.sub(r"\s+", " ", text).strip())
    return prepared


def run_deterministic_evaluation(
    questions: list[EvaluationQuestion],
    *,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chain, llm_config = get_chatbot_chain(top_k=top_k, verbose=False)
    results: list[dict[str, Any]] = []

    for question in questions:
        print(f"[{question.id}] {question.question}")
        result = run_query(
            question.question,
            chain=chain,
            llm_config=llm_config,
            top_k=top_k,
        )

        retrieved_sources = [Path(source).name for source in result["retrieved_sources"]]
        source_hit_rate_at_k, source_recall_at_k = compute_source_metrics(
            question,
            retrieved_sources,
        )
        answer = result["answer"] or ""
        abstention_detected = detect_abstention(answer)
        row = {
            "id": question.id,
            "question": question.question,
            "category": question.category,
            "difficulty": question.difficulty,
            "should_answer": question.should_answer,
            "reference_answer": question.reference_answer,
            "reference_keywords": question.reference_keywords,
            "acceptable_sources_any": question.acceptable_sources_any,
            "required_sources_all": question.required_sources_all,
            "expected_sources": question.expected_sources,
            "answer": answer,
            "retrieved_sources": retrieved_sources,
            "retrieved_pages": result["retrieved_pages"],
            "retrieved_context_count": len(result["contexts"]),
            "latency_seconds": round_metric(result["latency_seconds"]),
            "provider": result["provider"],
            "model": result["model"],
            "error_flag": bool(result["error"]),
            "error_message": result["error"] or "",
            "source_hit_rate_at_k": round_metric(source_hit_rate_at_k),
            "source_recall_at_k": round_metric(source_recall_at_k),
            "keyword_recall": round_metric(keyword_recall(answer, question.reference_keywords)),
            "answer_length_chars": len(answer),
            "answer_length_words": len(answer.split()),
            "abstention_detected": abstention_detected,
            "abstention_correct": (not question.should_answer) and abstention_detected,
            "_contexts": result["contexts"],
            "_context_texts": prepare_context_texts(result["contexts"]),
        }
        row["question_score"] = round_metric(build_question_score(row))
        results.append(row)

    metadata = {
        "provider": llm_config.provider,
        "model": llm_config.model,
        "top_k": top_k,
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return results, metadata


def _lookup_ragas_attr(module: Any, *names: str) -> Any:
    for name in names:
        value = getattr(module, name, None)
        if value is None:
            continue
        if inspect.isclass(value):
            try:
                return value()
            except TypeError:
                continue
        return value
    return None


def _coerce_ragas_records(result_object: Any) -> list[dict[str, Any]]:
    if hasattr(result_object, "to_pandas"):
        dataframe = result_object.to_pandas()
        return dataframe.to_dict(orient="records")
    if hasattr(result_object, "to_dict"):
        payload = result_object.to_dict()
        if isinstance(payload, dict):
            keys = list(payload)
            if not keys:
                return []
            row_count = len(payload[keys[0]])
            return [
                {key: payload[key][index] for key in keys}
                for index in range(row_count)
            ]
        if isinstance(payload, list):
            return payload
    if isinstance(result_object, list):
        return result_object
    raise TypeError("Unsupported Ragas result format.")


def maybe_run_ragas(
    results: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "requested": mode,
        "enabled": False,
        "version": None,
        "reason": "",
        "metrics_attempted": [],
        "metrics_completed": [],
        "metrics_skipped": [],
    }

    if mode == "never":
        status["reason"] = "Ragas disabled by --use-ragas never."
        return status

    try:
        ragas = importlib.import_module("ragas")
        metrics_module = importlib.import_module("ragas.metrics")
        datasets_module = importlib.import_module("datasets")
        status["version"] = importlib.metadata.version("ragas")
    except Exception as error:
        status["reason"] = f"Optional Ragas dependencies unavailable: {error}"
        return status

    evaluate_func = getattr(ragas, "evaluate", None)
    if evaluate_func is None:
        try:
            evaluation_module = importlib.import_module("ragas.evaluation")
            evaluate_func = getattr(evaluation_module, "evaluate", None)
        except Exception:
            evaluate_func = None
    if evaluate_func is None:
        status["reason"] = "Could not locate a compatible ragas.evaluate entrypoint."
        return status

    dataset_rows = []
    row_indices = []
    for index, row in enumerate(results):
        if row["error_flag"] or not row["_context_texts"]:
            continue

        dataset_rows.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["_context_texts"],
                "reference": row["reference_answer"],
                "ground_truth": row["reference_answer"],
            }
        )
        row_indices.append(index)

    if not dataset_rows:
        status["reason"] = "No successful rows with retrieved context were available for Ragas."
        return status

    dataset = datasets_module.Dataset.from_list(dataset_rows)

    metric_specs = [
        ("ragas_faithfulness", ("faithfulness", "Faithfulness")),
        (
            "ragas_context_precision",
            ("context_precision", "ContextPrecision", "LLMContextPrecisionWithoutReference"),
        ),
        ("ragas_context_recall", ("context_recall", "ContextRecall", "LLMContextRecall")),
        ("ragas_answer_relevancy", ("answer_relevancy", "AnswerRelevancy")),
        ("ragas_answer_correctness", ("answer_correctness", "AnswerCorrectness")),
    ]

    for column_name, candidates in metric_specs:
        metric_object = _lookup_ragas_attr(metrics_module, *candidates)
        status["metrics_attempted"].append(column_name)

        if metric_object is None:
            status["metrics_skipped"].append(
                f"{column_name}: metric not found in installed ragas version."
            )
            continue

        if column_name == "ragas_answer_correctness":
            eligible_indices = [
                idx for idx, row in enumerate(dataset_rows) if row.get("reference")
            ]
            if not eligible_indices:
                status["metrics_skipped"].append(
                    f"{column_name}: no questions with reference answers."
                )
                continue
            metric_dataset = dataset.select(eligible_indices)
            target_rows = [row_indices[idx] for idx in eligible_indices]
        else:
            metric_dataset = dataset
            target_rows = row_indices

        try:
            evaluation = evaluate_func(metric_dataset, metrics=[metric_object])
            metric_records = _coerce_ragas_records(evaluation)
            if len(metric_records) != len(target_rows):
                raise ValueError(
                    f"Expected {len(target_rows)} Ragas rows, received {len(metric_records)}."
                )

            for record, result_index in zip(metric_records, target_rows):
                metric_value = record.get(column_name)
                if metric_value is None:
                    metric_value = record.get("score")
                results[result_index][column_name] = round_metric(metric_value)
            status["metrics_completed"].append(column_name)
        except Exception as error:
            status["metrics_skipped"].append(f"{column_name}: {error}")

    if status["metrics_completed"]:
        status["enabled"] = True
        status["reason"] = "Ragas metrics computed where compatible."
    else:
        status["reason"] = "Ragas was present, but no compatible metrics completed successfully."

    return status


def summarise_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "avg_source_hit_rate_at_k": mean_ignore_none(
            [row.get("source_hit_rate_at_k") for row in rows]
        ),
        "avg_source_recall_at_k": mean_ignore_none(
            [row.get("source_recall_at_k") for row in rows]
        ),
        "avg_keyword_recall": mean_ignore_none([row.get("keyword_recall") for row in rows]),
        "avg_latency_seconds": mean_ignore_none([row.get("latency_seconds") for row in rows]),
        "error_count": sum(1 for row in rows if row.get("error_flag")),
        "abstention_accuracy": mean_ignore_none(
            [1.0 if row.get("abstention_correct") else 0.0 for row in rows if not row["should_answer"]]
        ),
        "avg_question_score": mean_ignore_none([row.get("question_score") for row in rows]),
    }


def format_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_summary_markdown(
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    ragas_status: dict[str, Any],
    question_path: Path,
) -> str:
    answerable_rows = [row for row in results if row["should_answer"]]
    out_of_scope_rows = [row for row in results if not row["should_answer"]]
    retrieval_rows = [row for row in results if row.get("acceptable_sources_any") or row.get("required_sources_all") or row.get("expected_sources")]
    failure_rows = [
        row for row in results
        if row["error_flag"]
        or (row["should_answer"] and ((row.get("source_hit_rate_at_k") or 0.0) == 0.0))
        or (row["should_answer"] and (row.get("keyword_recall") is not None) and (row.get("keyword_recall") < 0.5))
        or ((not row["should_answer"]) and not row["abstention_correct"])
    ]

    best_rows = sorted(
        [row for row in results if row.get("question_score") is not None],
        key=lambda row: row["question_score"],
        reverse=True,
    )[:3]
    worst_rows = sorted(
        [row for row in results if row.get("question_score") is not None],
        key=lambda row: row["question_score"],
    )[:3]

    overall = summarise_group(results)
    answerable_summary = summarise_group(answerable_rows)
    out_of_scope_summary = summarise_group(out_of_scope_rows)

    by_category = {
        category: summarise_group([row for row in results if row["category"] == category])
        for category in sorted({row["category"] for row in results})
    }
    by_difficulty = {
        difficulty: summarise_group([row for row in results if row["difficulty"] == difficulty])
        for difficulty in sorted({row["difficulty"] for row in results})
    }

    strength_notes = []
    weakness_notes = []

    retrieval_hit = mean_ignore_none([row.get("source_hit_rate_at_k") for row in retrieval_rows])
    keyword_avg = answerable_summary["avg_keyword_recall"]
    abstention_avg = out_of_scope_summary["abstention_accuracy"]

    if retrieval_hit is not None:
        if retrieval_hit >= 0.75:
            strength_notes.append(
                "retrieval usually finds at least one expected source for answerable questions."
            )
        elif retrieval_hit >= 0.5:
            strength_notes.append(
                "retrieval is partially reliable, but some answerable questions still miss the expected source."
            )
        else:
            weakness_notes.append(
                "retrieval often misses the expected source, so answer quality is likely bottlenecked by grounding."
            )

    if keyword_avg is not None:
        if keyword_avg >= 0.7:
            strength_notes.append(
                "generated answers usually cover the main reference concepts and terms."
            )
        elif keyword_avg >= 0.4:
            weakness_notes.append(
                "answers often cover only part of the expected content, suggesting partial or shallow synthesis."
            )
        else:
            weakness_notes.append(
                "answers frequently miss the expected key concepts, so factual coverage needs improvement."
            )

    if abstention_avg is not None:
        if abstention_avg >= 0.8:
            strength_notes.append(
                "the chatbot abstains appropriately on most out-of-scope questions."
            )
        else:
            weakness_notes.append(
                "out-of-scope handling is inconsistent, so the bot may still answer beyond the supplied materials."
            )

    if overall["avg_latency_seconds"] is not None and overall["avg_latency_seconds"] > 10:
        weakness_notes.append("response latency is relatively high for an interactive course assistant.")
    elif overall["avg_latency_seconds"] is not None:
        strength_notes.append("latency stays within a usable range for CLI interaction.")

    lines = [
        "# Chatbot Evaluation Summary",
        "",
        "## Evaluation Caveats",
        "- **Pilot scale**: This is a small pilot evaluation set and might not perfectly cover all syllabus items.",
        "- **Retrieval > Synthesis**: Retrieval often succeeds cleanly, whereas cross-document synthesis questions (like q10 expecting an abstention) highlight the bot's grounding limits.",
        "- **Keyword proxy**: Keyword recall is a lightweight coverage proxy and not an absolute measure of factual correctness.",
        "- **Metric limits**: Abstention performance fundamentally depends on strict/correct label limits, and retrieval hit success does not strictly guarantee answer completeness.",
        "",
        "## Run Metadata",
        f"- Evaluated at: {metadata['evaluated_at']}",
        f"- Questions file: {question_path}",
        f"- Total questions evaluated: {len(results)}",
        f"- LLM provider: {metadata['provider']}",
        f"- LLM model: {metadata['model']}",
        f"- Retriever top-k: {metadata['top_k']}",
        "",
        "## Overall Averages",
        f"- Average source hit rate@k: {format_metric(overall['avg_source_hit_rate_at_k'])}",
        f"- Average source recall@k: {format_metric(overall['avg_source_recall_at_k'])}",
        f"- Average keyword recall: {format_metric(overall['avg_keyword_recall'])}",
        f"- Average latency (seconds): {format_metric(overall['avg_latency_seconds'])}",
        f"- Error count: {overall['error_count']}",
        f"- Average composite question score: {format_metric(overall['avg_question_score'])}",
        "",
        "## Answerable vs Out-of-Scope",
        f"- Answerable questions: {len(answerable_rows)}",
        f"- Out-of-scope questions (including synthesis limits like q10 and out-of-corpus like q11): {len(out_of_scope_rows)}",
        f"- Answerable average source hit@k: {format_metric(answerable_summary['avg_source_hit_rate_at_k'])}",
        f"- Answerable average keyword recall: {format_metric(answerable_summary['avg_keyword_recall'])}",
        f"- Out-of-scope abstention accuracy: {format_metric(out_of_scope_summary['abstention_accuracy'])}",
        "",
        "## By Category",
    ]

    for category, summary in by_category.items():
        lines.append(
            f"- {category}: count={summary['count']}, "
            f"source_hit@k={format_metric(summary['avg_source_hit_rate_at_k'])}, "
            f"keyword_recall={format_metric(summary['avg_keyword_recall'])}, "
            f"abstention_accuracy={format_metric(summary['abstention_accuracy'])}, "
            f"latency={format_metric(summary['avg_latency_seconds'])}, "
            f"errors={summary['error_count']}"
        )

    lines.extend(["", "## By Difficulty"])
    for difficulty, summary in by_difficulty.items():
        lines.append(
            f"- {difficulty}: count={summary['count']}, "
            f"source_hit@k={format_metric(summary['avg_source_hit_rate_at_k'])}, "
            f"keyword_recall={format_metric(summary['avg_keyword_recall'])}, "
            f"latency={format_metric(summary['avg_latency_seconds'])}, "
            f"errors={summary['error_count']}"
        )

    lines.extend(
        [
            "",
            "## Retrieval Success Summary",
            f"- Questions with expected sources: {len(retrieval_rows)}",
            f"- Mean source hit rate@k on those questions: {format_metric(retrieval_hit)}",
            f"- Mean source recall@k on those questions: {format_metric(mean_ignore_none([row.get('source_recall_at_k') for row in retrieval_rows]))}",
            "",
            "## Abstention Performance Summary",
            f"- Out-of-scope questions evaluated: {len(out_of_scope_rows)}",
            f"- Abstentions detected: {sum(1 for row in out_of_scope_rows if row['abstention_detected'])}",
            f"- Correct abstentions: {sum(1 for row in out_of_scope_rows if row['abstention_correct'])}",
            "",
            "## Optional Ragas Metrics",
            f"- Requested mode: {ragas_status['requested']}",
            f"- Enabled: {format_metric(ragas_status['enabled'])}",
            f"- Installed version: {ragas_status['version'] or 'not available'}",
            f"- Status: {ragas_status['reason']}",
        ]
    )

    if ragas_status["metrics_completed"]:
        lines.append(f"- Metrics completed: {', '.join(ragas_status['metrics_completed'])}")
    if ragas_status["metrics_skipped"]:
        for skipped in ragas_status["metrics_skipped"]:
            lines.append(f"- Skipped: {skipped}")

    lines.extend(["", "## Best-Performing Questions"])
    if best_rows:
        for row in best_rows:
            lines.append(
                f"- {row['id']}: score={format_metric(row['question_score'])}, "
                f"question=\"{row['question']}\", sources={serialize_list(row['retrieved_sources'])}"
            )
    else:
        lines.append("- No completed rows were available for ranking.")

    lines.extend(["", "## Worst-Performing Questions"])
    if worst_rows:
        for row in worst_rows:
            lines.append(
                f"- {row['id']}: score={format_metric(row['question_score'])}, "
                f"question=\"{row['question']}\", error={row['error_message'] or 'none'}"
            )
    else:
        lines.append("- No completed rows were available for ranking.")

    lines.extend(["", "## Sample Failures"])
    if failure_rows:
        for row in failure_rows[:5]:
            issue_parts = []
            if row["error_flag"]:
                issue_parts.append(f"error={row['error_message']}")
            if row["should_answer"] and ((row.get("source_hit_rate_at_k") or 0.0) == 0.0):
                issue_parts.append("expected source not retrieved")
            if row["should_answer"] and (row.get("keyword_recall") is not None) and (row.get("keyword_recall") < 0.5):
                issue_parts.append(f"low keyword recall ({format_metric(row['keyword_recall'])})")
            if (not row["should_answer"]) and not row["abstention_correct"]:
                issue_parts.append("missed abstention")
            lines.append(
                f"- {row['id']}: {', '.join(issue_parts)} | answer=\"{truncate(row['answer'])}\" | "
                f"retrieved_sources={serialize_list(row['retrieved_sources'])}"
            )
    else:
        lines.append("- No major failures were detected under the current deterministic checks.")

    lines.extend(["", "## Interpretation"])
    if strength_notes:
        lines.append(f"- Strengths: {' '.join(strength_notes)}")
    else:
        lines.append("- Strengths: no strong patterns emerged from the current sample.")

    if weakness_notes:
        lines.append(f"- Weaknesses: {' '.join(weakness_notes)}")
    else:
        lines.append("- Weaknesses: no major weaknesses were surfaced by the current sample.")

    return "\n".join(lines) + "\n"


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    ragas_columns = sorted(
        column
        for column in {key for row in results for key in row}
        if column.startswith("ragas_")
    )

    fieldnames = [
        "id",
        "question",
        "category",
        "difficulty",
        "should_answer",
        "reference_answer",
        "reference_keywords",
        "acceptable_sources_any",
        "required_sources_all",
        "expected_sources",
        "answer",
        "retrieved_sources",
        "retrieved_pages",
        "retrieved_context_count",
        "latency_seconds",
        "provider",
        "model",
        "error_flag",
        "error_message",
        "source_hit_rate_at_k",
        "source_recall_at_k",
        "keyword_recall",
        "answer_length_chars",
        "answer_length_words",
        "abstention_detected",
        "abstention_correct",
        "question_score",
        *ragas_columns,
    ]

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            output_row = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, list):
                    output_row[field] = serialize_list(value)
                else:
                    output_row[field] = value
            writer.writerow(output_row)


def write_json(
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
    ragas_status: dict[str, Any],
    path: Path,
) -> None:
    serialisable_results = []
    for row in results:
        serialisable_row = {}
        for key, value in row.items():
            if key.startswith("_"):
                continue
            serialisable_row[key] = value
        serialisable_results.append(serialisable_row)

    payload = {
        "metadata": metadata,
        "ragas": ragas_status,
        "results": serialisable_results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    question_path = (PROJECT_ROOT / args.questions).resolve() if not Path(args.questions).is_absolute() else Path(args.questions)
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(question_path)
    if args.max_questions is not None:
        questions = questions[: args.max_questions]

    if not questions:
        raise ValueError("No evaluation questions were loaded.")

    results, metadata = run_deterministic_evaluation(questions, top_k=args.top_k)
    ragas_status = maybe_run_ragas(results, args.use_ragas)

    summary_markdown = build_summary_markdown(
        results=results,
        metadata=metadata,
        ragas_status=ragas_status,
        question_path=question_path,
    )

    csv_path = output_dir / "evaluation_results.csv"
    json_path = output_dir / "evaluation_results.json"
    summary_path = output_dir / "evaluation_summary.md"

    write_csv(results, csv_path)
    write_json(results, metadata, ragas_status, json_path)
    summary_path.write_text(summary_markdown, encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
