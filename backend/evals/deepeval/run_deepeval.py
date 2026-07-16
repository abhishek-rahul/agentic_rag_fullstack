from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from evals.shared import BACKEND_DIR, load_record_envelope


OUTPUT_PATH = BACKEND_DIR / "eval_results" / "deepeval" / "latest.json"


def validate_record(record: dict[str, Any]) -> None:
    if record.get("generation_error"):
        raise ValueError(f"{record['id']}: generation failed")
    if not record.get("guardrail", {}).get("passed"):
        failure_type = record.get("guardrail", {}).get("failure_type") or "unknown"
        raise ValueError(f"{record['id']}: guardrail failed ({failure_type})")
    if not record.get("raw_answer"):
        raise ValueError(f"{record['id']}: raw answer is missing")
    if not record.get("contexts"):
        raise ValueError(f"{record['id']}: retrieved contexts are missing")


def build_test_case(record: dict[str, Any]) -> LLMTestCase:
    validate_record(record)
    return LLMTestCase(
        input=record["question"],
        actual_output=record["raw_answer"],
        retrieval_context=record["contexts"],
    )


def build_metrics(judge_model: str, threshold: float) -> list[Any]:
    return [
        FaithfulnessMetric(
            threshold=threshold,
            model=judge_model,
            include_reason=False,
            async_mode=False,
        ),
        AnswerRelevancyMetric(
            threshold=threshold,
            model=judge_model,
            include_reason=False,
            async_mode=False,
        ),
    ]


def _write_report(report: dict[str, Any], path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temp_file:
        json.dump(report, temp_file, indent=2)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def main() -> int:
    load_dotenv(BACKEND_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is required for the DeepEval judge")
        return 2

    try:
        judge_model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
        threshold = float(os.getenv("DEEPEVAL_METRIC_THRESHOLD", "0.6"))
        if not 0 <= threshold <= 1:
            raise ValueError("DEEPEVAL_METRIC_THRESHOLD must be between 0 and 1")
        envelope = load_record_envelope()
        rows = []
        for record in envelope["records"]:
            test_case = build_test_case(record)
            faithfulness, answer_relevancy = build_metrics(judge_model, threshold)
            faithfulness_score = float(
                faithfulness.measure(
                    test_case,
                    _show_indicator=False,
                    _log_metric_to_confident=False,
                )
            )
            relevancy_score = float(
                answer_relevancy.measure(
                    test_case,
                    _show_indicator=False,
                    _log_metric_to_confident=False,
                )
            )
            rows.append({
                "id": record["id"],
                "faithfulness": faithfulness_score,
                "answer_relevancy": relevancy_score,
                "passed": (
                    faithfulness_score >= threshold
                    and relevancy_score >= threshold
                ),
            })

        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "judge_model": judge_model,
            "threshold": threshold,
            "source_records_generated_at": envelope["generated_at"],
            "generation_provider": envelope["generation_provider"],
            "generation_model": envelope["generation_model"],
            "prompt_variant": envelope["prompt_variant"],
            "case_count": len(rows),
            "records": rows,
            "passed": all(row["passed"] for row in rows),
        }
        _write_report(report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"DeepEval report generation failed: {str(exc)[:500]}")
        return 2
    except Exception as exc:
        print(f"DeepEval judge execution failed: {str(exc)[:500]}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
