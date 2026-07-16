import argparse
import csv
import json
import os
from datetime import datetime, timezone

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision, Faithfulness

from evals.shared import BACKEND_DIR, load_record_envelope


OUTPUT_DIR = BACKEND_DIR / "eval_results" / "ragas"


def _validate_record(record: dict) -> None:
    if record.get("generation_error"):
        raise ValueError(f"{record['id']}: generation failed")
    if not record.get("guardrail", {}).get("passed"):
        failure_type = record.get("guardrail", {}).get("failure_type") or "unknown"
        raise ValueError(f"{record['id']}: guardrail failed ({failure_type})")
    if not record.get("raw_answer"):
        raise ValueError(f"{record['id']}: raw answer is missing")
    if not record.get("contexts"):
        raise ValueError(f"{record['id']}: retrieved contexts are missing")
    if not record.get("reference_answer"):
        raise ValueError(f"{record['id']}: reference answer is missing")


def _score_value(result) -> float:
    value = getattr(result, "value", result)
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score frozen RAG records with Ragas")
    parser.add_argument("--case", dest="case_id", help="Score one case for smoke tests")
    args = parser.parse_args()

    load_dotenv(BACKEND_DIR / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the Ragas judge")

    envelope = load_record_envelope()
    records = envelope["records"]
    if args.case_id:
        records = [record for record in records if record["id"] == args.case_id]
        if not records:
            raise SystemExit(f"Unknown evaluation case: {args.case_id}")

    for record in records:
        _validate_record(record)

    judge_model = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
    judge = llm_factory(judge_model, client=AsyncOpenAI())
    faithfulness = Faithfulness(llm=judge)
    context_precision = ContextPrecision(llm=judge)

    rows = []
    for record in records:
        faithfulness_result = faithfulness.score(
            user_input=record["question"],
            response=record["raw_answer"],
            retrieved_contexts=record["contexts"],
        )
        precision_result = context_precision.score(
            user_input=record["question"],
            reference=record["reference_answer"],
            retrieved_contexts=record["contexts"],
        )
        rows.append({
            "id": record["id"],
            "faithfulness": _score_value(faithfulness_result),
            "context_precision": _score_value(precision_result),
        })

    averages = {
        "faithfulness": sum(row["faithfulness"] for row in rows) / len(rows),
        "context_precision": (
            sum(row["context_precision"] for row in rows) / len(rows)
        ),
    }
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge_model,
        "source_records_generated_at": envelope["generated_at"],
        "generation_provider": envelope["generation_provider"],
        "generation_model": envelope["generation_model"],
        "prompt_variant": envelope["prompt_variant"],
        "case_count": len(rows),
        "records": rows,
        "averages": averages,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "latest.json").open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    with (OUTPUT_DIR / "latest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["id", "faithfulness", "context_precision"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
