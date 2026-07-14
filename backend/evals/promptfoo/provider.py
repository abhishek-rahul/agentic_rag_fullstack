from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from evals.shared import load_core_eval_cases, run_application_case


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PRODUCTION_PROMPT = (PROMPT_DIR / "production.txt").read_text(encoding="utf-8").strip()


def generate_tests(config: dict | None = None) -> list[dict[str, Any]]:
    tests = []
    for case in load_core_eval_cases():
        tests.append({
            "description": case["id"],
            "vars": {
                "case_id": case["id"],
                "question": case["question"],
            },
            "assert": [
                {"type": "not-equals", "value": ""},
                {
                    "type": "icontains-all",
                    "value": case["expected_keywords"],
                },
            ],
        })
    return tests


def call_api(prompt: str, options: dict, context: dict) -> dict[str, Any]:
    provider_config = options.get("config", {})
    app_provider = provider_config["app_provider"]
    app_model = provider_config["app_model"]
    case_id = context.get("vars", {}).get("case_id")

    cases = {case["id"]: case for case in load_core_eval_cases()}
    if case_id not in cases:
        return {"error": "Unknown shared evaluation case"}

    prompt_template = prompt.strip()
    prompt_variant = (
        "production"
        if prompt_template == PRODUCTION_PROMPT
        else "strict_grounded"
    )

    try:
        record = run_application_case(
            cases[case_id],
            provider=app_provider,
            model=app_model,
            prompt_variant=prompt_variant,
            system_prompt_template=prompt_template,
        )
    except Exception:
        return {"error": "Application evaluation call failed"}

    if record.generation_error or not record.guardrail["passed"]:
        failure_type = record.guardrail.get("failure_type") or "output"
        return {
            "error": f"Application output was safely blocked ({failure_type})",
            "output": record.answer,
            "metadata": {
                "case_id": record.id,
                "guardrail_passed": False,
                "failure_type": failure_type,
            },
        }

    return {
        "output": record.answer,
        "metadata": {
            "case_id": record.id,
            "guardrail_passed": True,
            "grounded_in_context": record.grounded_in_context,
            "source_filenames": record.source_filenames,
        },
    }
