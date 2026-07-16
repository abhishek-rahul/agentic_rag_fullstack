from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from evals.shared import BACKEND_DIR, CORE_EVAL_CASE_COUNT


EXPECTED_CASE_IDS = {"wfh-001", "wfh-005", "refund-001", "refund-002"}
LOCAL_PROMPTFOO_PROVIDER = "ollama-qwen2.5-0.5b"
FULL_PROMPTFOO_PROVIDERS = {
    LOCAL_PROMPTFOO_PROVIDER,
    "openai-gpt-4o-mini",
}
EXPECTED_PROMPT_VARIANTS = {"production", "strict_grounded"}


class EvalGateConfig(BaseModel):
    mode: Literal["local", "full"] = "local"
    report_max_age_hours: float = Field(default=24, gt=0)
    minimum_deepeval_score: float = Field(default=0.6, ge=0, le=1)
    minimum_ragas_faithfulness: float = Field(default=0.6, ge=0, le=1)
    minimum_ragas_context_precision: float = Field(default=0.6, ge=0, le=1)
    minimum_promptfoo_pass_rate: float = Field(default=0.75, ge=0, le=1)
    expected_case_count: int = Field(default=CORE_EVAL_CASE_COUNT, ge=1)
    expected_generation_provider: str = "ollama"
    expected_generation_model: str = "qwen2.5:0.5b"
    expected_judge_model: str = "gpt-4o-mini"


class GateCheckResult(BaseModel):
    name: str
    passed: bool
    required: bool
    details: str
    score: float | None = None


class EvalGateReport(BaseModel):
    schema_version: int = 1
    generated_at: str
    mode: Literal["local", "full"]
    ready: bool
    checks: list[GateCheckResult]


@dataclass(frozen=True)
class GatePaths:
    shared_records: Path = BACKEND_DIR / "eval_results" / "shared" / "core_rag_records.json"
    deepeval_report: Path = BACKEND_DIR / "eval_results" / "deepeval" / "latest.json"
    ragas_report: Path = BACKEND_DIR / "eval_results" / "ragas" / "latest.json"
    promptfoo_report: Path = BACKEND_DIR / "eval_results" / "promptfoo" / "latest.json"
    output_json: Path = BACKEND_DIR / "eval_results" / "gate" / "latest.json"
    output_markdown: Path = BACKEND_DIR / "eval_results" / "gate" / "latest.md"


CommandRunner = Callable[[list[str]], Any]


def config_from_environment(mode: str) -> EvalGateConfig:
    load_dotenv(BACKEND_DIR / ".env")
    return EvalGateConfig(
        mode=mode,
        report_max_age_hours=os.getenv("EVAL_REPORT_MAX_AGE_HOURS", "24"),
        minimum_deepeval_score=os.getenv("DEEPEVAL_METRIC_THRESHOLD", "0.6"),
        minimum_ragas_faithfulness=os.getenv(
            "EVAL_MIN_RAGAS_FAITHFULNESS", "0.6"
        ),
        minimum_ragas_context_precision=os.getenv(
            "EVAL_MIN_RAGAS_CONTEXT_PRECISION", "0.6"
        ),
        minimum_promptfoo_pass_rate=os.getenv(
            "EVAL_MIN_PROMPTFOO_PASS_RATE", "0.75"
        ),
        expected_generation_provider=os.getenv(
            "EVAL_GENERATION_PROVIDER", "ollama"
        ),
        expected_generation_model=os.getenv(
            "EVAL_GENERATION_MODEL", "qwen2.5:0.5b"
        ),
        expected_judge_model=os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini"),
    )


def default_command_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        check=False,
    )


def _pytest_check(
    name: str,
    test_target: str,
    required: bool,
    command_runner: CommandRunner,
) -> GateCheckResult:
    result = command_runner([
        sys.executable,
        "-m",
        "pytest",
        test_target,
        "-q",
        "-p",
        "no:cacheprovider",
    ])
    passed = result.returncode == 0
    output = (result.stdout or result.stderr or "").strip().splitlines()
    details = output[-1][:500] if output else f"pytest exit code {result.returncode}"
    return GateCheckResult(
        name=name,
        passed=passed,
        required=required,
        details=details,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as report_file:
        value = json.load(report_file)
    if not isinstance(value, dict):
        raise ValueError("report root must be a JSON object")
    return value


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("generated timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("generated timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _require_fresh(timestamp: Any, config: EvalGateConfig, now: datetime) -> None:
    generated_at = _parse_timestamp(timestamp)
    if now - generated_at > timedelta(hours=config.report_max_age_hours):
        raise ValueError(
            f"report is older than {config.report_max_age_hours:g} hours"
        )


def _metadata_errors(
    report: dict[str, Any],
    config: EvalGateConfig,
    shared_envelope: dict[str, Any] | None = None,
) -> list[str]:
    errors = []
    expected_timestamp = (
        shared_envelope.get("generated_at") if shared_envelope else None
    )
    expected = {
        "generation_provider": config.expected_generation_provider,
        "generation_model": config.expected_generation_model,
        "prompt_variant": "production",
    }
    for field, value in expected.items():
        if report.get(field) != value:
            errors.append(f"{field} must be {value!r}")
    if expected_timestamp is not None and report.get("source_records_generated_at") != expected_timestamp:
        errors.append("source record timestamp does not match shared records")
    if report.get("case_count") != config.expected_case_count:
        errors.append(f"case_count must be {config.expected_case_count}")
    return errors


def _shared_records_check(
    config: EvalGateConfig,
    paths: GatePaths,
    now: datetime,
) -> tuple[GateCheckResult, dict[str, Any] | None]:
    try:
        report = _load_json(paths.shared_records)
        _require_fresh(report.get("generated_at"), config, now)
        if report.get("schema_version") != 1:
            raise ValueError("unsupported shared-record schema version")
        if report.get("generation_provider") != config.expected_generation_provider:
            raise ValueError("generation provider does not match gate configuration")
        if report.get("generation_model") != config.expected_generation_model:
            raise ValueError("generation model does not match gate configuration")
        if report.get("prompt_variant") != "production":
            raise ValueError("shared records must use the production prompt")

        records = report.get("records")
        if not isinstance(records, list) or len(records) != config.expected_case_count:
            raise ValueError(f"expected {config.expected_case_count} shared records")
        if {record.get("id") for record in records} != EXPECTED_CASE_IDS:
            raise ValueError("shared record case IDs do not match the core dataset")
        for record in records:
            if record.get("generation_error") is not None:
                raise ValueError(f"{record.get('id')}: generation failed")
            if not record.get("guardrail", {}).get("passed"):
                raise ValueError(f"{record.get('id')}: guardrail did not pass")
            if not record.get("raw_answer"):
                raise ValueError(f"{record.get('id')}: raw answer is missing")
            if not record.get("contexts"):
                raise ValueError(f"{record.get('id')}: contexts are missing")
        return (
            GateCheckResult(
                name="Shared evaluation records",
                passed=True,
                required=True,
                details=f"{len(records)} fresh validated records",
            ),
            report,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return (
            GateCheckResult(
                name="Shared evaluation records",
                passed=False,
                required=True,
                details=str(exc)[:500],
            ),
            None,
        )


def _deepeval_check(
    required: bool,
    config: EvalGateConfig,
    paths: GatePaths,
    now: datetime,
    shared: dict[str, Any] | None,
) -> GateCheckResult:
    if not required:
        return GateCheckResult(
            name="DeepEval",
            passed=False,
            required=False,
            details="not required in local mode",
        )
    try:
        report = _load_json(paths.deepeval_report)
        _require_fresh(report.get("generated_at"), config, now)
        errors = _metadata_errors(report, config, shared)
        if report.get("judge_model") != config.expected_judge_model:
            errors.append("judge model does not match gate configuration")
        records = report.get("records", [])
        if {row.get("id") for row in records} != EXPECTED_CASE_IDS:
            errors.append("DeepEval case IDs do not match the core dataset")
        for row in records:
            if float(row.get("faithfulness", -1)) < config.minimum_deepeval_score:
                errors.append(f"{row.get('id')}: faithfulness below threshold")
            if float(row.get("answer_relevancy", -1)) < config.minimum_deepeval_score:
                errors.append(f"{row.get('id')}: answer relevancy below threshold")
        if report.get("passed") is not True:
            errors.append("DeepEval report is not passed")
        if errors:
            raise ValueError("; ".join(errors))
        minimum = min(
            min(float(row["faithfulness"]), float(row["answer_relevancy"]))
            for row in records
        )
        return GateCheckResult(
            name="DeepEval",
            passed=True,
            required=True,
            details="all four cases passed faithfulness and answer relevancy",
            score=minimum,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return GateCheckResult(
            name="DeepEval",
            passed=False,
            required=True,
            details=str(exc)[:500],
        )


def _ragas_check(
    required: bool,
    config: EvalGateConfig,
    paths: GatePaths,
    now: datetime,
    shared: dict[str, Any] | None,
) -> GateCheckResult:
    if not required:
        return GateCheckResult(
            name="Ragas",
            passed=False,
            required=False,
            details="not required in local mode",
        )
    try:
        report = _load_json(paths.ragas_report)
        _require_fresh(report.get("generated_at"), config, now)
        errors = _metadata_errors(report, config, shared)
        if report.get("judge_model") != config.expected_judge_model:
            errors.append("judge model does not match gate configuration")
        if {row.get("id") for row in report.get("records", [])} != EXPECTED_CASE_IDS:
            errors.append("Ragas case IDs do not match the core dataset")
        averages = report.get("averages", {})
        faithfulness = float(averages.get("faithfulness", -1))
        precision = float(averages.get("context_precision", -1))
        if faithfulness < config.minimum_ragas_faithfulness:
            errors.append("average faithfulness is below threshold")
        if precision < config.minimum_ragas_context_precision:
            errors.append("average context precision is below threshold")
        if errors:
            raise ValueError("; ".join(errors))
        return GateCheckResult(
            name="Ragas",
            passed=True,
            required=True,
            details=(
                f"faithfulness={faithfulness:.3f}, "
                f"context_precision={precision:.3f}"
            ),
            score=min(faithfulness, precision),
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return GateCheckResult(
            name="Ragas",
            passed=False,
            required=True,
            details=str(exc)[:500],
        )


def _prompt_variant(row: dict[str, Any]) -> str | None:
    label = str(row.get("prompt", {}).get("label", "")).replace("\\", "/")
    if "strict_grounded.txt" in label:
        return "strict_grounded"
    if "production.txt" in label:
        return "production"
    return None


def _promptfoo_check(
    required: bool,
    config: EvalGateConfig,
    paths: GatePaths,
    now: datetime,
) -> GateCheckResult:
    try:
        report = _load_json(paths.promptfoo_report)
        result_block = report.get("results", {})
        _require_fresh(result_block.get("timestamp"), config, now)
        rows = result_block.get("results")
        if not isinstance(rows, list):
            raise ValueError("Promptfoo result rows are missing")

        if required:
            selected = rows
            expected_providers = FULL_PROMPTFOO_PROVIDERS
            expected_count = config.expected_case_count * 2 * 2
        else:
            selected = [
                row
                for row in rows
                if row.get("provider", {}).get("label") == LOCAL_PROMPTFOO_PROVIDER
            ]
            expected_providers = {LOCAL_PROMPTFOO_PROVIDER}
            expected_count = config.expected_case_count * 2

        providers = {row.get("provider", {}).get("label") for row in selected}
        case_ids = {row.get("vars", {}).get("case_id") for row in selected}
        prompt_variants = {_prompt_variant(row) for row in selected}
        if len(selected) != expected_count:
            raise ValueError(f"expected {expected_count} Promptfoo results")
        if providers != expected_providers:
            raise ValueError("Promptfoo providers do not match the expected matrix")
        if case_ids != EXPECTED_CASE_IDS:
            raise ValueError("Promptfoo case IDs do not match the core dataset")
        if prompt_variants != EXPECTED_PROMPT_VARIANTS:
            raise ValueError("Promptfoo prompt variants do not match")

        successes = sum(row.get("success") is True for row in selected)
        pass_rate = successes / len(selected)
        passed = pass_rate >= config.minimum_promptfoo_pass_rate
        return GateCheckResult(
            name="Promptfoo full matrix" if required else "Promptfoo Ollama matrix",
            passed=passed,
            required=required,
            details=(
                f"pass rate {pass_rate:.1%} "
                f"({successes}/{len(selected)}), required "
                f"{config.minimum_promptfoo_pass_rate:.1%}"
            ),
            score=pass_rate,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return GateCheckResult(
            name="Promptfoo full matrix" if required else "Promptfoo Ollama matrix",
            passed=False,
            required=required,
            details=str(exc)[:500],
        )


def evaluate_gate(
    config: EvalGateConfig,
    paths: GatePaths | None = None,
    command_runner: CommandRunner = default_command_runner,
    now: datetime | None = None,
) -> EvalGateReport:
    paths = paths or GatePaths()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    checks = [
        _pytest_check(
            "Guardrail tests", "tests/test_guardrails.py", True, command_runner
        ),
        _pytest_check(
            "Retrieval tests", "tests/test_retrieval.py", True, command_runner
        ),
        _pytest_check(
            "Tool guardrail tests",
            "tests/test_tool_guardrails.py",
            True,
            command_runner,
        ),
    ]
    shared_check, shared = _shared_records_check(config, paths, now)
    checks.append(shared_check)

    full = config.mode == "full"
    if full:
        checks.append(
            _pytest_check(
                "Live keyword tests",
                "tests/test_answer_keywords.py",
                True,
                command_runner,
            )
        )
    else:
        checks.append(GateCheckResult(
            name="Live keyword tests",
            passed=False,
            required=False,
            details="not required in local mode",
        ))

    checks.extend([
        _deepeval_check(full, config, paths, now, shared),
        _ragas_check(full, config, paths, now, shared),
        _promptfoo_check(full, config, paths, now),
    ])
    ready = all(check.passed for check in checks if check.required)
    return EvalGateReport(
        generated_at=now.isoformat(),
        mode=config.mode,
        ready=ready,
        checks=checks,
    )


def _markdown(report: EvalGateReport) -> str:
    lines = [
        f"# {report.mode.upper()} evaluation gate",
        "",
        f"Generated: {report.generated_at}",
        "",
        "| Status | Check | Details |",
        "|---|---|---|",
    ]
    for check in report.checks:
        status = "PASS" if check.passed else ("FAIL" if check.required else "WARN")
        details = check.details.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {status} | {check.name} | {details} |")
    lines.extend([
        "",
        (
            "## READY FOR NEXT LOCAL STAGE"
            if report.ready
            else "## NOT READY - FIX FAILED CHECKS"
        ),
        "",
    ])
    return "\n".join(lines)


def write_gate_report(report: EvalGateReport, paths: GatePaths | None = None) -> None:
    paths = paths or GatePaths()
    paths.output_json.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        paths.output_json: json.dumps(
            report.model_dump(mode="json"), indent=2, ensure_ascii=False
        ) + "\n",
        paths.output_markdown: _markdown(report),
    }
    for path, content in outputs.items():
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        temp_path.replace(path)


def print_report(report: EvalGateReport) -> None:
    print(f"{report.mode.upper()} EVAL GATE\n")
    for check in report.checks:
        status = "PASS" if check.passed else ("FAIL" if check.required else "WARN")
        print(f"{status:<5} {check.name}: {check.details}")
    print()
    print(
        "READY FOR NEXT LOCAL STAGE"
        if report.ready
        else "NOT READY - FIX FAILED CHECKS"
    )


def main(
    argv: list[str] | None = None,
    *,
    paths: GatePaths | None = None,
    command_runner: CommandRunner = default_command_runner,
    now: datetime | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Apply the local evaluation gate")
    parser.add_argument("--mode", choices=("local", "full"), default="local")
    try:
        args = parser.parse_args(argv)
        config = config_from_environment(args.mode)
        report = evaluate_gate(config, paths, command_runner, now)
        write_gate_report(report, paths)
        print_report(report)
        return 0 if report.ready else 1
    except (ValidationError, OSError, ValueError, TypeError) as exc:
        print(f"Evaluation gate configuration or execution error: {str(exc)[:500]}")
        return 2
    except Exception as exc:
        print(f"Evaluation gate execution error: {str(exc)[:500]}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
