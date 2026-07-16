import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals.gate import (
    EvalGateConfig,
    GatePaths,
    evaluate_gate,
    main,
    write_gate_report,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
CASE_IDS = ["wfh-001", "wfh-005", "refund-001", "refund-002"]


def paths(tmp_path: Path) -> GatePaths:
    return GatePaths(
        shared_records=tmp_path / "shared.json",
        deepeval_report=tmp_path / "deepeval.json",
        ragas_report=tmp_path / "ragas.json",
        promptfoo_report=tmp_path / "promptfoo.json",
        output_json=tmp_path / "gate" / "latest.json",
        output_markdown=tmp_path / "gate" / "latest.md",
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def shared_report(generated_at: datetime = NOW) -> dict:
    return {
        "schema_version": 1,
        "generated_at": generated_at.isoformat(),
        "generation_provider": "ollama",
        "generation_model": "qwen2.5:0.5b",
        "prompt_variant": "production",
        "records": [
            {
                "id": case_id,
                "raw_answer": "safe answer",
                "contexts": ["policy context"],
                "generation_error": None,
                "guardrail": {"passed": True},
            }
            for case_id in CASE_IDS
        ],
    }


def deepeval_report() -> dict:
    return {
        "schema_version": 1,
        "generated_at": NOW.isoformat(),
        "judge_model": "gpt-4o-mini",
        "threshold": 0.6,
        "source_records_generated_at": NOW.isoformat(),
        "generation_provider": "ollama",
        "generation_model": "qwen2.5:0.5b",
        "prompt_variant": "production",
        "case_count": 4,
        "records": [
            {
                "id": case_id,
                "faithfulness": 0.6,
                "answer_relevancy": 0.6,
                "passed": True,
            }
            for case_id in CASE_IDS
        ],
        "passed": True,
    }


def ragas_report() -> dict:
    return {
        "schema_version": 1,
        "generated_at": NOW.isoformat(),
        "judge_model": "gpt-4o-mini",
        "source_records_generated_at": NOW.isoformat(),
        "generation_provider": "ollama",
        "generation_model": "qwen2.5:0.5b",
        "prompt_variant": "production",
        "case_count": 4,
        "records": [{"id": case_id} for case_id in CASE_IDS],
        "averages": {"faithfulness": 0.6, "context_precision": 0.6},
    }


def promptfoo_report(full: bool = True, failed_rows: int = 4) -> dict:
    providers = ["ollama-qwen2.5-0.5b"]
    if full:
        providers.append("openai-gpt-4o-mini")
    rows = []
    for provider in providers:
        for case_id in CASE_IDS:
            for prompt in ("production", "strict_grounded"):
                rows.append({
                    "success": len(rows) >= failed_rows,
                    "provider": {"label": provider},
                    "vars": {"case_id": case_id},
                    "prompt": {"label": f"prompts\\{prompt}.txt: prompt"},
                })
    return {
        "results": {
            "timestamp": NOW.isoformat(),
            "results": rows,
        }
    }


def successful_runner(command: list[str]) -> SimpleNamespace:
    return SimpleNamespace(returncode=0, stdout="10 passed", stderr="")


def write_full_artifacts(gate_paths: GatePaths) -> None:
    write_json(gate_paths.shared_records, shared_report())
    write_json(gate_paths.deepeval_report, deepeval_report())
    write_json(gate_paths.ragas_report, ragas_report())
    write_json(gate_paths.promptfoo_report, promptfoo_report())


def test_local_gate_is_ready_when_required_checks_pass(tmp_path: Path) -> None:
    gate_paths = paths(tmp_path)
    write_json(gate_paths.shared_records, shared_report())

    report = evaluate_gate(
        EvalGateConfig(mode="local"),
        gate_paths,
        successful_runner,
        NOW,
    )

    assert report.ready is True
    promptfoo = next(check for check in report.checks if "Promptfoo" in check.name)
    assert promptfoo.required is False
    assert promptfoo.passed is False


def test_required_pytest_failure_makes_gate_not_ready(tmp_path: Path) -> None:
    gate_paths = paths(tmp_path)
    write_json(gate_paths.shared_records, shared_report())

    def failing_runner(command: list[str]) -> SimpleNamespace:
        failed = command[3] == "tests/test_retrieval.py"
        return SimpleNamespace(
            returncode=1 if failed else 0,
            stdout="failed" if failed else "passed",
            stderr="",
        )

    report = evaluate_gate(
        EvalGateConfig(mode="local"), gate_paths, failing_runner, NOW
    )

    assert report.ready is False
    retrieval = next(check for check in report.checks if check.name == "Retrieval tests")
    assert retrieval.passed is False


def test_stale_shared_records_make_gate_not_ready(tmp_path: Path) -> None:
    gate_paths = paths(tmp_path)
    write_json(
        gate_paths.shared_records,
        shared_report(NOW - timedelta(hours=25)),
    )

    report = evaluate_gate(
        EvalGateConfig(mode="local"), gate_paths, successful_runner, NOW
    )

    assert report.ready is False
    shared = next(
        check for check in report.checks if check.name == "Shared evaluation records"
    )
    assert "older than 24 hours" in shared.details


def test_full_gate_passes_with_scores_exactly_at_threshold(tmp_path: Path) -> None:
    gate_paths = paths(tmp_path)
    write_full_artifacts(gate_paths)

    report = evaluate_gate(
        EvalGateConfig(mode="full"), gate_paths, successful_runner, NOW
    )

    assert report.ready is True
    assert all(check.passed for check in report.checks if check.required)


@pytest.mark.parametrize(
    ("artifact", "mutate", "expected_check"),
    [
        (
            "deepeval",
            lambda report: report["records"][0].update({"faithfulness": 0.59}),
            "DeepEval",
        ),
        (
            "ragas",
            lambda report: report["averages"].update({"context_precision": 0.59}),
            "Ragas",
        ),
        (
            "ragas",
            lambda report: report.update({"generation_model": "wrong-model"}),
            "Ragas",
        ),
    ],
)
def test_full_gate_rejects_low_scores_or_wrong_metadata(
    tmp_path: Path,
    artifact: str,
    mutate,
    expected_check: str,
) -> None:
    gate_paths = paths(tmp_path)
    write_full_artifacts(gate_paths)
    target_path = getattr(gate_paths, f"{artifact}_report")
    value = json.loads(target_path.read_text(encoding="utf-8"))
    mutate(value)
    write_json(target_path, value)

    report = evaluate_gate(
        EvalGateConfig(mode="full"), gate_paths, successful_runner, NOW
    )

    assert report.ready is False
    failed = next(check for check in report.checks if check.name == expected_check)
    assert failed.passed is False


def test_promptfoo_below_threshold_blocks_only_full_mode(tmp_path: Path) -> None:
    gate_paths = paths(tmp_path)
    write_full_artifacts(gate_paths)
    write_json(
        gate_paths.promptfoo_report,
        promptfoo_report(full=True, failed_rows=5),
    )

    full = evaluate_gate(
        EvalGateConfig(mode="full"), gate_paths, successful_runner, NOW
    )
    local = evaluate_gate(
        EvalGateConfig(mode="local"), gate_paths, successful_runner, NOW
    )

    assert full.ready is False
    assert local.ready is True


def test_malformed_required_report_is_a_failed_check(tmp_path: Path) -> None:
    gate_paths = paths(tmp_path)
    gate_paths.shared_records.write_text("not-json", encoding="utf-8")

    report = evaluate_gate(
        EvalGateConfig(mode="local"), gate_paths, successful_runner, NOW
    )

    assert report.ready is False


def test_gate_writes_matching_json_and_markdown(tmp_path: Path) -> None:
    gate_paths = paths(tmp_path)
    write_json(gate_paths.shared_records, shared_report())
    report = evaluate_gate(
        EvalGateConfig(mode="local"), gate_paths, successful_runner, NOW
    )

    write_gate_report(report, gate_paths)

    saved = json.loads(gate_paths.output_json.read_text(encoding="utf-8"))
    markdown = gate_paths.output_markdown.read_text(encoding="utf-8")
    assert saved["ready"] is True
    assert "READY FOR NEXT LOCAL STAGE" in markdown


def test_execution_error_returns_exit_code_two(tmp_path: Path) -> None:
    gate_paths = paths(tmp_path)

    def broken_runner(command: list[str]) -> SimpleNamespace:
        raise OSError("cannot start pytest")

    exit_code = main(
        ["--mode", "local"],
        paths=gate_paths,
        command_runner=broken_runner,
        now=NOW,
    )

    assert exit_code == 2
