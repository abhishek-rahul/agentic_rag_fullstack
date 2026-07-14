from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.chat_service import ChatService
from app.services.graph_service import DEFAULT_SYSTEM_PROMPT_TEMPLATE, GraphService
from app.services.guardrail_service import GuardrailService
from evals.shared import run_application_case, sanitize_generation_error


TEST_CASE = {
    "id": "test-case",
    "question": "What is the policy?",
    "reference_answer": "The policy is available.",
    "expected_keywords": ["policy"],
    "expected_source": "policy.txt",
}


def graph_result(answer: str, generation_error: str | None = None) -> dict:
    return {
        "answer": answer,
        "context": "Relevant policy context.",
        "contexts": ["Relevant policy context."],
        "sources": [{"source": "C:/docs/policy.txt", "content_preview": "Policy"}],
        "best_retrieval_score": 0.2,
        "generation_error": generation_error,
        "llm_response": {
            "answer": answer,
            "grounded_in_context": True,
        } if not generation_error else None,
    }


class StaticGraphService:
    def __init__(self, result: dict):
        self.result = result

    def run(self, **kwargs) -> dict:
        return deepcopy(self.result)


def application_service(result: dict) -> ChatService:
    service = ChatService.__new__(ChatService)
    service.settings = SimpleNamespace(guardrail_enabled=True)
    service.graph_service = StaticGraphService(result)
    service.guardrail_service = GuardrailService()
    return service


def test_output_guardrail_returns_copy_without_mutating_graph_result() -> None:
    original = graph_result("The employee email is learner@example.com.")
    snapshot = deepcopy(original)
    service = application_service(original)

    safe_result, guardrail = service._apply_output_guardrail(original)

    assert original == snapshot
    assert guardrail.passed is False
    assert safe_result["llm_response"] is None
    assert "learner@example.com" not in safe_result["answer"]


def test_passed_record_retains_raw_answer_but_not_llm_response() -> None:
    record = run_application_case(
        TEST_CASE,
        provider="ollama",
        model="test-model",
        service=application_service(graph_result("The policy is available.")),
    )
    serialized = record.to_dict()

    assert record.guardrail["passed"] is True
    assert record.raw_answer == "The policy is available."
    assert record.grounded_in_context is True
    assert "llm_response" not in serialized


@pytest.mark.parametrize(
    ("result", "failure_type"),
    [
        (graph_result("Email learner@example.com"), "pii"),
        (graph_result("", generation_error="parser failed"), "schema"),
        ({**graph_result("Answer"), "contexts": [], "context": "", "sources": []}, "retrieval"),
        ({**graph_result(""), "llm_response": None}, "output"),
    ],
)
def test_every_blocked_record_discards_raw_output(
    result: dict,
    failure_type: str,
) -> None:
    record = run_application_case(
        TEST_CASE,
        provider="ollama",
        model="test-model",
        service=application_service(result),
    )
    serialized = record.to_dict()

    assert record.guardrail["passed"] is False
    assert record.guardrail["failure_type"] == failure_type
    assert record.raw_answer is None
    assert record.grounded_in_context is None
    assert "llm_response" not in serialized
    raw_answer = result.get("answer", "")
    if raw_answer:
        assert raw_answer not in record.answer


def test_generation_error_is_redacted_flattened_and_bounded() -> None:
    error = "parser failed for learner@example.com\n" + ("x" * 800)

    sanitized = sanitize_generation_error(error)

    assert sanitized is not None
    assert "learner@example.com" not in sanitized
    assert "[REDACTED]" in sanitized
    assert "\n" not in sanitized
    assert len(sanitized) == 500


def test_default_graph_prompt_is_the_current_production_prompt() -> None:
    graph = GraphService.__new__(GraphService)
    graph.system_prompt_template = DEFAULT_SYSTEM_PROMPT_TEMPLATE

    rendered = graph.system_prompt_template.format(
        context="retrieved context",
        memory_text="conversation memory",
    )

    assert "Use the retrieved company context when it is relevant." in rendered
    assert "retrieved context" in rendered
    assert "conversation memory" in rendered

    prompt_file = (
        Path(__file__).resolve().parents[1]
        / "evals"
        / "promptfoo"
        / "prompts"
        / "production.txt"
    )
    assert prompt_file.read_text(encoding="utf-8").strip() == (
        DEFAULT_SYSTEM_PROMPT_TEMPLATE
    )
