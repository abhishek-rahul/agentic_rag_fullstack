from types import SimpleNamespace

import pytest

from app.domain.models import ChatRequest
from app.services.chat_service import ChatService
from app.services.guardrail_service import GuardrailService


def valid_rag_result(answer: str) -> dict:
    return {
        "answer": answer,
        "context": "Relevant company policy context.",
        "sources": [
            {
                "source": "policy.txt",
                "content_preview": "Relevant company policy context.",
            }
        ],
        "best_retrieval_score": 0.25,
        "generation_error": None,
        "llm_response": {
            "answer": answer,
            "grounded_in_context": True,
        },
    }


@pytest.mark.parametrize(
    ("answer", "expected_sensitive_type"),
    [
        ("Contact the employee at learner@example.com.", "email address"),
        ("Call the employee at 9876543210.", "phone number"),
        ("Call the employee at +91 98765 43210.", "phone number"),
        ("The card number is 4111 1111 1111 1111.", "card-like number"),
    ],
)
def test_sensitive_output_is_categorized_without_exposing_value(
    answer: str,
    expected_sensitive_type: str,
) -> None:
    result = GuardrailService().validate_rag_result(valid_rag_result(answer))

    assert result.passed is False
    assert result.failure_type == "pii"
    assert expected_sensitive_type in result.reason
    assert answer not in result.reason


def test_generation_error_is_categorized_as_schema_failure() -> None:
    result_data = valid_rag_result("")
    result_data["generation_error"] = "raw parser failure"

    result = GuardrailService().validate_rag_result(result_data)

    assert result.passed is False
    assert result.failure_type == "schema"
    assert result.reason == "Structured output generation failed"
    assert "raw parser failure" not in result.reason


@pytest.mark.parametrize(
    ("failure_type", "expected_fallback"),
    [
        (
            "pii",
            "I can't provide that response because it may contain sensitive information.",
        ),
        (
            "schema",
            "I couldn't generate a valid response. Please try again.",
        ),
        (
            "retrieval",
            "I don't have enough reliable information in the available documents "
            "to answer this question accurately.",
        ),
    ],
)
def test_failure_type_selects_safe_fallback(
    failure_type: str,
    expected_fallback: str,
) -> None:
    fallback = GuardrailService().fallback_answer("test reason", failure_type)

    assert fallback == expected_fallback


class StaticGraphService:
    def __init__(self, result: dict):
        self.result = result

    def run(self, **kwargs) -> dict:
        return dict(self.result)


class RecordingMemoryService:
    def __init__(self):
        self.messages: list[tuple[str, str, str]] = []

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.messages.append((session_id, role, content))


class RecordingLoggingService:
    def __init__(self):
        self.records: list[dict] = []

    def log_chat_run(self, data: dict) -> None:
        self.records.append(dict(data))


def test_pii_output_returns_fallback_and_only_safe_answer_reaches_memory() -> None:
    raw_answer = "The employee email is private.employee@example.com."
    graph_result = valid_rag_result(raw_answer)

    service = ChatService.__new__(ChatService)
    service.settings = SimpleNamespace(
        guardrail_enabled=True,
        openai_default_model="gpt-4o-mini",
        ollama_default_model="qwen2.5:0.5b",
    )
    service.graph_service = StaticGraphService(graph_result)
    service.guardrail_service = GuardrailService()
    service.memory_service = RecordingMemoryService()
    service.logging_service = RecordingLoggingService()

    response = service.chat(
        ChatRequest(
            message="What is the employee email?",
            session_id="pii-integration-test",
            provider="ollama",
            model="qwen2.5:0.5b",
        )
    )

    expected_fallback = (
        "I can't provide that response because it may contain sensitive information."
    )
    assert response.answer == expected_fallback
    assert response.llm_response is None
    assert service.memory_service.messages == [
        ("pii-integration-test", "assistant", expected_fallback)
    ]
    assert raw_answer not in str(service.memory_service.messages)

    log_record = service.logging_service.records[0]
    assert log_record["guardrail_failure_type"] == "pii"
    assert log_record["llm_response"] is None
    assert raw_answer not in log_record["answer"]
