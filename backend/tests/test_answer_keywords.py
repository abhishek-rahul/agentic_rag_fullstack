from collections import defaultdict
from uuid import uuid4

import pytest

from app.core.config import get_settings
from app.domain.models import ChatRequest
from app.services.chat_service import ChatService
from app.services.graph_service import GraphService
from app.services.guardrail_service import GuardrailService
from app.services.rag_service import RAGService
from tests.helpers import contains_expected_keywords, load_eval_dataset


ANSWER_TEST_CASES = [
    test_case
    for test_case in load_eval_dataset()
    if test_case.get("run_answer_test")
]

PROVIDER_UNAVAILABLE_MESSAGES = (
    "connection refused",
    "failed to connect",
    "all connection attempts failed",
    "openai_api_key is missing",
    "not found, try pulling it first",
    "model not found",
    "timed out",
)


def provider_is_unavailable(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(indicator in message for indicator in PROVIDER_UNAVAILABLE_MESSAGES)


class EphemeralMemoryService:
    def __init__(self):
        self.messages: dict[str, list[dict[str, str]]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.messages[session_id].append({"role": role, "content": content})

    def load_memory(self, session_id: str) -> list[dict[str, str]]:
        return list(self.messages[session_id])

    def format_for_prompt(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            return "No previous conversation."

        return "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in messages
        )


class NoOpLoggingService:
    def log_chat_run(self, data: dict) -> None:
        return None


@pytest.fixture(scope="module")
def chat_service() -> ChatService:
    memory_service = EphemeralMemoryService()
    rag_service = RAGService()

    service = ChatService.__new__(ChatService)
    service.settings = get_settings()
    service.memory_service = memory_service
    service.rag_service = rag_service
    service.graph_service = GraphService(memory_service, rag_service)
    service.logging_service = NoOpLoggingService()
    service.guardrail_service = GuardrailService()
    return service


@pytest.mark.llm
@pytest.mark.parametrize(
    "test_case",
    ANSWER_TEST_CASES,
    ids=[test_case["id"] for test_case in ANSWER_TEST_CASES],
)
def test_answer_contains_expected_keywords(
    chat_service: ChatService,
    test_case: dict,
) -> None:
    settings = get_settings()
    request = ChatRequest(
        message=test_case["question"],
        session_id=f"test-{test_case['id']}-{uuid4()}",
        provider=settings.test_llm_provider,
        model=settings.test_llm_model,
    )

    try:
        response = chat_service.chat(request)
    except Exception as exc:
        if provider_is_unavailable(exc):
            pytest.skip(
                "Configured LLM or embedding provider is unavailable: "
                f"{exc}"
            )
        raise

    print(
        f"\nCase: {test_case['id']}"
        f"\nQuestion: {test_case['question']}"
        f"\nActual answer: {response.answer}\n"
    )

    assert response.answer.strip(), (
        f"Expected a non-empty answer for '{test_case['question']}'"
    )

    expected_keywords = test_case["expected_keywords"]
    if not contains_expected_keywords(response.answer, expected_keywords):
        missing_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.casefold() not in response.answer.casefold()
        ]
        pytest.fail(
            f"Expected keywords {missing_keywords} were missing from answer:\n"
            f"{response.answer}"
        )
