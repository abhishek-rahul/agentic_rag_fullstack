import re
from typing import Any


class GuardrailResult:
    def __init__(
        self,
        passed: bool,
        reason: str,
        failure_type: str | None = None,
    ):
        self.passed = passed
        self.reason = reason
        self.failure_type = failure_type


class GuardrailService:
    def __init__(self):
        self.allowed_providers = {"openai", "ollama"}
        self.max_answer_length = 2500
        self._sensitive_patterns = (
            (
                "email address",
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            ),
            (
                "phone number",
                re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)"),
            ),
            (
                "phone number",
                re.compile(r"(?<!\d)\+91[\s-]?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
            ),
            (
                "card-like number",
                re.compile(r"(?<!\d)(?:\d[ -]?){15}\d(?!\d)"),
            ),
        )

    def _detect_sensitive_data(self, answer: str) -> str | None:
        for sensitive_type, pattern in self._sensitive_patterns:
            if pattern.search(answer):
                return sensitive_type

        return None

    def validate_request(self, provider: str, model: str | None, message: str) -> GuardrailResult:
        if provider not in self.allowed_providers:
            return GuardrailResult(
                False,
                f"Unsupported provider: {provider}",
                "request",
            )

        if not model or not model.strip():
            return GuardrailResult(False, "Model is required", "request")

        if not message or not message.strip():
            return GuardrailResult(False, "Message cannot be empty", "request")

        return GuardrailResult(True, "Request passed")

    def validate_rag_result(self, result: dict[str, Any]) -> GuardrailResult:
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        context = result.get("context", "")
        best_score = result.get("best_retrieval_score")
        generation_error = result.get("generation_error")

        if generation_error:
            return GuardrailResult(
                False,
                "Structured output generation failed",
                "schema",
            )

        if best_score is None:
            return GuardrailResult(False, "No retrieval score found", "retrieval")

        if best_score > 1.2:
            return GuardrailResult(
                False,
                f"Retrieved context is not relevant enough. Score={best_score}",
                "retrieval",
            )

        if not context or not context.strip():
            return GuardrailResult(False, "No relevant context found", "retrieval")

        if not answer or not answer.strip():
            return GuardrailResult(False, "Empty answer generated", "output")

        sensitive_type = self._detect_sensitive_data(answer)
        if sensitive_type:
            return GuardrailResult(
                False,
                f"Sensitive data detected: {sensitive_type}",
                "pii",
            )

        if len(answer) > self.max_answer_length:
            return GuardrailResult(False, "Answer is too long", "output")

        if not sources:
            return GuardrailResult(False, "Sources missing", "retrieval")

        return GuardrailResult(True, "RAG result passed")

    def fallback_answer(
        self,
        reason: str,
        failure_type: str | None = None,
    ) -> str:
        if failure_type == "pii":
            return (
                "I can't provide that response because it may contain "
                "sensitive information."
            )

        if failure_type == "schema":
            return "I couldn't generate a valid response. Please try again."

        return (
            "I don't have enough reliable information in the available documents "
            "to answer this question accurately."
        )
