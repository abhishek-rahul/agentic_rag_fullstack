from typing import Any


class GuardrailResult:
    def __init__(self, passed: bool, reason: str):
        self.passed = passed
        self.reason = reason


class GuardrailService:
    def __init__(self):
        self.allowed_providers = {"openai", "ollama"}
        self.max_answer_length = 2500

    def validate_request(self, provider: str, model: str | None, message: str) -> GuardrailResult:
        if provider not in self.allowed_providers:
            return GuardrailResult(False, f"Unsupported provider: {provider}")

        if not model or not model.strip():
            return GuardrailResult(False, "Model is required")

        if not message or not message.strip():
            return GuardrailResult(False, "Message cannot be empty")

        return GuardrailResult(True, "Request passed")

    def validate_rag_result(self, result: dict[str, Any]) -> GuardrailResult:
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        context = result.get("context", "")
        best_score = result.get("best_retrieval_score")

        if best_score is None:
            return GuardrailResult(False, "No retrieval score found")
        
        if best_score > 1.2:
            return GuardrailResult(False, f"Retrieved context is not relevant enough. Score={best_score}")
        
        if not context or not context.strip():
            return GuardrailResult(False, "No relevant context found")

        if not answer or not answer.strip():
            return GuardrailResult(False, "Empty answer generated")

        if len(answer) > self.max_answer_length:
            return GuardrailResult(False, "Answer is too long")

        if not sources:
            return GuardrailResult(False, "Sources missing")

        return GuardrailResult(True, "RAG result passed")

    def fallback_answer(self, reason: str) -> str:
        return (
            "I don't have enough reliable information in the available documents "
            "to answer this question accurately."
        )