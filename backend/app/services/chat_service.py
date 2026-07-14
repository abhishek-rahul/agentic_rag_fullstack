from app.core.config import get_settings
from app.domain.models import ChatRequest, ChatResponse, SourceDocument
from app.services.graph_service import GraphService
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.services.logging_service import LoggingService
from app.services.guardrail_service import GuardrailService


class ChatService:
    def __init__(self):
        self.settings = get_settings()
        self.memory_service = MemoryService()
        self.rag_service = RAGService()
        self.graph_service = GraphService(self.memory_service, self.rag_service)
        self.logging_service = LoggingService()
        self.guardrail_service = GuardrailService()

    def chat(self, request: ChatRequest) -> ChatResponse:
        selected_model = request.model
        if not selected_model:
            selected_model = (
                self.settings.openai_default_model
                if request.provider == "openai"
                else self.settings.ollama_default_model
            )

        if self.settings.guardrail_enabled:
            request_guardrail = self.guardrail_service.validate_request(
                provider=request.provider,
                model=selected_model,
                message=request.message,
            )

            if not request_guardrail.passed:
                fallback = self.guardrail_service.fallback_answer(request_guardrail.reason)

                self.logging_service.log_chat_run({
                    "session_id": request.session_id,
                    "provider": request.provider,
                    "model": selected_model,
                    "user_message": request.message,
                    "answer": fallback,
                    "sources": [],
                    "guardrail_enabled": True,
                    "guardrail_passed": False,
                    "guardrail_reason": request_guardrail.reason,
                })

                return ChatResponse(
                    answer=fallback,
                    session_id=request.session_id,
                    provider=request.provider,
                    model=selected_model,
                    sources=[],
                )

        result = self.graph_service.run(
            message=request.message,
            session_id=request.session_id,
            provider=request.provider,
            model=selected_model,
        )

        guardrail_log = {"guardrail_enabled": self.settings.guardrail_enabled}
        if self.settings.guardrail_enabled:
            rag_guardrail = self.guardrail_service.validate_rag_result(result)
            guardrail_log.update({
                "guardrail_passed": rag_guardrail.passed,
                "guardrail_reason": rag_guardrail.reason,
            })

            if not rag_guardrail.passed:
                result["answer"] = self.guardrail_service.fallback_answer(rag_guardrail.reason)
                result["sources"] = []

        sources = [SourceDocument(**source) for source in result.get("sources", [])]

        self.logging_service.log_chat_run({
            "session_id": request.session_id,
            "provider": request.provider,
            "model": selected_model,
            "user_message": request.message,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            **guardrail_log,
        })

        return ChatResponse(
            answer=result["answer"],
            session_id=request.session_id,
            provider=request.provider,
            model=selected_model,
            sources=sources,
        )
