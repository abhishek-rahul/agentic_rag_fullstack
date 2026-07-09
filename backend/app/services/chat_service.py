from app.core.config import get_settings
from app.domain.models import ChatRequest, ChatResponse, SourceDocument
from app.services.graph_service import GraphService
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService

from app.services.logging_service import LoggingService

class ChatService:
    def __init__(self):
        self.settings = get_settings()
        self.logging_service = LoggingService()
        self.memory_service = MemoryService()
        self.rag_service = RAGService()
        self.graph_service = GraphService(self.memory_service, self.rag_service)

    def chat(self, request: ChatRequest) -> ChatResponse:
        selected_model = request.model
        if not selected_model:
            selected_model = (
                self.settings.openai_default_model
                if request.provider == "openai"
                else self.settings.ollama_default_model
            )

        result = self.graph_service.run(
            message=request.message,
            session_id=request.session_id,
            provider=request.provider,
            model=selected_model,
        )

        self.logging_service.log_chat_run({
            "session_id": request.session_id,
            "provider": request.provider,
            "model": request.model,
            "user_message": request.message,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
        })

        
        sources = [SourceDocument(**source) for source in result.get("sources", [])]
        return ChatResponse(
            answer=result["answer"],
            session_id=request.session_id,
            provider=request.provider,
            model=selected_model,
            sources=sources,
        )
