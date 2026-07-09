from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.domain.models import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from app.services.chat_service import ChatService
from app.services.rag_service import RAGService

router = APIRouter()
settings = get_settings()
chat_service = ChatService()
rag_service = RAGService()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest():
    try:
        documents_loaded, chunks_created = rag_service.ingest_documents()
        return IngestResponse(
            status="completed",
            documents_loaded=documents_loaded,
            chunks_created=chunks_created,
            vector_store_path=str(settings.faiss_index_dir),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return chat_service.chat(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        message = str(exc)
        if "Connection refused" in message or "Failed to connect" in message:
            raise HTTPException(
                status_code=503,
                detail="Could not connect to Ollama. Start it with: ollama serve",
            ) from exc
        raise HTTPException(status_code=500, detail=message) from exc
