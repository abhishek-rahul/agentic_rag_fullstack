from typing import Literal

from pydantic import BaseModel, Field


Provider = Literal["openai", "ollama"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    provider: Provider = "ollama"
    model: str | None = None


class SourceDocument(BaseModel):
    source: str | None = None
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    provider: Provider
    model: str
    sources: list[SourceDocument] = []


class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    vector_store_path: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
