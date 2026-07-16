from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "Local Agentic RAG Assistant"
    app_env: str = "local"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    openai_api_key: str | None = None
    openai_default_model: str = "gpt-4o-mini"

    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "qwen2.5:0.5b"

    # Embeddings are separate from chat models. Ollama is default for a local-first setup.
    embeddings_provider: str = "ollama"  # ollama or openai
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_embedding_model: str = "nomic-embed-text:latest"

    docs_dir: Path = BASE_DIR / "data" / "docs"
    faiss_index_dir: Path = BASE_DIR / "data" / "faiss_index"
    sqlite_db_path: Path = BASE_DIR / "data" / "memory.db"

    chunk_size: int = 800
    chunk_overlap: int = 120
    retriever_k: int = 4
    short_term_turns: int = 6
    guardrail_enabled: bool = True
    agent_max_steps: int = 3
    agent_max_repeat_calls: int = 1
    test_llm_provider: str = "ollama"
    test_llm_model: str = "qwen2.5:0.5b"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
