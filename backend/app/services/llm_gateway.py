from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import get_settings


class LLMGateway:
    def __init__(self):
        self.settings = get_settings()

    def get_chat_model(self, provider: str, model: str | None = None) -> BaseChatModel:
        provider = provider.lower()

        if provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is missing. Add it to backend/.env.")

            return ChatOpenAI(
                model=model or self.settings.openai_default_model,
                api_key=self.settings.openai_api_key,
                temperature=0.2,
            )

        if provider == "ollama":
            return ChatOllama(
                model=model or self.settings.ollama_default_model,
                base_url=self.settings.ollama_base_url,
                temperature=0.2,
            )

        raise ValueError("Unsupported provider. Use 'openai' or 'ollama'.")

    def get_embeddings(self):
        provider = self.settings.embeddings_provider.lower()

        if provider == "openai":
            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is missing for OpenAI embeddings.")

            return OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                api_key=self.settings.openai_api_key,
            )

        if provider == "ollama":
            return OllamaEmbeddings(
                model=self.settings.ollama_embedding_model,
                base_url=self.settings.ollama_base_url,
            )

        raise ValueError("Unsupported embeddings provider. Use 'openai' or 'ollama'.")
