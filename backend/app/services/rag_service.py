from typing import List, Tuple

from langchain_core.documents import Document

from app.core.config import get_settings
from app.infrastructure.vector_store import LocalVectorStore


class RAGService:
    def __init__(self):
        self.settings = get_settings()
        self.vector_store = LocalVectorStore()

    def ingest_documents(self) -> tuple[int, int]:
        return self.vector_store.ingest()

    def retrieve(self, query: str) -> Tuple[str, List[dict]]:
        store = self.vector_store.load()
        if store is None:
            return "", []

        retriever = store.as_retriever(search_kwargs={"k": self.settings.retriever_k})
        docs: List[Document] = retriever.invoke(query)

        context_parts = []
        sources = []
        for doc in docs:
            source = doc.metadata.get("source")
            context_parts.append(f"Source: {source}\nContent: {doc.page_content}")
            sources.append(
                {
                    "source": source,
                    "content_preview": doc.page_content[:300],
                }
            )

        return "\n\n".join(context_parts), sources
