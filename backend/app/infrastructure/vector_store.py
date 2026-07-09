from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.services.llm_gateway import LLMGateway


class LocalVectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.gateway = LLMGateway()

    def _load_documents(self) -> List[Document]:
        loader = DirectoryLoader(
            str(self.settings.docs_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        return loader.load()

    def ingest(self) -> tuple[int, int]:
        documents = self._load_documents()
        if not documents:
            return 0, 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        embeddings = self.gateway.get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(str(self.settings.faiss_index_dir))
        return len(documents), len(chunks)

    def load(self) -> FAISS | None:
        index_file = Path(self.settings.faiss_index_dir) / "index.faiss"
        pkl_file = Path(self.settings.faiss_index_dir) / "index.pkl"
        if not index_file.exists() or not pkl_file.exists():
            return None

        embeddings = self.gateway.get_embeddings()
        return FAISS.load_local(
            str(self.settings.faiss_index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
