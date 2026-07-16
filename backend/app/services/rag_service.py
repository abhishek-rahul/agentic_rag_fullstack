from app.infrastructure.vector_store import LocalVectorStore


class RAGService:
    def __init__(self):
        self.vector_store = LocalVectorStore()
        self.min_relevance_score = 1.2

    def ingest_documents(self) -> tuple[int, int]:
        return self.vector_store.ingest()

    def retrieve(self, query: str, k: int = 4) -> dict:
        store = self.vector_store.load()

        if store is None:
            return {
                "context": "",
                "contexts": [],
                "sources": [],
                "best_score": None,
            }

        docs_with_scores = store.similarity_search_with_score(query, k=k)

        if not docs_with_scores:
            return {
                "context": "",
                "contexts": [],
                "sources": [],
                "best_score": None,
            }

        filtered_docs = []

        for doc, score in docs_with_scores:
            #print("score is -- " + str(score))
            if score <= self.min_relevance_score:
                filtered_docs.append((doc, score))

        if not filtered_docs:
            return {
                "context": "",
                "contexts": [],
                "sources": [],
                "best_score": float(docs_with_scores[0][1]),
            }

        contexts = [doc.page_content for doc, score in filtered_docs]
        context = "\n\n".join(contexts)

        sources = []
        for doc, score in filtered_docs:
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "content_preview": doc.page_content[:300],
                "score": float(score),
            })

        return {
            "context": context,
            "contexts": contexts,
            "sources": sources,
            "best_score": float(filtered_docs[0][1]),
        }
