from pathlib import Path

import pytest

from app.services.rag_service import RAGService
from tests.helpers import load_eval_dataset


TEST_CASES = load_eval_dataset()


@pytest.fixture(scope="module")
def rag_service() -> RAGService:
    service = RAGService()
    index_dir = service.vector_store.settings.faiss_index_dir
    required_files = (index_dir / "index.faiss", index_dir / "index.pkl")

    if not all(path.exists() for path in required_files):
        try:
            documents_loaded, chunks_created = service.ingest_documents()
        except Exception as exc:
            pytest.fail(f"Could not build the FAISS test index: {exc}")

        assert documents_loaded > 0, "No documents were available for FAISS ingestion"
        assert chunks_created > 0, "FAISS ingestion did not create any chunks"

    return service


@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[test_case["id"] for test_case in TEST_CASES],
)
def test_retrieval_matches_expected_context(
    rag_service: RAGService,
    test_case: dict,
) -> None:
    result = rag_service.retrieve(test_case["question"])
    context = result.get("context", "")
    sources = result.get("sources", [])
    source_filenames = [
        Path(source.get("source", "")).name
        for source in sources
    ]

    if test_case["should_find_context"]:
        assert context.strip(), (
            f"Expected context for '{test_case['question']}', but retrieval returned none"
        )
        assert sources, (
            f"Expected sources for '{test_case['question']}', but retrieval returned none"
        )
        assert test_case["expected_source"] in source_filenames, (
            f"Expected source '{test_case['expected_source']}' but received "
            f"{source_filenames}"
        )
        return

    assert not context.strip() and not sources, (
        "Expected no relevant context, but retrieval returned: "
        f"{source_filenames}"
    )
