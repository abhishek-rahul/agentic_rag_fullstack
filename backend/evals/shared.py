from __future__ import annotations

import json
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


BACKEND_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BACKEND_DIR / "app" / "data" / "eval_dataset.json"
RECORDS_PATH = BACKEND_DIR / "eval_results" / "shared" / "core_rag_records.json"

CORE_EVAL_CASE_COUNT = 4
SCHEMA_VERSION = 1

_SENSITIVE_PATTERNS = (
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    re.compile(r"(?<!\d)(?:\d[ -]?){15}\d(?!\d)"),
)


@dataclass(frozen=True)
class EvaluationRecord:
    id: str
    question: str
    reference_answer: str
    expected_keywords: list[str]
    expected_source: str
    provider: str
    model: str
    prompt_variant: str
    answer: str
    raw_answer: str | None
    contexts: list[str]
    source_filenames: list[str]
    best_retrieval_score: float | None
    grounded_in_context: bool | None
    guardrail: dict[str, Any]
    generation_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EphemeralMemoryService:
    def __init__(self) -> None:
        self.messages: dict[str, list[dict[str, str]]] = defaultdict(list)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.messages[session_id].append({"role": role, "content": content})

    def load_memory(self, session_id: str) -> list[dict[str, str]]:
        return list(self.messages[session_id])

    def format_for_prompt(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            return "No previous conversation."
        return "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in messages
        )


class NoOpLoggingService:
    def log_chat_run(self, data: dict[str, Any]) -> None:
        return None


def load_eval_dataset() -> list[dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)

    if not isinstance(dataset, list):
        raise ValueError("Evaluation dataset must be a JSON array")
    return dataset


def load_core_eval_cases() -> list[dict[str, Any]]:
    cases = [case for case in load_eval_dataset() if case.get("run_core_eval")]
    if len(cases) != CORE_EVAL_CASE_COUNT:
        raise ValueError(
            f"Expected {CORE_EVAL_CASE_COUNT} core evaluation cases, found {len(cases)}"
        )

    required_fields = {
        "id",
        "question",
        "reference_answer",
        "expected_keywords",
        "expected_source",
    }
    for case in cases:
        missing = required_fields.difference(case)
        if missing:
            raise ValueError(
                f"Core case '{case.get('id', 'unknown')}' is missing {sorted(missing)}"
            )
    return cases


def sanitize_generation_error(error: Any) -> str | None:
    if error is None:
        return None

    sanitized = " ".join(str(error).split())
    for pattern in _SENSITIVE_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized[:500]


def _build_application_service(system_prompt_template: str | None = None):
    # Production imports stay inside this function so judge-only environments can
    # import record helpers without importing LangChain, LangGraph, or the app.
    from app.core.config import get_settings
    from app.services.chat_service import ChatService
    from app.services.graph_service import GraphService
    from app.services.guardrail_service import GuardrailService
    from app.services.rag_service import RAGService

    memory_service = EphemeralMemoryService()
    rag_service = RAGService()

    service = ChatService.__new__(ChatService)
    service.settings = get_settings()
    service.memory_service = memory_service
    service.rag_service = rag_service
    service.graph_service = GraphService(
        memory_service,
        rag_service,
        system_prompt_template=system_prompt_template,
    )
    service.logging_service = NoOpLoggingService()
    service.guardrail_service = GuardrailService()
    return service


def ensure_faiss_index(rag_service: Any) -> None:
    index_dir = rag_service.vector_store.settings.faiss_index_dir
    required_files = (index_dir / "index.faiss", index_dir / "index.pkl")
    if all(path.exists() for path in required_files):
        return

    documents_loaded, chunks_created = rag_service.ingest_documents()
    if documents_loaded <= 0 or chunks_created <= 0:
        raise RuntimeError("FAISS ingestion did not create a usable index")


def _request_failure_record(
    test_case: dict[str, Any],
    provider: str,
    model: str,
    prompt_variant: str,
    guardrail_service: Any,
    guardrail_result: Any,
) -> EvaluationRecord:
    fallback = guardrail_service.fallback_answer(
        guardrail_result.reason,
        guardrail_result.failure_type,
    )
    return EvaluationRecord(
        id=test_case["id"],
        question=test_case["question"],
        reference_answer=test_case["reference_answer"],
        expected_keywords=list(test_case["expected_keywords"]),
        expected_source=test_case["expected_source"],
        provider=provider,
        model=model,
        prompt_variant=prompt_variant,
        answer=fallback,
        raw_answer=None,
        contexts=[],
        source_filenames=[],
        best_retrieval_score=None,
        grounded_in_context=None,
        guardrail={
            "passed": False,
            "reason": guardrail_result.reason,
            "failure_type": guardrail_result.failure_type,
        },
        generation_error=None,
    )


def run_application_case(
    test_case: dict[str, Any],
    provider: str,
    model: str,
    prompt_variant: str = "production",
    system_prompt_template: str | None = None,
    service: Any | None = None,
) -> EvaluationRecord:
    application = service or _build_application_service(system_prompt_template)
    if not application.settings.guardrail_enabled:
        raise RuntimeError("Evaluation requires GUARDRAIL_ENABLED=true")

    request_guardrail = application.guardrail_service.validate_request(
        provider=provider,
        model=model,
        message=test_case["question"],
    )
    if not request_guardrail.passed:
        return _request_failure_record(
            test_case,
            provider,
            model,
            prompt_variant,
            application.guardrail_service,
            request_guardrail,
        )

    graph_result = application.graph_service.run(
        message=test_case["question"],
        session_id=f"eval-{test_case['id']}-{uuid4()}",
        provider=provider,
        model=model,
    )
    raw_result = dict(graph_result)
    safe_result, output_guardrail = application._apply_output_guardrail(raw_result)
    passed = bool(output_guardrail.passed)

    llm_response = raw_result.get("llm_response")
    grounded_in_context = None
    if passed and isinstance(llm_response, dict):
        grounded_value = llm_response.get("grounded_in_context")
        if isinstance(grounded_value, bool):
            grounded_in_context = grounded_value

    source_filenames = [
        Path(source.get("source", "")).name
        for source in raw_result.get("sources", [])
    ]

    return EvaluationRecord(
        id=test_case["id"],
        question=test_case["question"],
        reference_answer=test_case["reference_answer"],
        expected_keywords=list(test_case["expected_keywords"]),
        expected_source=test_case["expected_source"],
        provider=provider,
        model=model,
        prompt_variant=prompt_variant,
        answer=safe_result.get("answer", ""),
        raw_answer=raw_result.get("answer") if passed else None,
        contexts=list(raw_result.get("contexts", [])),
        source_filenames=source_filenames,
        best_retrieval_score=raw_result.get("best_retrieval_score"),
        grounded_in_context=grounded_in_context,
        guardrail={
            "passed": passed,
            "reason": output_guardrail.reason,
            "failure_type": output_guardrail.failure_type,
        },
        generation_error=sanitize_generation_error(
            raw_result.get("generation_error")
        ),
    )


def validate_retrieval(record: EvaluationRecord) -> list[str]:
    errors: list[str] = []
    if not record.contexts or not any(context.strip() for context in record.contexts):
        errors.append("retrieval returned no context")
    if record.expected_source not in record.source_filenames:
        errors.append(
            f"expected source '{record.expected_source}' was not retrieved"
        )

    combined_context = "\n".join(record.contexts).casefold()
    missing_keywords = [
        keyword
        for keyword in record.expected_keywords
        if keyword.casefold() not in combined_context
    ]
    if missing_keywords:
        errors.append(f"retrieved context is missing keywords {missing_keywords}")
    return errors


def write_record_envelope(
    records: list[EvaluationRecord],
    provider: str,
    model: str,
    prompt_variant: str,
    path: Path = RECORDS_PATH,
) -> None:
    if len(records) != CORE_EVAL_CASE_COUNT:
        raise ValueError(
            f"Refusing to write a partial record file with {len(records)} records"
        )

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_provider": provider,
        "generation_model": model,
        "prompt_variant": prompt_variant,
        "records": [record.to_dict() for record in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as temp_file:
        json.dump(envelope, temp_file, indent=2, ensure_ascii=False)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)
    temp_path.replace(path)


def load_record_envelope(path: Path = RECORDS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as record_file:
        envelope = json.load(record_file)

    if envelope.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported evaluation record schema version")
    records = envelope.get("records")
    if not isinstance(records, list) or len(records) != CORE_EVAL_CASE_COUNT:
        raise ValueError("Evaluation record file must contain exactly four records")
    return envelope
