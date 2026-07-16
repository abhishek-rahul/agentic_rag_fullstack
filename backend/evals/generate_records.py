import os

from dotenv import load_dotenv

from evals.shared import (
    BACKEND_DIR,
    ensure_faiss_index,
    load_core_eval_cases,
    run_application_case,
    validate_retrieval,
    write_record_envelope,
)


def main() -> int:
    load_dotenv(BACKEND_DIR / ".env")
    provider = os.getenv("EVAL_GENERATION_PROVIDER", "ollama")
    model = os.getenv("EVAL_GENERATION_MODEL", "qwen2.5:0.5b")

    # Build one isolated application instance so FAISS and model configuration are
    # shared while every case still receives a unique session id.
    from evals.shared import _build_application_service

    service = _build_application_service()
    ensure_faiss_index(service.rag_service)

    records = []
    failures: list[str] = []
    for test_case in load_core_eval_cases():
        record = run_application_case(
            test_case,
            provider=provider,
            model=model,
            prompt_variant="production",
            service=service,
        )
        records.append(record)

        case_errors = validate_retrieval(record)
        if not record.guardrail["passed"]:
            case_errors.append(
                "guardrail failed: "
                f"{record.guardrail.get('failure_type') or 'unknown'}"
            )
        failures.extend(f"{record.id}: {error}" for error in case_errors)

    write_record_envelope(records, provider, model, "production")
    print(f"Generated {len(records)} sanitized evaluation records.")

    if failures:
        print("Evaluation record validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
