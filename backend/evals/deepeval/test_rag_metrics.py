import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import pytest
from dotenv import load_dotenv

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from evals.shared import BACKEND_DIR, load_record_envelope


load_dotenv(BACKEND_DIR / ".env")

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o-mini")
THRESHOLD = float(os.getenv("DEEPEVAL_METRIC_THRESHOLD", "0.6"))

try:
    RECORDS = load_record_envelope()["records"]
    RECORD_LOAD_ERROR = None
except (FileNotFoundError, ValueError) as exc:
    RECORDS = [{"id": "records-unavailable"}]
    RECORD_LOAD_ERROR = str(exc)


@pytest.mark.deepeval
@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_rag_metrics(record: dict) -> None:
    if RECORD_LOAD_ERROR:
        pytest.fail(
            "Generate the four shared records before running DeepEval: "
            f"{RECORD_LOAD_ERROR}"
        )
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for the DeepEval judge")

    assert record["generation_error"] is None
    assert record["guardrail"]["passed"] is True
    assert record["raw_answer"]
    assert record["contexts"]

    test_case = LLMTestCase(
        input=record["question"],
        actual_output=record["raw_answer"],
        retrieval_context=record["contexts"],
    )
    assert_test(
        test_case,
        [
            FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL),
            AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL),
        ],
    )
