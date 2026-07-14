import json
from pathlib import Path
from typing import Any


DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "data" / "eval_dataset.json"
)


def load_eval_dataset() -> list[dict[str, Any]]:
    with DATASET_PATH.open(encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def contains_expected_keywords(
    answer: str,
    expected_keywords: list[str],
) -> bool:
    normalized_answer = answer.casefold()
    return all(keyword.casefold() in normalized_answer for keyword in expected_keywords)
