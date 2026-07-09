import json
from datetime import datetime
from pathlib import Path
from typing import Any


class LoggingService:
    def __init__(self):
        self.log_dir = Path(__file__).resolve().parent.parent / "data" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "chat_runs.jsonl"

    def log_chat_run(self, data: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            **data,
        }

        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")