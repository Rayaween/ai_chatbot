from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

LOG_PATH = Path("logs/requests.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

COST_PER_1K_INPUT = 0.00015
COST_PER_1K_OUTPUT = 0.00060

def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000.0) * COST_PER_1K_INPUT + (output_tokens / 1000.0) * COST_PER_1K_OUTPUT

def log_request(
        *,
        endpoint: str,
        session_id: str,
        question: str,
        answer: str,
        context: List[Dict],
        input_tokens_est: int,
        output_tokens_est: int,
        total_latency_sec: float,
        first_token_latency_sec: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "ts": time.time(),
        "endpoint": endpoint,
        "session_id": session_id,
        "question": question,
        "answer_len": len(answer),
        "context_len": len(context),
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
        "cost_estimate": estimate_cost(input_tokens_est, output_tokens_est),
        "total_latency_sec": total_latency_sec,
        "first_token_latency_sec": first_token_latency_sec,
    }
    if extra:
        record.update(extra)

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record