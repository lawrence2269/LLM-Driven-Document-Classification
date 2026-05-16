import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


def append_audit_log_entry(
    run_id: str,
    original_path: Union[str, Path],
    classification: str,
    confidence: str,
    rationale: str,
    model: str,
    sha256: str,
    log_path: Union[str, Path] = Path("logs/run_log.jsonl"),
    timestamp: Optional[str] = None,
) -> Path:
    """Append an audit record to logs/run_log.jsonl as a single JSON line."""
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    entry = {
        "run_id": run_id,
        "timestamp": timestamp,
        "original_path": str(original_path),
        "classification": classification,
        "confidence": confidence,
        "rationale": rationale,
        "model": model,
        "sha256": sha256,
    }

    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True))
        handle.write("\n")

    return log_file
