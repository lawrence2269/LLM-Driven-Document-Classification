import json
from pathlib import Path

from src.logger.logger import append_audit_log_entry


def test_append_audit_log_entry_writes_valid_jsonl(tmp_path):
    log_path = tmp_path / "logs" / "run_log.jsonl"
    entry_path = append_audit_log_entry(
        run_id="run-123",
        original_path="inbox/doc.pdf",
        classification="Green",
        confidence="high",
        rationale="Approved for storage.",
        model="test-model",
        sha256="abc123",
        log_path=log_path,
    )

    assert entry_path == log_path
    assert log_path.exists()

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    loaded = json.loads(lines[0])
    assert loaded["run_id"] == "run-123"
    assert loaded["original_path"] == "inbox/doc.pdf"
    assert loaded["classification"] == "Green"
    assert loaded["confidence"] == "high"
    assert loaded["rationale"] == "Approved for storage."
    assert loaded["model"] == "test-model"
    assert loaded["sha256"] == "abc123"
    assert "timestamp" in loaded


def test_append_audit_log_entry_appends_multiple_lines(tmp_path):
    log_path = tmp_path / "logs" / "run_log.jsonl"
    append_audit_log_entry(
        run_id="run-1",
        original_path="inbox/doc1.pdf",
        classification="Green",
        confidence="medium",
        rationale="First file.",
        model="test-model",
        sha256="hash1",
        log_path=log_path,
    )
    append_audit_log_entry(
        run_id="run-2",
        original_path="inbox/doc2.pdf",
        classification="Red",
        confidence="low",
        rationale="Second file.",
        model="test-model",
        sha256="hash2",
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "run-1"
    assert json.loads(lines[1])["run_id"] == "run-2"
