import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.classify.classify import (
    ClassificationError,
    ClassificationResult,
    classify_document,
    load_hf_credentials,
)
from src.metadata.metadata import DocumentMetadata


def create_env_example(env_dir: Path) -> Path:
    env_dir.mkdir(parents=True, exist_ok=True)
    example_file = env_dir / ".env.example"
    example_file.write_text("HF_API_TOKEN=test-token\nHF_MODEL_NAME=test-model\n")
    return env_dir


def test_load_hf_credentials_from_env_file(tmp_path):
    env_dir = create_env_example(tmp_path / ".env")

    token, model = load_hf_credentials(env_path=env_dir)

    assert token == "test-token"
    assert model == "test-model"


def test_classify_document_parses_mocked_inference_response(tmp_path):
    env_dir = create_env_example(tmp_path / ".env")

    metadata = DocumentMetadata(
        filename="doc.pdf",
        size_bytes=1234,
        created_timestamp=1.0,
        modified_timestamp=2.0,
        sha256_hash="abc123",
        language="en",
        word_count=10,
        pdf_author="Author",
        pdf_title="Title",
    )
    text = "Some sample PDF text for classification."
    expected_payload = {
        "classification": "Green",
        "confidence": "high",
        "rationale": "Document appears non-sensitive.",
    }

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def text_generation(self, prompt, *, model, max_new_tokens, temperature, return_full_text, **kwargs):
            return f"{prompt}\n\n{json.dumps(expected_payload)}"

    with patch("src.classify.classify.InferenceClient", DummyClient):
        result = classify_document(metadata, text, env_path=env_dir)

    assert isinstance(result, ClassificationResult)
    assert result.classification == "Green"
    assert result.confidence == "high"
    assert result.rationale == "Document appears non-sensitive."


def test_classify_document_retries_on_malformed_json(tmp_path):
    env_dir = create_env_example(tmp_path / ".env")

    metadata = DocumentMetadata(
        filename="doc.pdf",
        size_bytes=1234,
        created_timestamp=1.0,
        modified_timestamp=2.0,
        sha256_hash="abc123",
        language="en",
        word_count=10,
        pdf_author="Author",
        pdf_title="Title",
    )
    text = "Some sample PDF text for classification."
    expected_payload = {
        "classification": "Red",
        "confidence": "medium",
        "rationale": "Contains potentially sensitive content.",
    }

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def text_generation(self, prompt, *, model, max_new_tokens, temperature, return_full_text, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "Invalid response"
            return f"{prompt}\n\n{json.dumps(expected_payload)}"

    with patch("src.classify.classify.InferenceClient", DummyClient):
        result = classify_document(metadata, text, env_path=env_dir)

    assert result.classification == "Red"
    assert result.confidence == "medium"
    assert result.rationale == "Contains potentially sensitive content."


def test_classify_document_fails_after_two_bad_attempts(tmp_path):
    env_dir = create_env_example(tmp_path / ".env")

    metadata = DocumentMetadata(
        filename="doc.pdf",
        size_bytes=1234,
        created_timestamp=1.0,
        modified_timestamp=2.0,
        sha256_hash="abc123",
        language="en",
        word_count=10,
        pdf_author="Author",
        pdf_title="Title",
    )
    text = "Some sample PDF text for classification."

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def text_generation(self, prompt, *, model, max_new_tokens, temperature, return_full_text, **kwargs):
            return "Invalid response"

    with patch("src.classify.classify.InferenceClient", DummyClient):
        with pytest.raises(ClassificationError):
            classify_document(metadata, text, env_path=env_dir)
