import json
from unittest.mock import patch

import pytest

from src.classify.classify import (
    ClassificationError,
    ClassificationResult,
    build_classification_prompts,
    classify_document,
    parse_classification_response,
)
from src.metadata.metadata import DocumentMetadata


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return " ".join(token_ids)


def patch_tokenizer():
    return patch("src.classify.classify.load_transformers_tokenizer", return_value=DummyTokenizer())


def make_metadata():
    return DocumentMetadata(
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


def test_parse_classification_response_valid_green():
    payload = json.dumps({
        "classification": "Green",
        "confidence": "high",
        "rationale": "Safe to store.",
    })
    result = parse_classification_response(payload)
    assert isinstance(result, ClassificationResult)
    assert result.classification == "Green"
    assert result.confidence == "high"
    assert result.rationale == "Safe to store."


def test_parse_classification_response_invalid_confidence_raises():
    bad = json.dumps({
        "classification": "Red",
        "confidence": "unknown",
        "rationale": "Something fishy.",
    })
    with pytest.raises(ClassificationError):
        parse_classification_response(bad)


def test_build_prompts_and_length():
    md = make_metadata()
    text = "Sample text content for building a prompt."
    system, user = build_classification_prompts(md, text)
    assert "Policy" in system
    assert "Document metadata summary" in user


def test_classify_document_returns_green(tmp_path):
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / ".env.example").write_text("HF_API_TOKEN=test-token\nHF_MODEL_NAME=test-model\n")

    metadata = make_metadata()
    text = "Test document text"

    expected = {"classification": "Green", "confidence": "high", "rationale": "Okay."}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def text_generation(self, prompt, *, model, max_new_tokens, temperature, return_full_text, **kwargs):
            return f"{prompt}\n\n{json.dumps(expected)}"

    with patch_tokenizer():
        with patch("src.classify.classify.InferenceClient", DummyClient):
            result = classify_document(metadata, text, env_path=env_dir)

    assert result.classification == "Green"
    assert result.confidence == "high"
    assert result.rationale == "Okay."


def test_classify_document_retries_and_fails(tmp_path):
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / ".env.example").write_text("HF_API_TOKEN=test-token\nHF_MODEL_NAME=test-model\n")

    metadata = make_metadata()
    text = "Test document text"

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def text_generation(self, prompt, *, model, max_new_tokens, temperature, return_full_text, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return "not json"
            return "still not json"

    with patch_tokenizer():
        with patch("src.classify.classify.InferenceClient", DummyClient):
            with pytest.raises(ClassificationError):
                classify_document(metadata, text, env_path=env_dir)
