import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.classify.classify import (
    ClassificationError,
    ClassificationResult,
    classify_document,
    load_hf_credentials,
)
from src.metadata.metadata import DocumentMetadata


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return " ".join(token_ids)


def patch_tokenizer():
    return patch("src.classify.classify.load_transformers_tokenizer", return_value=DummyTokenizer())


def create_env_example(env_dir: Path) -> Path:
    env_dir.mkdir(parents=True, exist_ok=True)
    example_file = env_dir / ".env.example"
    example_file.write_text(
        "HF_INFERENCE_BACKEND=api\nHF_API_TOKEN=test-token\nHF_MODEL_NAME=test-model\n"
    )
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

    with patch_tokenizer():
        with patch("src.classify.classify.InferenceClient", DummyClient):
            result = classify_document(metadata, text, env_path=env_dir)

    assert isinstance(result, ClassificationResult)
    assert result.classification == "Green"
    assert result.confidence == "high"
    assert result.rationale == "Document appears non-sensitive."


def test_classify_document_uses_local_transformers_backend(tmp_path):
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / ".env.example").write_text(
        "HF_INFERENCE_BACKEND=local\nHF_MODEL_NAME=test-model\n"
    )

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

    def fake_transformers(prompt, model, max_new_tokens, temperature):
        return f"{prompt}\n\n{json.dumps(expected_payload)}"

    with patch_tokenizer():
        with patch("src.classify.classify.call_transformers_inference", fake_transformers):
            result = classify_document(metadata, text, env_path=env_dir)

    assert result.classification == "Green"
    assert result.confidence == "high"
    assert result.rationale == "Document appears non-sensitive."


def test_local_transformers_inference_uses_model_generate_and_decodes_prompt(tmp_path):
    env_dir = tmp_path / ".env"
    env_dir.mkdir()
    (env_dir / ".env.example").write_text(
        "HF_INFERENCE_BACKEND=local\nHF_MODEL_NAME=test-model\n"
    )

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
        "rationale": "Local model returned JSON.",
    }

    class DummyTensor:
        def __init__(self, data):
            self.data = data

        def to(self, device):
            return self

    class DummyTorch:
        @staticmethod
        def tensor(value):
            return DummyTensor(value)

        @staticmethod
        def device(value):
            return f"device:{value}"

        @staticmethod
        def no_grad():
            from contextlib import nullcontext

            return nullcontext()

        class cuda:
            @staticmethod
            def is_available():
                return False

    class DummyTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def __call__(self, prompt, return_tensors="pt"):
            return {
                "input_ids": DummyTensor([[1, 2, 3]]),
                "attention_mask": DummyTensor([[1, 1, 1]]),
            }

        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, token_ids, skip_special_tokens=True):
            return f"PROMPT\n\n{json.dumps(expected_payload)}"

    class DummyModel:
        def to(self, device):
            return self

        def eval(self):
            return None

        def generate(
            self,
            input_ids,
            attention_mask=None,
            max_new_tokens=None,
            temperature=None,
            do_sample=None,
            pad_token_id=None,
            eos_token_id=None,
            use_cache=None,
        ):
            return [[1, 2, 3]]

    fake_torch = DummyTorch()

    with patch.dict("src.classify.classify._TRANSFORMERS_TOKENIZER_CACHE", {}, clear=True):
        with patch.dict("src.classify.classify._TRANSFORMERS_MODEL_CACHE", {}, clear=True):
            with patch("src.classify.classify.AutoTokenizer", SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyTokenizer())):
                with patch("src.classify.classify.AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda *args, **kwargs: DummyModel())):
                    with patch("src.classify.classify.torch", fake_torch):
                        result = classify_document(metadata, text, env_path=env_dir)

    assert result.classification == "Green"
    assert result.confidence == "high"
    assert result.rationale == "Local model returned JSON."


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

    with patch_tokenizer():
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

    with patch_tokenizer():
        with patch("src.classify.classify.InferenceClient", DummyClient):
            with pytest.raises(ClassificationError):
                classify_document(metadata, text, env_path=env_dir)
