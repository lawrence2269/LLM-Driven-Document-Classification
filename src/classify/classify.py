from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from dotenv import dotenv_values
from huggingface_hub import InferenceClient

from src.metadata.metadata import DocumentMetadata


DEFAULT_POLICY_PATH = Path("policy/classification_policy.md")
DEFAULT_ENV_PATH = Path(".env")
DEFAULT_ENV_EXAMPLE_PATH = Path(".env.example")
MAX_TEXT_PREVIEW_CHARS = 12000
TOKEN_CHAR_RATIO = 4.0
MAX_TOKENS = 4000
MAX_PROMPT_CHARS = int(MAX_TOKENS * TOKEN_CHAR_RATIO)
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
VALID_CLASSIFICATIONS = {"green", "red"}
VALID_CONFIDENCES = {"high", "medium", "low"}


class ClassificationError(Exception):
    pass


@dataclass
class ClassificationResult:
    classification: str
    confidence: str
    rationale: str


def load_policy_text(policy_path: Union[str, Path] = DEFAULT_POLICY_PATH) -> str:
    policy_file = Path(policy_path)
    if not policy_file.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_file}")
    return policy_file.read_text(encoding="utf-8")


def build_system_prompt(policy_text: str) -> str:
    return (
        "You are an expert document classification assistant. "
        "Classify documents according to the policy provided below. "
        "When answering, obey the JSON output format exactly and do not add any extra text.\n\n"
        "Policy:\n"
        f"{policy_text.strip()}\n\n"
        "Output requirements:\n"
        "Return valid JSON only with the following keys:"
        " `classification`, `confidence`, and `rationale`.\n"
        "`classification` must be either `Green` or `Red`.\n"
        "`confidence` must be one of `high`, `medium`, or `low`.\n"
        "`rationale` must be a brief explanation in plain English.\n"
        "If the document is ambiguous, default to Red."
    )


def _normalize_metadata(metadata: Union[DocumentMetadata, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(metadata, DocumentMetadata):
        return asdict(metadata)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    raise TypeError("metadata must be DocumentMetadata or a mapping")


def build_metadata_summary(metadata: Union[DocumentMetadata, Mapping[str, Any]]) -> str:
    data = _normalize_metadata(metadata)

    fields = [
        ("filename", "Filename"),
        ("size_bytes", "Size (bytes)"),
        ("created_timestamp", "Created timestamp"),
        ("modified_timestamp", "Modified timestamp"),
        ("sha256_hash", "SHA-256 hash"),
        ("language", "Language"),
        ("word_count", "Word count"),
        ("pdf_author", "PDF author"),
        ("pdf_title", "PDF title"),
    ]

    lines = ["Document metadata summary:"]
    for key, label in fields:
        value = data.get(key)
        if value is None:
            value = ""
        lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def _truncate_text_for_prompt(text: str, max_chars: int = MAX_TEXT_PREVIEW_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rstrip()
    return f"{truncated}\n\n[TRUNCATED]"


def build_user_prompt(
    metadata: Union[DocumentMetadata, Mapping[str, Any]],
    text: str,
    max_text_chars: int = MAX_TEXT_PREVIEW_CHARS,
) -> str:
    metadata_summary = build_metadata_summary(metadata)
    text_preview = _truncate_text_for_prompt(text, max_text_chars)

    return (
        f"{metadata_summary}\n\n"
        "Document text preview (first characters shown):\n"
        f"{text_preview}\n\n"
        "Classify this document using the policy provided in the system prompt. "
        "Respond with valid JSON only, using exactly the keys: `classification`, "
        "`confidence`, and `rationale`.\n"
        "`classification` must be either `Green` or `Red`. "
        "`confidence` must be one of `high`, `medium`, or `low`. "
        "`rationale` should be a concise explanation."
    )


def build_classification_prompts(
    metadata: Union[DocumentMetadata, Mapping[str, Any]],
    text: str,
    policy_path: Union[str, Path] = DEFAULT_POLICY_PATH,
    max_text_chars: int = MAX_TEXT_PREVIEW_CHARS,
) -> tuple[str, str]:
    policy_text = load_policy_text(policy_path)
    system_prompt = build_system_prompt(policy_text)
    user_prompt = build_user_prompt(metadata, text, max_text_chars)

    if len(system_prompt) + len(user_prompt) > MAX_PROMPT_CHARS:
        raise ValueError(
            "Built prompt exceeds the maximum token budget. "
            f"Prompt length is {len(system_prompt) + len(user_prompt)} chars."
        )

    return system_prompt, user_prompt


def load_env_config(
    env_path: Union[str, Path] = DEFAULT_ENV_PATH,
    example_path: Union[str, Path] = DEFAULT_ENV_EXAMPLE_PATH,
) -> Dict[str, str]:
    env_path = Path(env_path)
    example_path = Path(example_path)

    if env_path.is_dir():
        env_file = env_path / ".env"
        example_file = env_path / ".env.example"
    else:
        env_file = env_path
        if example_path.is_dir():
            example_file = example_path / ".env.example"
        else:
            example_file = example_path if example_path.is_file() else env_path.parent / ".env.example"

    base_config = dotenv_values(example_file) if example_file.is_file() else {}
    env_config = dotenv_values(env_file) if env_file.is_file() else {}
    merged = {**base_config, **env_config}

    # Environment variables take precedence if present.
    for key, value in os.environ.items():
        if value is not None:
            merged[key] = value

    return {k: v for k, v in merged.items() if v is not None}


def load_hf_credentials(
    env_path: Union[str, Path] = DEFAULT_ENV_PATH,
    example_path: Union[str, Path] = DEFAULT_ENV_EXAMPLE_PATH,
) -> Tuple[str, str]:
    config = load_env_config(env_path, example_path)
    token = config.get("HF_API_TOKEN", "")
    model = config.get("HF_MODEL_NAME", "")

    if not token:
        raise ValueError(
            "HF_API_TOKEN must be set in .env/.env.example before calling the HuggingFace API."
        )
    if not model:
        raise ValueError(
            "HF_MODEL_NAME must be set in .env/.env.example before calling the HuggingFace API."
        )

    return token, model


def _normalize_inference_response(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Iterable):
        response_list = list(response)
        if len(response_list) == 0:
            return ""
        first = response_list[-1]
        if isinstance(first, str):
            return first
        if hasattr(first, "generated_text"):
            return getattr(first, "generated_text")
    if hasattr(response, "generated_text"):
        return getattr(response, "generated_text")
    return str(response)


def _strip_prompt_echo(response_text: str, prompt: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith(prompt):
        return stripped[len(prompt) :].strip()
    if prompt in stripped:
        return stripped.split(prompt, 1)[1].strip()
    return stripped


def _extract_json_payload(response_text: str) -> str:
    try:
        json.loads(response_text)
        return response_text
    except json.JSONDecodeError:
        pass

    matches = re.findall(r"\{.*?\}", response_text, flags=re.DOTALL)
    if matches:
        return matches[-1]
    raise ClassificationError("Could not extract JSON object from model response.")


def _normalize_classification_value(value: Any, valid_values: set[str], default: Optional[str] = None) -> str:
    if value is None:
        raise ClassificationError("Missing classification field in model response.")
    normalized = str(value).strip()
    if normalized.lower() in valid_values:
        return normalized.lower()
    raise ClassificationError(
        f"Invalid value '{normalized}' for classification field."
    )


def _normalize_confidence_value(value: Any) -> str:
    if value is None:
        raise ClassificationError("Missing confidence field in model response.")
    normalized = str(value).strip().lower()
    if normalized in VALID_CONFIDENCES:
        return normalized
    raise ClassificationError(f"Invalid confidence value '{normalized}'.")


def parse_classification_response(response_text: str) -> ClassificationResult:
    payload = _extract_json_payload(response_text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ClassificationError(
            "Model response could not be parsed as JSON."
        ) from exc

    classification = _normalize_classification_value(data.get("classification"), VALID_CLASSIFICATIONS)
    confidence = _normalize_confidence_value(data.get("confidence"))
    rationale = data.get("rationale")
    if rationale is None or not str(rationale).strip():
        raise ClassificationError("Missing or empty rationale field in model response.")

    return ClassificationResult(
        classification=classification.capitalize(),
        confidence=confidence,
        rationale=str(rationale).strip(),
    )


def call_hf_inference(
    client: InferenceClient,
    prompt: str,
    model: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    response = client.text_generation(
        prompt,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_full_text=True,
    )
    return _normalize_inference_response(response)


def classify_document(
    metadata: Union[DocumentMetadata, Mapping[str, Any]],
    text: str,
    env_path: Union[str, Path] = DEFAULT_ENV_PATH,
    example_path: Union[str, Path] = DEFAULT_ENV_EXAMPLE_PATH,
    max_text_chars: int = MAX_TEXT_PREVIEW_CHARS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_attempts: int = 2,
) -> ClassificationResult:
    system_prompt, user_prompt = build_classification_prompts(
        metadata, text, max_text_chars=max_text_chars
    )
    prompt = f"{system_prompt}\n\n{user_prompt}"
    token, model = load_hf_credentials(env_path, example_path)

    client = InferenceClient(token=token)
    last_error: Optional[ClassificationError] = None
    for attempt in range(1, max_attempts + 1):
        response_text = call_hf_inference(client, prompt, model, max_new_tokens, temperature)
        stripped = _strip_prompt_echo(response_text, prompt)
        try:
            return parse_classification_response(stripped)
        except ClassificationError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise ClassificationError(
                    "Failed to parse classification response after multiple attempts."
                ) from exc
    assert last_error is not None
    raise last_error
