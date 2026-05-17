"""
classify.py — LLM-based document classification using local Transformers inference.

Loads GPT-SW3 (or any causal LM) directly via the Transformers library.
No API calls, no network access required after the model is cached locally.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metadata.metadata import DocumentMetadata


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME   = "AI-Sweden-Models/gpt-sw3-126m"
DEFAULT_POLICY_PATH  = Path("policy/classification_policy.md")

# Token budget constants.
# MAX_PROMPT_TOKENS is a safe default but build_classification_prompt will
# tighten it further based on the model's actual max_position_embeddings.
MAX_PROMPT_TOKENS    = 1_500
MAX_NEW_TOKENS       = 150

# Hard ceiling: never feed more than this many tokens regardless of model config.
# GPT-SW3-126m has a 2 048-token context window.
_MODEL_MAX_CONTEXT   = 2_048

# JSON keys / valid values
VALID_CLASSIFICATIONS = {"green", "red"}
VALID_CONFIDENCES     = {"high", "medium", "low"}

# Prefix that ends the prompt — the model continues from here.
JSON_OUTPUT_PREFIX = '{"classification":'


# ---------------------------------------------------------------------------
# Module-level model cache (loaded once per process)
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: Dict[str, AutoTokenizer]          = {}
_MODEL_CACHE:     Dict[str, AutoModelForCausalLM]   = {}


# ---------------------------------------------------------------------------
# Exceptions & result type
# ---------------------------------------------------------------------------

class ClassificationError(Exception):
    """Raised when classification fails or the model output cannot be parsed."""


@dataclass
class ClassificationResult:
    classification: str   # "Green" or "Red"
    confidence:     str   # "high", "medium", or "low"
    rationale:      str   # plain-English explanation


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

def _load_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """
    Load (or return cached) tokenizer and model, placed on the best available device.
    On first call this downloads weights from HuggingFace Hub and caches them locally.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name not in _TOKENIZER_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _TOKENIZER_CACHE[model_name] = tokenizer

    if model_name not in _MODEL_CACHE:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()
        _MODEL_CACHE[model_name] = model

    return _TOKENIZER_CACHE[model_name], _MODEL_CACHE[model_name], device


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy_text(policy_path: Union[str, Path] = DEFAULT_POLICY_PATH) -> str:
    """Read the plain-text classification policy from disk."""
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Metadata summary
# ---------------------------------------------------------------------------

def _normalize_metadata(
    metadata: Union[DocumentMetadata, Mapping[str, Any]],
) -> Dict[str, Any]:
    if isinstance(metadata, DocumentMetadata):
        return asdict(metadata)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    raise TypeError("metadata must be a DocumentMetadata instance or a plain mapping.")


def build_metadata_summary(
    metadata: Union[DocumentMetadata, Mapping[str, Any]],
) -> str:
    """Return a concise, human-readable metadata block for inclusion in the prompt."""
    data = _normalize_metadata(metadata)
    fields = [
        ("filename",           "Filename"),
        ("size_bytes",         "Size (bytes)"),
        ("created_timestamp",  "Created"),
        ("modified_timestamp", "Modified"),
        ("language",           "Language"),
        ("word_count",         "Word count"),
        ("pdf_author",         "Author"),
        ("pdf_title",          "Title"),
    ]
    lines = ["Document metadata:"]
    for key, label in fields:
        value = data.get(key)
        if value is not None and str(value).strip():
            lines.append(f"  {label}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token-aware text truncation (T-05)
# ---------------------------------------------------------------------------

def _truncate_to_token_budget(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> str:
    """
    Encode *text*, slice to *max_tokens* tokens, then decode back to a string.
    This is token-accurate truncation — not character-based — which matters
    for multilingual models like GPT-SW3.
    """
    if max_tokens <= 0:
        return "[TRUNCATED]"

    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text

    truncated = tokenizer.decode(
        ids[:max_tokens],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    return f"{truncated}\n\n[TRUNCATED]"


# ---------------------------------------------------------------------------
# Prompt building (T-05)
# ---------------------------------------------------------------------------

def build_classification_prompt(
    metadata: Union[DocumentMetadata, Mapping[str, Any]],
    text: str,
    policy_path: Union[str, Path] = DEFAULT_POLICY_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> str:
    """
    Build a single causal-LM completion prompt for GPT-SW3.

    GPT-SW3 is a decoder-only model with no system/user role tokens.
    The entire context is one flat string; the model continues it.

    The prompt ends with the JSON output prefix  {"classification":
    so the model's first generated token is the value of that key,
    steering it toward valid JSON rather than free prose.

    Token budget strategy
    ---------------------
    1. Measure the fixed overhead (policy + metadata + framing text).
    2. Whatever tokens remain go to the document body.
    3. If even the overhead exceeds the budget, raise ValueError.
    """
    tokenizer, model, _ = _load_model(model_name)
    policy_text          = load_policy_text(policy_path)
    metadata_summary     = build_metadata_summary(metadata)

    # Derive the hard ceiling from the model's own config so we never exceed
    # the context window regardless of what MAX_PROMPT_TOKENS is set to.
    # prompt tokens + generated tokens must fit within max_position_embeddings.
    model_max_ctx = getattr(model.config, "max_position_embeddings", _MODEL_MAX_CONTEXT)
    hard_ceiling  = model_max_ctx - MAX_NEW_TOKENS - 10   # 10-token safety margin
    effective_max = min(max_prompt_tokens, hard_ceiling)

    # Fixed parts of the prompt (policy, metadata, instructions).
    prefix = (
        "You are a document classification assistant.\n\n"
        "Classification policy:\n"
        f"{policy_text}\n\n"
        f"{metadata_summary}\n\n"
        "Document text:\n"
    )

    suffix = (
        "\n\n"
        "Classify this document according to the policy above.\n"
        "Respond with valid JSON only. Use exactly these keys:\n"
        "  classification: Green or Red\n"
        "  confidence: high, medium, or low\n"
        "  rationale: one concise sentence\n"
        "If the document is ambiguous, default to Red.\n\n"
        f"{JSON_OUTPUT_PREFIX}"
    )

    # Count tokens used by the fixed parts.
    fixed_tokens = len(tokenizer.encode(prefix + suffix, add_special_tokens=False))
    available_for_body = effective_max - fixed_tokens

    if available_for_body <= 0:
        raise ValueError(
            "Policy + metadata already exceed the prompt token budget. "
            "Shorten the policy file or increase MAX_PROMPT_TOKENS."
        )

    # Truncate document body to whatever token budget remains.
    body = _truncate_to_token_budget(text, tokenizer, available_for_body)
    prompt = prefix + body + suffix

    # The assembled prompt may exceed the budget by a handful of tokens because
    # the tokenizer encodes prefix+body+suffix as a single string slightly
    # differently than the parts measured individually (e.g. whitespace at
    # boundaries merges into fewer or more tokens than expected).
    # Rather than hard-failing, re-truncate the body against the actual overage.
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual_tokens > effective_max:
        overage = actual_tokens - effective_max
        adjusted_budget = max(available_for_body - overage, 0)
        if adjusted_budget <= 0:
            raise ValueError(
                "Policy + metadata already exceed the prompt token budget "
                "after adjustment. Shorten the policy file or increase MAX_PROMPT_TOKENS."
            )
        body = _truncate_to_token_budget(text, tokenizer, adjusted_budget)
        prompt = prefix + body + suffix

    return prompt


# ---------------------------------------------------------------------------
# Local inference (T-06)
# ---------------------------------------------------------------------------

def _run_inference(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Tokenize the prompt, run model.generate(), then decode ONLY the newly
    generated token ids by slicing off the input ids before decoding.

    This is the correct way to isolate generated text from causal LMs —
    slicing the output tensor is exact and never fails, unlike string-based
    prompt stripping which breaks when the tokenizer round-trip alters
    whitespace or special characters.
    """
    tokenizer, model, device = _load_model(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    n_input_tokens = input_ids.shape[1]

    # GPT-SW3 may not define pad_token_id; fall back to eos_token_id.
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    if pad_token_id is None:
        raise ClassificationError(
            "Tokenizer defines neither pad_token_id nor eos_token_id. "
            "Cannot run generation safely."
        )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy — deterministic output
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Slice off the input tokens — decode only what the model generated.
    # This is exact and immune to tokenizer round-trip whitespace differences.
    new_token_ids = output_ids[0][n_input_tokens:]

    if new_token_ids.shape[0] == 0:
        raise ClassificationError(
            "Model generated zero new tokens. "
            "The prompt may be at or over the model's context limit."
        )

    generated = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return generated



# ---------------------------------------------------------------------------
# Response parsing (T-06)
# ---------------------------------------------------------------------------

def _repair_json(text: str) -> str:
    """
    Apply lightweight heuristic repairs to text that is almost-but-not-quite
    valid JSON, covering the most common GPT-SW3 output quirks:

    - Trailing comma before closing brace  {"a": 1,}  →  {"a": 1}
    - Missing closing brace (truncated output)
    - Single-quoted strings  {'key': 'val'}  →  {"key": "val"}
    """
    # Single quotes → double quotes (only outside already-valid double-quoted regions).
    text = re.sub(r"(?<![\\])'", '"', text)

    # Trailing commas before } or ].
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # If there is an opening { but no closing }, append one.
    if text.count("{") > text.count("}"):
        text = text + "}"

    return text


def _extract_json_from_generated(generated_text: str) -> str:
    """
    Extract a JSON object from the model's generated continuation.

    The model was seeded with  {"classification":  as the last tokens of the
    prompt, so the continuation is everything *after* that prefix.
    We reconstruct the full object, then try progressively looser strategies
    until one yields parseable JSON.

    Strategies (in order):
    1. Prepend prefix + parse the whole continuation.
    2. Greedy regex to find the outermost {...} block (handles surrounding prose).
    3. Apply _repair_json to the best candidate and retry.
    4. Field-by-field regex extraction as a last resort.
    """
    # Re-attach the JSON prefix the model was seeded with.
    candidate = JSON_OUTPUT_PREFIX + generated_text

    # --- Strategy 1: parse the whole candidate string ---
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    # --- Strategy 2: greedy outermost {...} block ---
    # Use a greedy (not non-greedy) match so we capture the full object,
    # not just up to the first closing brace.
    brace_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if brace_match:
        block = brace_match.group(0)
        try:
            json.loads(block)
            return block
        except json.JSONDecodeError:
            pass

        # --- Strategy 3: repair and retry ---
        repaired = _repair_json(block)
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

    # --- Strategy 4: field-by-field regex extraction ---
    # If the model produced valid field values but malformed JSON structure,
    # reconstruct a clean object from individually captured values.
    cls_match  = re.search(r'"?classification"?\s*:\s*"?(\w+)"?', candidate, re.IGNORECASE)
    conf_match = re.search(r'"?confidence"?\s*:\s*"?(\w+)"?',     candidate, re.IGNORECASE)
    rat_match  = re.search(r'"?rationale"?\s*:\s*"([^"]*)"',       candidate, re.IGNORECASE)

    if cls_match and conf_match and rat_match:
        reconstructed = json.dumps({
            "classification": cls_match.group(1).strip(),
            "confidence":     conf_match.group(1).strip(),
            "rationale":      rat_match.group(1).strip(),
        })
        return reconstructed

    raise ClassificationError(
        "Could not extract a JSON object from the model output.\n"
        f"Raw generated text: {generated_text!r}"
    )


def parse_classification_response(generated_text: str) -> ClassificationResult:
    """Parse and validate the model's generated text into a ClassificationResult."""
    payload = _extract_json_from_generated(generated_text)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ClassificationError(
            f"Model output is not valid JSON.\nPayload: {payload!r}\n"
            f"Raw generated text: {generated_text!r}"
        ) from exc

    # --- classification ---
    raw_cls = data.get("classification")
    if raw_cls is None:
        raise ClassificationError("'classification' key missing from model output.")
    if str(raw_cls).strip().lower() not in VALID_CLASSIFICATIONS:
        raise ClassificationError(
            f"Invalid classification value: {raw_cls!r}. Expected 'Green' or 'Red'."
        )
    classification = str(raw_cls).strip().capitalize()

    # --- confidence ---
    # Cast to str first — the model occasionally emits an integer (e.g. 1, 0).
    raw_conf = str(data.get("confidence", "")).strip().lower()
    if not raw_conf or raw_conf not in VALID_CONFIDENCES:
        confidence = "low"
    else:
        confidence = raw_conf

    # --- rationale ---
    # Cast to str — guard against the model emitting a non-string value.
    rationale = str(data.get("rationale", "")).strip()
    if not rationale:
        rationale = f"Document classified as {classification} by model."

    return ClassificationResult(
        classification=classification,
        confidence=confidence,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_document(
    metadata: Union[DocumentMetadata, Mapping[str, Any]],
    text: str,
    policy_path: Union[str, Path] = DEFAULT_POLICY_PATH,
    model_name: str = DEFAULT_MODEL_NAME,
    max_new_tokens: int = MAX_NEW_TOKENS,
    max_attempts: int = 2,
) -> ClassificationResult:
    """
    Classify a document using local Transformers inference.

    Parameters
    ----------
    metadata      : DocumentMetadata or plain dict with file / document attributes.
    text          : Extracted plain text of the document.
    policy_path   : Path to the classification policy Markdown file.
    model_name    : HuggingFace model identifier (default: GPT-SW3 126 M).
    max_new_tokens: Maximum tokens the model may generate per attempt.
    max_attempts  : Number of retry attempts if JSON parsing fails.

    Returns
    -------
    ClassificationResult with classification, confidence, and rationale.

    Raises
    ------
    ClassificationError if all attempts fail to produce parseable output.
    """
    prompt = build_classification_prompt(
        metadata,
        text,
        policy_path=policy_path,
        model_name=model_name,
    )

    retry_note = (
        "\n\nIMPORTANT: Your previous response was not valid JSON. "
        "Return only a JSON object with keys: classification, confidence, rationale."
    )

    last_error: Optional[ClassificationError] = None

    for attempt in range(1, max_attempts + 1):
        current_prompt = prompt if attempt == 1 else prompt + retry_note

        try:
            generated = _run_inference(
                current_prompt,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
            )
            return parse_classification_response(generated)

        except ClassificationError as exc:
            last_error = exc
            if attempt < max_attempts:
                continue  # retry with the amended prompt

    root_cause = str(last_error) if last_error and str(last_error) else repr(last_error)
    raise ClassificationError(
        f"Classification failed after {max_attempts} attempt(s). "
        f"Root cause: {root_cause}"
    ) from last_error
