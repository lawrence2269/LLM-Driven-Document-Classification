from pathlib import Path

from src.classify.classify import (
    DEFAULT_POLICY_PATH,
    build_classification_prompts,
    build_metadata_summary,
    build_system_prompt,
    build_user_prompt,
    load_policy_text,
)
from src.metadata.metadata import DocumentMetadata


def test_load_policy_text_loads_policy_file():
    policy_text = load_policy_text(DEFAULT_POLICY_PATH)
    assert "Classification Policy" in policy_text
    assert "Green" in policy_text
    assert "Red" in policy_text


def test_build_system_prompt_contains_role_and_policy():
    policy_text = "Policy line 1\nPolicy line 2"
    system_prompt = build_system_prompt(policy_text)

    assert "expert document classification assistant" in system_prompt
    assert "Policy:" in system_prompt
    assert "Policy line 1" in system_prompt
    assert "Policy line 2" in system_prompt
    assert "classification` must be either `Green` or `Red`" in system_prompt


def test_build_metadata_summary_includes_metadata_fields():
    metadata = DocumentMetadata(
        filename="doc.pdf",
        size_bytes=1234,
        created_timestamp=1.0,
        modified_timestamp=2.0,
        sha256_hash="abc123",
        language="en",
        word_count=10,
        pdf_author="Author Name",
        pdf_title="Title"
    )
    summary = build_metadata_summary(metadata)

    assert "Filename: doc.pdf" in summary
    assert "Size (bytes): 1234" in summary
    assert "Language: en" in summary
    assert "PDF author: Author Name" in summary
    assert "PDF title: Title" in summary


def test_build_user_prompt_includes_metadata_and_text_preview():
    metadata = DocumentMetadata(
        filename="doc.pdf",
        size_bytes=1234,
        created_timestamp=1.0,
        modified_timestamp=2.0,
        sha256_hash="abc123",
        language="en",
        word_count=3,
        pdf_author=None,
        pdf_title=None,
    )
    text = "This is a short document text for prompt generation."
    user_prompt = build_user_prompt(metadata, text, max_text_chars=100)

    assert "Document metadata summary:" in user_prompt
    assert "doc.pdf" in user_prompt
    assert "This is a short document text" in user_prompt
    assert "valid JSON only" in user_prompt


def test_build_classification_prompts_are_within_token_budget():
    metadata = DocumentMetadata(
        filename="doc.pdf",
        size_bytes=1234,
        created_timestamp=1.0,
        modified_timestamp=2.0,
        sha256_hash="abc123",
        language="en",
        word_count=300,
        pdf_author="Author Name",
        pdf_title="Title"
    )
    text = "word " * 3000
    system_prompt, user_prompt = build_classification_prompts(metadata, text)
    combined_length = len(system_prompt) + len(user_prompt)

    assert combined_length < 16000
    assert "classification" in user_prompt
    assert "confidence" in user_prompt
    assert "rationale" in user_prompt


def test_build_classification_prompts_raises_when_prompt_too_large():
    metadata = DocumentMetadata(
        filename="doc.pdf",
        size_bytes=1234,
        created_timestamp=1.0,
        modified_timestamp=2.0,
        sha256_hash="abc123",
        language="en",
        word_count=300,
        pdf_author="Author Name",
        pdf_title="Title"
    )
    text = "x" * 20000

    try:
        system_prompt, user_prompt = build_classification_prompts(metadata, text, max_text_chars=20000)
    except ValueError as exc:
        assert "maximum token budget" in str(exc)
    else:
        assert False, "Expected ValueError for oversized prompt"
