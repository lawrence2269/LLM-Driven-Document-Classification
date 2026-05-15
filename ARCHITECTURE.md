# Architecture — LLM-Based Document Classification POC

## Overview

A lightweight pipeline that ingests unclassified documents, extracts metadata, calls an LLM to classify each document against the existing classification policy, and physically moves the file into the appropriate output folder (`G` or `R`), while persisting a metadata record alongside it.

---

## Components

```
┌──────────────────────────────────────────────────────────┐
│                        INPUT                             │
│          /inbox   (10–20 unclassified documents)         │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│               1. Document Ingestion Layer                │
│  - Walk the inbox folder                                 │
│  - Detect file type (PDF, DOCX, TXT, …)                 │
│  - Extract raw text content                              │
│  - Compute file hash (SHA-256) for deduplication        │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│               2. Metadata Extraction Layer               │
│  - File name, size, extension, created/modified dates    │
│  - Author, title (from DOCX/PDF properties)              │
│  - Word count, language detection                        │
│  - SHA-256 hash                                          │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              3. LLM Classification Layer                 │
│  - Build prompt: document text + classification policy   │
│  - Call LLM API (Claude claude-sonnet-4-20250514)               │
│  - Parse structured response: label (Green/Red) +        │
│    confidence + short rationale                          │
│  - Retry / fallback on API errors                        │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              4. Decision & Routing Layer                 │
│  - Green  →  /G/<original_filename>                      │
│  - Red    →  /R/<original_filename>                      │
│  - Write sidecar metadata JSON next to each moved file   │
│    e.g. /G/report.pdf  +  /G/report.pdf.meta.json        │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              5. Audit Log / Run Report                   │
│  - Append one record per document to run_log.jsonl       │
│  - Print summary table to stdout (file, label, reason)   │
└──────────────────────────────────────────────────────────┘
```

---

## Folder Structure

```
project/
├── inbox/                   # Drop unclassified documents here
├── G/                       # Green documents land here
├── R/                       # Red documents land here
├── logs/
│   └── run_log.jsonl        # Append-only audit log
├── policy/
│   └── classification_policy.md   # Human-readable policy consumed by prompt
├── src/
│   ├── ingest.py            # File walking & text extraction
│   ├── metadata.py          # Metadata extraction helpers
│   ├── classify.py          # LLM prompt construction & API call
│   ├── router.py            # Move file + write sidecar JSON
│   ├── logger.py            # Audit log writer
│   └── main.py              # Orchestrator / entry point
├── tests/
│   └── test_classify.py
├── .env                     # ANTHROPIC_API_KEY (not committed)
├── requirements.txt
└── README.md
```

---

## Technology Choices

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.11+ | Rich ecosystem for file handling & LLM SDKs |
| LLM | Claude (claude-sonnet-4-20250514) via Anthropic SDK | Accurate instruction-following, structured output |
| PDF text extraction | `pymupdf` (fitz) | Fast, reliable, no Java dependency |
| DOCX extraction | `python-docx` | Native DOCX support |
| Metadata schema | JSON sidecar file | No database needed for POC; easy to evolve |
| Env management | `python-dotenv` | Keeps secrets out of code |

---

## Data Flow — Metadata Sidecar Schema

Each processed file produces a `.meta.json` alongside it:

```json
{
  "original_filename": "report_q1.pdf",
  "sha256": "a3f2...",
  "file_size_bytes": 204800,
  "created_at": "2024-11-01T09:00:00Z",
  "modified_at": "2024-11-15T14:22:00Z",
  "author": "Jane Doe",
  "word_count": 1523,
  "language": "en",
  "classification": "Green",
  "confidence": "high",
  "rationale": "Document contains only public-facing marketing content with no sensitive data.",
  "classified_at": "2025-05-15T10:05:33Z",
  "model": "claude-sonnet-4-20250514"
}
```

---

## Scalability Notes

- The POC uses a flat folder design; the metadata sidecar pattern is the foundation for migrating to a proper document management system or object store later.
- The classification policy is kept in a separate Markdown file so non-technical stakeholders can update it without touching code.
- The LLM prompt includes the full policy text, making the system policy-driven by design.
