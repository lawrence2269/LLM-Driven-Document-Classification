# Architecture: LLM-Based Document Classification System

## Overview

A scalable, AI-driven pipeline that ingests documents from a shared network drive, extracts metadata, classifies each document as **Green** or **Red** using an LLM, and moves it to the corresponding folder (`G` or `R`) — while enriching it with structured metadata for future information management.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Shared Network Drive                     │
│                    (~1 million documents)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      1. Ingestion Layer                         │
│  - Crawl network drive                                          │
│  - Queue documents for processing (e.g. message queue)         │
│  - Track state: pending / in-progress / done / failed          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. Document Parsing Layer                     │
│  - Extract raw text (PDF, DOCX, XLSX, TXT, images via OCR…)    │
│  - Normalize encoding and language                             │
│  - Chunk large documents if needed                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  3. Metadata Extraction Layer                   │
│  - File-level: name, path, size, type, timestamps, author      │
│  - Content-level (LLM-assisted):                               │
│      • Document type / category                                │
│      • Language                                                │
│      • Key topics / entities                                   │
│      • Sensitivity signals                                     │
│      • Summary                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                4. LLM Classification Layer                      │
│  - Prompt LLM with document text + metadata                    │
│  - Apply classification policy rules                           │
│  - Output: Green (G) or Red (R) + confidence score + rationale │
│  - Low-confidence items → human review queue                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   5. Decision & Routing Layer                   │
│  - High confidence → auto-move to /G or /R                     │
│  - Low confidence → flag for manual review                     │
│  - Write metadata sidecar file (JSON) alongside document       │
│  - Update central metadata store / database                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   6. Metadata Store                             │
│  - Central database (e.g. PostgreSQL + vector index optional)  │
│  - Stores: file path, classification, confidence, metadata,    │
│    timestamps, audit trail                                     │
│  - Powers search, reporting, and future re-classification      │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               7. Monitoring & Review Dashboard                  │
│  - Progress tracking (processed / remaining / errors)          │
│  - Human review queue UI                                       │
│  - Audit log and classification statistics                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Ingestion Layer
- Recursive crawler over the network drive using a worker pool
- Deduplication via file hash (SHA-256)
- Distributed task queue (e.g. Celery + Redis, or AWS SQS) for parallelism
- State table to enable resumable processing and avoid reprocessing

### Document Parsing
| Format | Tool |
|---|---|
| PDF | `pdfplumber` / `PyMuPDF` |
| DOCX / XLSX | `python-docx` / `openpyxl` |
| Images / Scanned PDFs | Tesseract OCR / Azure Document Intelligence |
| Plain text / HTML | Direct extraction |

### Metadata Extraction
Metadata is extracted in two passes:
1. **Filesystem pass** — file attributes, MIME type, hash
2. **LLM pass** — structured JSON response from the LLM describing content-level metadata

### LLM Classification
- Model: configurable (e.g. Claude, GPT-4o, or a self-hosted model)
- Prompt includes: document excerpt, metadata, and the classification policy
- Output is a structured JSON: `{ "classification": "Green"|"Red", "confidence": 0.0–1.0, "rationale": "..." }`
- Confidence threshold (e.g. < 0.80) triggers human review

### Storage Layout (Filesystem)
```
/output/
  G/                        ← Green documents
  R/                        ← Red documents
  review/                   ← Low-confidence, awaiting human decision
  metadata/                 ← JSON sidecar files (mirroring original paths)
```

### Metadata Store Schema (simplified)
```sql
documents (
  id              UUID PRIMARY KEY,
  original_path   TEXT,
  dest_folder     CHAR(1),        -- 'G' or 'R'
  classification  TEXT,
  confidence      FLOAT,
  rationale       TEXT,
  file_hash       TEXT,
  file_type       TEXT,
  metadata_json   JSONB,
  status          TEXT,           -- pending | done | review | failed
  processed_at    TIMESTAMP
)
```

---

## Technology Stack (POC)

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Task Queue | Celery + Redis |
| Document Parsing | pdfplumber, python-docx, pytesseract |
| LLM | Anthropic Claude API (claude-sonnet-4) |
| Metadata Store | PostgreSQL (SQLite for POC) |
| Dashboard | Streamlit (POC) |
| Orchestration | Docker Compose (POC) |

---

## Security & Compliance Considerations
- Documents are never sent externally unless the LLM endpoint is approved — use an on-premise model or private cloud endpoint for sensitive data
- All classification decisions are logged with rationale for auditability
- Access to `/R` folder is restricted by filesystem ACLs
- PII detected during metadata extraction is flagged, not stored in plain text

---

## Scalability Path (Beyond POC)
- Horizontal scaling: increase worker count in the task queue
- Batch LLM calls to reduce cost and latency
- Replace SQLite with PostgreSQL + pgvector for semantic search over metadata
- Integrate with enterprise DMS (SharePoint, OpenText) via connector layer
