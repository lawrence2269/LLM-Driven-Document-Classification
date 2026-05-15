# Tasks: LLM Document Classification POC

Tracks all work items for the Proof of Concept. Each task includes scope, acceptance criteria, and dependencies.

---

## Phase 0 — Setup & Foundations

### T-001 · Repository & Project Scaffold
**Goal:** Initialize a clean, reproducible project structure.
- [ ] Create Git repository with `.gitignore`, `README.md`
- [ ] Define folder structure: `ingestion/`, `parsing/`, `classification/`, `routing/`, `db/`, `dashboard/`, `tests/`
- [ ] Set up `pyproject.toml` / `requirements.txt` with pinned dependencies
- [ ] Add `docker-compose.yml` with services: app, redis, postgres (or sqlite for POC)
- [ ] Configure environment variable management (`.env` + `python-dotenv`)

**Acceptance:** `docker compose up` starts all services without errors.

---

### T-002 · Classification Policy Document
**Goal:** Formalize what makes a document Green vs. Red.
- [ ] Interview stakeholders or review existing policy documentation
- [ ] Define clear, unambiguous rules for Green and Red classification
- [ ] Document edge cases and examples for each class
- [ ] Translate policy into an LLM system prompt / rubric

**Acceptance:** A written policy document (`POLICY.md`) that a non-technical reviewer can validate, and a corresponding LLM prompt template.

---

## Phase 1 — Ingestion Layer

### T-003 · Network Drive Crawler
**Goal:** Enumerate all documents on the shared drive.
- [ ] Implement recursive directory walker (handle symlinks, permissions errors gracefully)
- [ ] Filter by supported file types (PDF, DOCX, XLSX, TXT, images)
- [ ] Compute SHA-256 hash per file for deduplication
- [ ] Write discovered files into a `documents` state table with status `pending`
- [ ] Log skipped / unreadable files

**Acceptance:** Crawler runs against a test folder of 100 mixed files and populates the state table correctly with no duplicates.

---

### T-004 · Task Queue Setup
**Goal:** Enable parallel, resumable processing.
- [ ] Integrate Celery + Redis (or equivalent)
- [ ] Define a `process_document` task that accepts a document ID
- [ ] Implement retry logic with exponential backoff (max 3 retries)
- [ ] Ensure idempotency: re-queuing a completed document does nothing

**Acceptance:** 50 tasks dispatched concurrently complete correctly; a simulated failure retries and eventually succeeds.

---

## Phase 2 — Document Parsing

### T-005 · Multi-Format Text Extraction
**Goal:** Extract clean text from all supported document types.
- [ ] PDF: implement with `pdfplumber`; fall back to `PyMuPDF` for complex layouts
- [ ] DOCX: extract with `python-docx` (preserve headings and tables as text)
- [ ] XLSX: extract cell values with `openpyxl`
- [ ] TXT / HTML: direct read with encoding detection (`chardet`)
- [ ] Scanned PDF / image: OCR with `pytesseract` (or Azure Document Intelligence)
- [ ] Truncate/chunk text to fit LLM context window (configurable max tokens)

**Acceptance:** Extraction tests pass for at least one sample file of each supported type; character accuracy ≥ 95% on a test set.

---

### T-006 · File-Level Metadata Extraction
**Goal:** Capture structured filesystem metadata for every document.
- [ ] Extract: filename, extension, size (bytes), created/modified timestamps, MIME type, SHA-256 hash, original path
- [ ] Store in `metadata_json` column of the documents table

**Acceptance:** All fields populated for every processed file; no null values for mandatory fields.

---

## Phase 3 — LLM Classification

### T-007 · LLM Integration & Prompt Engineering
**Goal:** Classify each document using the LLM and the classification policy.
- [ ] Set up Anthropic Claude API client (configurable model, e.g. `claude-sonnet-4`)
- [ ] Build prompt template: system prompt (policy rubric) + user message (metadata + document excerpt)
- [ ] Parse structured JSON output: `{ "classification": "Green"|"Red", "confidence": float, "rationale": string }`
- [ ] Handle API errors, rate limits, and timeouts gracefully
- [ ] Log full prompt + response for each document (audit trail)

**Acceptance:** 20 hand-labeled test documents classified with ≥ 80% agreement against human ground truth.

---

### T-008 · Content-Level Metadata Extraction (LLM-assisted)
**Goal:** Extract rich, searchable metadata from document content.
- [ ] In the same LLM call (or a separate cheap call), extract:
  - Document type / category
  - Language
  - Key topics / named entities
  - Sensitivity signals (PII, confidential terms)
  - One-sentence summary
- [ ] Store as structured JSON in the metadata store

**Acceptance:** Metadata fields populated for all processed test documents; spot-check accuracy validated by a human reviewer.

---

### T-009 · Confidence Threshold & Human Review Queue
**Goal:** Catch uncertain classifications before they are acted upon.
- [ ] Define a configurable confidence threshold (default: 0.80)
- [ ] Documents below threshold → status set to `review`, moved to `/review/` folder
- [ ] Expose a simple review interface (CLI or Streamlit page) to accept/override decisions
- [ ] Overrides are logged and feed back into prompt refinement

**Acceptance:** All sub-threshold documents land in the review queue; a reviewer can approve or override via the UI; decision is persisted.

---

## Phase 4 — Routing & File Management

### T-010 · Document Routing (Move to G / R)
**Goal:** Physically move classified documents to the correct output folder.
- [ ] Move (or copy + verify + delete) each classified file to `/G/` or `/R/`
- [ ] Preserve original relative directory structure inside `G/` and `R/` (optional, configurable)
- [ ] Write a JSON sidecar file to `/metadata/<original_path>.json`
- [ ] Update document status to `done` in the state table
- [ ] Verify file integrity after move (hash comparison)

**Acceptance:** 100 test documents routed correctly; sidecar files created; no data loss (hash matches).

---

### T-011 · Metadata Store Persistence
**Goal:** Maintain a queryable record of all classified documents.
- [ ] Finalize DB schema (see `ARCHITECTURE.md`)
- [ ] Implement data access layer (ORM or raw SQL)
- [ ] Ensure atomic write: classification + routing happen in a single transaction
- [ ] Support querying by classification, date range, file type, confidence

**Acceptance:** All processed documents queryable; rollback on partial failure works correctly.

---

## Phase 5 — Monitoring & Dashboard

### T-012 · Progress & Statistics Dashboard (Streamlit POC)
**Goal:** Give stakeholders visibility into the pipeline.
- [ ] Show counters: total / processed / pending / failed / in review
- [ ] Breakdown by classification (G vs R) and file type
- [ ] Display recent classifications with rationale
- [ ] Show throughput over time (documents/hour)

**Acceptance:** Dashboard refreshes live and displays accurate numbers during a test run.

---

### T-013 · Audit Log & Reporting
**Goal:** Provide a complete, tamper-evident audit trail.
- [ ] Log every classification decision: document ID, timestamp, model version, prompt hash, output
- [ ] Export audit log as CSV/JSON on demand
- [ ] Generate a summary report: total classified, distribution (G/R), avg confidence, review rate

**Acceptance:** Audit log exported and verified to match the state table for a complete test run.

---

## Phase 6 — Testing & Validation

### T-014 · Unit & Integration Tests
- [ ] Unit tests for each parser (per file type)
- [ ] Unit tests for metadata extraction fields
- [ ] Integration test: end-to-end pipeline on a 50-document test set
- [ ] Mock LLM responses for deterministic testing

**Acceptance:** Test suite passes; coverage ≥ 70% on core modules.

---

### T-015 · POC Evaluation & Stakeholder Review
- [ ] Run pipeline on a representative sample (500–1000 real documents)
- [ ] Measure: throughput (docs/hour), classification accuracy, review rate, cost per document
- [ ] Present results to stakeholders
- [ ] Document findings, gaps, and recommendations for production rollout

**Acceptance:** Evaluation report delivered; go/no-go decision made for production phase.

---

## Task Summary

| ID | Task | Phase | Priority |
|---|---|---|---|
| T-001 | Repo & scaffold | 0 | High |
| T-002 | Classification policy | 0 | High |
| T-003 | Drive crawler | 1 | High |
| T-004 | Task queue | 1 | High |
| T-005 | Text extraction | 2 | High |
| T-006 | File metadata | 2 | Medium |
| T-007 | LLM classification | 3 | High |
| T-008 | Content metadata (LLM) | 3 | Medium |
| T-009 | Confidence & review queue | 3 | High |
| T-010 | Document routing | 4 | High |
| T-011 | Metadata store | 4 | Medium |
| T-012 | Dashboard | 5 | Low |
| T-013 | Audit log | 5 | Medium |
| T-014 | Tests | 6 | High |
| T-015 | POC evaluation | 6 | High |

---

## Dependencies

```
T-001 → T-003, T-004, T-005
T-002 → T-007
T-003 → T-004
T-005 → T-007, T-008
T-006 → T-008
T-007 → T-009, T-010
T-009 → T-010
T-010 → T-011
T-011 → T-013
T-012 → T-013
T-014 → T-015
```
