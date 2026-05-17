# Tasks — LLM-Based Document Classification POC

> Ordered by implementation sequence. Each task is self-contained and can be committed independently.

---

## Phase 1 — Project Setup

### T-01 · Repository & Environment Setup

- Initialise Git repo with `.gitignore` (exclude `.env`, `__pycache__`, `G/`, `R/`, `logs/`)
- Create `requirements.txt` with initial dependencies:
  - `transformers`, `torch`, `pymupdf`, `python-docx`, `python-dotenv`, `langdetect`
- Create `.env.example` (no API token required for local inference)
- Create folder skeleton: `inbox/`, `G/`, `R/`, `logs/`, `policy/`, `src/`, `tests/`
- **Done when:** `pip install -r requirements.txt` succeeds in a clean venv
- **Status:** ✅ Complete

---

### T-02 · Classification Policy File

- Write `policy/classification_policy.md` documenting the Green / Red rules
- Keep it plain English so the LLM can consume it directly in a prompt
- **Done when:** Policy clearly distinguishes what makes a document Green vs Red with examples
- **Status:** ✅ Complete

---

## Phase 2 — Document Ingestion & Metadata

### T-03 · File Walker (`src/ingest.py`)

- Recursively list files in `inbox/`
- Detect MIME type / extension
- Dispatch to correct extractor (PDF → PyMuPDF)
- Return `{path, text, extension}`
- **Done when:** All three file types yield non-empty text strings in a unit test
- **Status:** ✅ Complete

---

### T-04 · Metadata Extractor (`src/metadata.py`)

- Extract: filename, size, created/modified timestamps, SHA-256 hash
- For PDF: author, title from document properties
- Detect language with `langdetect`
- Count words in extracted text
- Return a typed `DocumentMetadata` dataclass
- **Done when:** Running against sample files produces a complete metadata dict with no `None` values for mandatory fields
- **Status:** ✅ Complete

---

## Phase 3 — LLM Classification

### T-05 · Prompt Builder & Tokenizer (`src/classify.py` — prompt section)

- Load policy text from `policy/classification_policy.md`
- Load tokenizer once at module level using `AutoTokenizer.from_pretrained("AI-Sweden-Models/gpt-sw3-126m")`
- Build a single completion-style prompt string (GPT-SW3 is a causal LM — no separate system/user roles):
  - Include policy rules, document metadata summary, and a truncated document body
- Truncate document body by encoding with the tokenizer and slicing to a safe token budget (e.g. max 1 500 tokens total), then decoding back to text — ensures truncation is token-accurate rather than character-based
- End the prompt with the expected JSON output prefix to steer the model's continuation: `{"classification":`
- **Done when:** Tokenized prompt length is verified to stay within the model's context window for all sample documents
- **Status:** ✅ Complete

---

### T-06 · Local Model Inference & Response Parsing (`src/classify.py` — call section)

- Load model once at module level using `AutoModelForCausalLM.from_pretrained()` with the model name obtained from (`HF_MODEL_NAME`) from .env.example file in .env directory or folder.
- Move model to appropriate device (`cuda` if available, else `cpu`)
- Tokenize the prompt with `tokenizer(prompt, return_tensors="pt")` and move input tensors to the same device
- Call `model.generate()` with appropriate parameters (e.g. `max_new_tokens=150`, `do_sample=False`)
- Decode the output with `tokenizer.decode()`, then strip the original prompt prefix from the decoded string to isolate only the newly generated text
- Parse JSON from the stripped output; handle malformed JSON with a retry using a slightly rephrased prompt suffix (up to 2 attempts)
- Raise a typed `ClassificationError` if both attempts fail
- **Done when:** Unit test with a mocked `model.generate()` response parses to correct dataclass fields

---

## Phase 4 — Routing & Output

### T-07 · File Router (`src/router.py`)

- Accept `(source_path, classification, metadata_dict)`
- Move file to `/G/` or `/R/` based on label (use `shutil.move`)
- Write sidecar `.meta.json` next to the moved file
- Handle filename collision: append `_1`, `_2`, … suffix
- **Done when:** Running against a test inbox moves files to correct folders and sidecar JSON is valid
- **Status:** ✅ Complete

---

### T-08 · Audit Logger (`src/logger.py`)

- Append one JSON line per document to `logs/run_log.jsonl`
- Fields: `run_id`, `timestamp`, `original_path`, `classification`, `confidence`, `rationale`, `model`, `sha256`
- **Done when:** After a run, `logs/run_log.jsonl` contains one record per processed file and is valid JSONL
- **Status:** ✅ Complete

---

## Phase 5 — Orchestration

### T-09 · Main Orchestrator (`src/main.py`)

- Wire together: ingest → metadata → classify → route → log
- Print a summary table to stdout on completion (filename | classification | confidence | rationale snippet)
- Accept optional `--inbox` and `--dry-run` CLI flags (`argparse`)
  - `--dry-run`: classify and log but do not move files
- **Done when:** `python src/main.py` successfully processes all files in `inbox/` end-to-end

---

## Phase 6 — Testing & Hardening

### T-10 · Unit Tests (`tests/test_classify.py`)

- Mock `huggingface_hub.InferenceClient` responses
- Test: valid Green response, valid Red response, malformed JSON triggers retry, retry exhaustion raises error
- **Done when:** `pytest` passes with ≥ 80 % coverage on `classify.py`

---

### T-11 · Sample Document Set

- The documents are placed in `inbox/` in PDF format
- Run `main.py` against the documents produces expected labels for clear-cut documents
- **Done when:** `python src/main.py` successfully processes all files in `inbox/` end-to-end

---

### T-12 · README

- Document: purpose, setup instructions, how to update the policy, how to run, output format
- Add a sample run output screenshot or copy-paste
- **Done when:** A new developer can clone the repo and run the POC in under 10 minutes following only the README

---

## Milestone Summary

| Milestone           | Tasks                  | Deliverable                                   |
| ------------------- | ---------------------- | --------------------------------------------- |
| M1 — Foundation     | T-01, T-02             | Repo ready, policy defined                    |
| M2 — Ingestion      | T-03, T-04             | Text + metadata extraction working            |
| M3 — Classification | T-05, T-06             | LLM classifying documents correctly           |
| M4 — Output         | T-07, T-08             | Files routed, sidecars written, logs appended |
| M5 — POC Complete   | T-09, T-10, T-11, T-12 | End-to-end run, tested, documented            |
