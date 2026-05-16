## (Run) Orchestrator

To run the document classification orchestrator from the repository root:

```bash
python -m src.cli
# or explicitly:
python -m src.main.main --inbox inbox --dry-run
```

Use `--dry-run` to classify and log without moving files.
LLM-Based Document Classification — POC
======================================

## Purpose

This repository is a small proof-of-concept for classifying documents (PDFs) as
`Green` (store) or `Red` (sensitive) using an LLM. It demonstrates ingestion,
metadata extraction, prompt construction, HuggingFace inference integration,
routing into `G/` and `R/`, and an append-only audit log.

## Quick Setup

1. Create and activate a Python 3.11+ virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Provide Hugging Face credentials in an `.env` folder (preferred) or file.

- Create a folder named `.env/` and place `.env.example` (copy from repo)
  -> edit `.env/.env.example` and set `HF_API_TOKEN` and `HF_MODEL_NAME`.

## Project Layout

- `inbox/` — place PDFs to process
- `G/` and `R/` — destinations for routed files
- `policy/classification_policy.md` — human-readable policy used in prompts
- `src/` — application code
- `tests/` — unit and integration tests

## Updating the Policy

Edit `policy/classification_policy.md` to change rules. The policy is embedded
in the system prompt sent to the model, so keep it clear and concise. Example
guidance is included to help the LLM decide between `Green` and `Red`.

## Running the Orchestrator

You can run the orchestrator in two safe ways:

```bash
# Safe wrapper
python -m src.cli

# Explicit module invocation with flags
python -m src.main.main --inbox inbox --dry-run
```

Flags

- `--inbox` : path to the inbox directory (default: `inbox`)
- `--dry-run` : classify and log but do not move files
- `--env` : path to `.env` directory or `.env` file (default: `.env`)

## Output and Files Produced

- Files moved to `G/` or `R/` when not in dry-run mode. Collisions are handled
  by appending `_1`, `_2`, ... to filenames.
- Each moved file has a sidecar JSON: `<filename>.meta.json` placed next to it.
  The sidecar contains the extracted metadata (filename, sha256, timestamps,
  language, word count, PDF author/title).
- Audit log: `logs/run_log.jsonl` — append-only JSON Lines file. Each line has
  these fields: `run_id`, `timestamp`, `original_path`, `classification`,
  `confidence`, `rationale`, `model`, `sha256`.

## Sample Summary Output

When a run completes the CLI prints a short summary table, for example:

```
Summary (filename | classification | confidence | rationale):
report.pdf | Green | high | Document appears non-sensitive.
contract.pdf | Red | medium | Contains personal identifiers.
```

## Testing

Run the unit and integration tests with:

```bash
python -m pytest
```

## Notes & Troubleshooting

- If tests or the application cannot find `.env`, ensure you created `.env/`
  and placed `.env.example` (with credentials) inside. The loader supports
  either a directory (`.env/`) or a file (`.env`).
- The project is intentionally minimal; for production use, add secure secret
  handling, robust rate-limiting for API calls, and persistent storage.

## Contributing

Feel free to add integration tests, CI configuration, or improvements to the
prompt and parsing logic. Open an issue or send a PR.
