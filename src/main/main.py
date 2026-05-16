import argparse
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from src.ingest.ingest import walk_inbox
from src.metadata.metadata import extract_metadata, DocumentMetadata
from src.classify.classify import classify_document, load_hf_credentials
from src.router.router import route_document
from src.logger.logger import append_audit_log_entry


def process_inbox(
    inbox_path: Path,
    dry_run: bool = False,
    env_path: Path | str = Path(".env"),
) -> List[Tuple[str, str, str, str]]:
    """Process all documents in `inbox_path` and return a summary list.

    Returns list of tuples: (filename, classification, confidence, rationale_snippet)
    """
    run_id = str(uuid.uuid4())

    # Resolve model name for logging
    try:
        _, model_name = load_hf_credentials(env_path)
    except Exception:
        model_name = ""

    documents = walk_inbox(str(inbox_path))
    summary = []

    for doc in documents:
        path = doc["path"]
        text = doc.get("text", "")
        try:
            metadata: DocumentMetadata = extract_metadata(path, text)
            result = classify_document(metadata, text, env_path=env_path)

            if not dry_run:
                # Route the document and write sidecar
                route_document(path, result.classification, asdict(metadata), root_dir=Path("."))

            # Append audit log (always log regardless of dry_run)
            append_audit_log_entry(
                run_id=run_id,
                original_path=path,
                classification=result.classification,
                confidence=result.confidence,
                rationale=result.rationale,
                model=model_name,
                sha256=metadata.sha256_hash,
            )

            rationale_snippet = (result.rationale[:80] + "...") if len(result.rationale) > 80 else result.rationale
            summary.append((metadata.filename, result.classification, result.confidence, rationale_snippet))

        except Exception as exc:  # keep going on errors
            print(f"Error processing {path}: {exc}", file=sys.stderr)

    return summary


def print_summary(summary: List[Tuple[str, str, str, str]]) -> None:
    if not summary:
        print("No documents processed.")
        return

    print("Summary (filename | classification | confidence | rationale):")
    for filename, classification, confidence, rationale in summary:
        print(f"{filename} | {classification} | {confidence} | {rationale}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Document classification orchestrator.")
    parser.add_argument("--inbox", help="Path to inbox directory", default="inbox")
    parser.add_argument("--dry-run", help="Classify and log but do not move files", action="store_true")
    parser.add_argument("--env", help="Path to .env directory or file", default=".env")

    args = parser.parse_args(argv)

    inbox_path = Path(args.inbox)
    if not inbox_path.exists():
        print(f"Inbox path does not exist: {inbox_path}", file=sys.stderr)
        sys.exit(2)

    summary = process_inbox(inbox_path, dry_run=bool(args.dry_run), env_path=args.env)
    print_summary(summary)


if __name__ == "__main__":
    main()
