import argparse
import sys
import uuid
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

from src.ingest.ingest import walk_inbox
from src.metadata.metadata import extract_metadata, DocumentMetadata
from src.classify.classify import classify_document, ClassificationError, DEFAULT_MODEL_NAME
from src.router.router import route_document
from src.logger.logger import append_audit_log_entry


def process_inbox(
    inbox_path: Path,
    dry_run: bool = False,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[Tuple[str, str, str, str]]:
    """Process all documents in `inbox_path` and return a summary list.

    Returns list of tuples: (filename, classification, confidence, rationale_snippet)
    Each failed document appears as ("filename", "ERROR", "-", "error message").
    """
    run_id = str(uuid.uuid4())
    documents = walk_inbox(str(inbox_path))
    summary = []

    for doc in documents:
        path = doc["path"]
        text = doc.get("text", "")
        filename = Path(path).name

        # --- Metadata extraction ---
        try:
            metadata: DocumentMetadata = extract_metadata(path, text)
        except Exception as exc:
            err_msg = str(exc) if str(exc) else repr(exc)
            print(f"[METADATA ERROR] {filename}: {err_msg}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            summary.append((filename, "ERROR", "-", f"Metadata extraction failed: {err_msg[:80]}"))
            continue

        # --- Classification ---
        try:
            result = classify_document(metadata, text, model_name=model_name)

        except ClassificationError as exc:
            # Walk the full cause chain so the root reason is always visible.
            causes = []
            current: BaseException | None = exc
            while current is not None:
                msg = str(current) if str(current) else repr(current)
                causes.append(msg)
                current = current.__cause__
            full_msg = " → ".join(causes)

            print(f"[CLASSIFICATION ERROR] {filename}:", file=sys.stderr)
            for i, cause in enumerate(causes):
                indent = "  " + ("└─ caused by: " if i > 0 else "")
                print(f"{indent}{cause}", file=sys.stderr)

            append_audit_log_entry(
                run_id=run_id,
                original_path=path,
                classification="ERROR",
                confidence="-",
                rationale=full_msg,
                model=model_name,
                sha256=metadata.sha256_hash,
            )
            summary.append((filename, "ERROR", "-", causes[0][:80]))
            continue

        except Exception as exc:
            # Unexpected error — include full traceback for diagnosis.
            err_msg = str(exc) if str(exc) else repr(exc)
            print(f"[UNEXPECTED ERROR] {filename}: {err_msg}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            summary.append((filename, "ERROR", "-", f"Unexpected error: {err_msg[:80]}"))
            continue

        # --- Routing ---
        if not dry_run:
            try:
                route_document(path, result.classification, asdict(metadata), root_dir=Path("."))
            except Exception as exc:
                err_msg = str(exc) if str(exc) else repr(exc)
                print(f"[ROUTING ERROR] {filename}: {err_msg}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
                summary.append((filename, result.classification, result.confidence, f"Routing failed: {err_msg[:80]}"))
                continue

        # --- Audit log (always, regardless of dry_run) ---
        try:
            append_audit_log_entry(
                run_id=run_id,
                original_path=path,
                classification=result.classification,
                confidence=result.confidence,
                rationale=result.rationale,
                model=model_name,
                sha256=metadata.sha256_hash,
            )
        except Exception as exc:
            # Non-fatal — don't abort the run if logging fails.
            print(f"[LOG WARNING] {filename}: audit log write failed: {exc}", file=sys.stderr)

        rationale_snippet = (result.rationale[:80] + "...") if len(result.rationale) > 80 else result.rationale
        summary.append((filename, result.classification, result.confidence, rationale_snippet))

    return summary


def print_summary(summary: List[Tuple[str, str, str, str]]) -> None:
    if not summary:
        print("No documents processed.")
        return

    total   = len(summary)
    errors  = sum(1 for _, cls, _, _ in summary if cls == "ERROR")
    green   = sum(1 for _, cls, _, _ in summary if cls == "Green")
    red     = sum(1 for _, cls, _, _ in summary if cls == "Red")

    print("\n" + "=" * 80)
    print(f"  Results: {total} document(s) — Green: {green}  Red: {red}  Error: {errors}")
    print("=" * 80)
    col_w = [40, 14, 10, 0]   # filename, classification, confidence, rationale
    header = (
        f"{'Filename':<{col_w[0]}} "
        f"{'Classification':<{col_w[1]}} "
        f"{'Confidence':<{col_w[2]}} "
        f"Rationale"
    )
    print(header)
    print("-" * 80)
    for filename, classification, confidence, rationale in summary:
        print(
            f"{filename:<{col_w[0]}.{col_w[0]}} "
            f"{classification:<{col_w[1]}} "
            f"{confidence:<{col_w[2]}} "
            f"{rationale}"
        )
    print("=" * 80)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Document classification orchestrator.")
    parser.add_argument("--inbox", help="Path to inbox directory", default="inbox")
    parser.add_argument("--dry-run", help="Classify and log but do not move files", action="store_true")
    parser.add_argument(
        "--model",
        help=f"HuggingFace model name to use for classification (default: {DEFAULT_MODEL_NAME})",
        default=DEFAULT_MODEL_NAME,
    )

    args = parser.parse_args(argv)

    inbox_path = Path(args.inbox)
    if not inbox_path.exists():
        print(f"Inbox path does not exist: {inbox_path}", file=sys.stderr)
        sys.exit(2)

    summary = process_inbox(inbox_path, dry_run=bool(args.dry_run), model_name=args.model)
    print_summary(summary)

    # Exit with a non-zero code if any document failed, so CI/scripts can detect it.
    error_count = sum(1 for _, cls, _, _ in summary if cls == "ERROR")
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
