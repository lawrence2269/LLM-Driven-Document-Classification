from pathlib import Path
from typing import Dict, List

import fitz

def _extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    document = fitz.open(path)
    text_parts = [page.get_text() for page in document]
    return "\n".join(text_parts).strip()


def extract_document(path: str) -> Dict[str, str]:
    """Extract text from a PDF document and return metadata.

    Returns a dictionary with keys: path, text, extension.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Only PDF files are supported. Got: {file_path.suffix.lower()}"
        )

    text = _extract_text_from_pdf(file_path)

    return {"path": str(file_path), "text": text, "extension": ".pdf"}


def walk_inbox(inbox_path: str) -> List[Dict[str, str]]:
    """Recursively walk an inbox directory and extract PDF documents."""
    root = Path(inbox_path)
    if not root.exists():
        raise FileNotFoundError(f"Inbox path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Inbox path is not a directory: {root}")

    documents: List[Dict[str, str]] = []
    for path in root.rglob("*.pdf"):
        if not path.is_file():
            continue

        extracted = extract_document(str(path))
        documents.append(extracted)

    return documents