import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import fitz
from langdetect import detect, LangDetectException


@dataclass
class DocumentMetadata:
    """Typed metadata for a document."""

    filename: str
    size_bytes: int
    created_timestamp: float
    modified_timestamp: float
    sha256_hash: str
    language: str
    word_count: int
    pdf_author: Optional[str] = None
    pdf_title: Optional[str] = None


def _calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _get_file_timestamps(file_path: Path) -> tuple[float, float]:
    """Get created and modified timestamps for a file.
    
    Returns (created_timestamp, modified_timestamp).
    Note: On most Unix systems, created_timestamp is the same as modified_timestamp.
    """
    stat_info = file_path.stat()
    # st_birthtime on macOS, st_ctime on Linux (change time, not create time)
    created_time = getattr(stat_info, "st_birthtime", stat_info.st_ctime)
    modified_time = stat_info.st_mtime
    return created_time, modified_time


def _detect_language(text: str) -> str:
    """Detect language from text using langdetect.
    
    Returns language code or 'unknown' if detection fails.
    """
    if not text or len(text.strip()) < 10:
        return "unknown"
    
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _extract_pdf_metadata(file_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Extract author and title from PDF document properties.
    
    Returns (author, title).
    """
    try:
        doc = fitz.open(file_path)
        metadata = doc.metadata
        
        author = metadata.get("author") if metadata else None
        title = metadata.get("title") if metadata else None
        
        doc.close()
        
        return author, title
    except Exception:
        return None, None


def extract_metadata(
    file_path: str,
    text: str,
) -> DocumentMetadata:
    """Extract complete metadata for a document.
    
    Args:
        file_path: Path to the document file
        text: Extracted text content from the document
    
    Returns:
        DocumentMetadata dataclass with all fields populated
    """
    path = Path(file_path)
    
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Extract all metadata components
    filename = path.name
    size_bytes = os.path.getsize(path)
    created_ts, modified_ts = _get_file_timestamps(path)
    sha256 = _calculate_sha256(path)
    language = _detect_language(text)
    word_count = _count_words(text)
    
    # PDF-specific metadata
    pdf_author = None
    pdf_title = None
    if path.suffix.lower() == ".pdf":
        pdf_author, pdf_title = _extract_pdf_metadata(path)
    
    return DocumentMetadata(
        filename=filename,
        size_bytes=size_bytes,
        created_timestamp=created_ts,
        modified_timestamp=modified_ts,
        sha256_hash=sha256,
        language=language,
        word_count=word_count,
        pdf_author=pdf_author,
        pdf_title=pdf_title,
    )
