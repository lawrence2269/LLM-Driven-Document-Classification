import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from src.ingest.ingest import extract_document
from src.metadata.metadata import DocumentMetadata, extract_metadata


@pytest.fixture
def sample_pdf():
    """Create a sample PDF with known properties for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "test_document.pdf"
        
        import fitz
        
        doc = fitz.open()
        
        # Set metadata
        metadata = doc.metadata
        metadata["author"] = "Test Author"
        metadata["title"] = "Test Document Title"
        doc.set_metadata(metadata)
        
        # Add content
        page = doc.new_page()
        content = (
            "This is a test PDF document. "
            "It contains multiple words for testing word count functionality. "
            "The document includes author and title metadata. "
            "We can detect the language of the content. "
            "This is helpful for document classification tasks."
        )
        page.insert_text((50, 50), content)
        
        doc.save(str(pdf_path))
        doc.close()
        
        yield pdf_path


@pytest.fixture
def sample_pdf_no_metadata():
    """Create a PDF with no author/title metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "no_metadata.pdf"
        
        import fitz
        
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Document without metadata properties.")
        doc.save(str(pdf_path))
        doc.close()
        
        yield pdf_path


def test_extract_metadata_returns_dataclass(sample_pdf):
    """Test that extract_metadata returns a DocumentMetadata dataclass."""
    text = "Sample text content for metadata extraction."
    metadata = extract_metadata(str(sample_pdf), text)
    
    assert isinstance(metadata, DocumentMetadata)


def test_filename_extraction(sample_pdf):
    """Test that filename is correctly extracted."""
    text = "Sample text."
    metadata = extract_metadata(str(sample_pdf), text)
    
    assert metadata.filename == "test_document.pdf"


def test_size_extraction(sample_pdf):
    """Test that file size is correctly extracted."""
    text = "Sample text."
    metadata = extract_metadata(str(sample_pdf), text)
    
    # Size should be greater than 0
    assert metadata.size_bytes > 0
    # Verify it matches actual file size
    assert metadata.size_bytes == os.path.getsize(sample_pdf)


def test_timestamps_extraction(sample_pdf):
    """Test that created and modified timestamps are extracted."""
    text = "Sample text."
    metadata = extract_metadata(str(sample_pdf), text)
    
    assert metadata.created_timestamp > 0
    assert metadata.modified_timestamp > 0


def test_sha256_hash_extraction(sample_pdf):
    """Test that SHA-256 hash is correctly calculated."""
    text = "Sample text."
    metadata = extract_metadata(str(sample_pdf), text)
    
    # Calculate hash independently
    sha256_hash = hashlib.sha256()
    with open(sample_pdf, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    expected_hash = sha256_hash.hexdigest()
    
    assert metadata.sha256_hash == expected_hash


def test_language_detection(sample_pdf):
    """Test that language is detected from text."""
    text = (
        "This is a test document in English. "
        "It contains multiple sentences to ensure proper language detection. "
        "The language should be detected as English."
    )
    metadata = extract_metadata(str(sample_pdf), text)
    
    assert metadata.language is not None
    assert metadata.language != "unknown"


def test_word_count_extraction(sample_pdf):
    """Test that word count is correctly calculated."""
    text = "one two three four five"
    metadata = extract_metadata(str(sample_pdf), text)
    
    assert metadata.word_count == 5


def test_pdf_author_extraction(sample_pdf):
    """Test that PDF author is extracted from document properties."""
    text = "Sample content."
    metadata = extract_metadata(str(sample_pdf), text)
    
    assert metadata.pdf_author == "Test Author"


def test_pdf_title_extraction(sample_pdf):
    """Test that PDF title is extracted from document properties."""
    text = "Sample content."
    metadata = extract_metadata(str(sample_pdf), text)
    
    assert metadata.pdf_title == "Test Document Title"


def test_pdf_no_metadata_fields_optional(sample_pdf_no_metadata):
    """Test that missing PDF metadata fields are handled gracefully."""
    text = "Sample content."
    metadata = extract_metadata(str(sample_pdf_no_metadata), text)
    
    # These fields can be None if not set in the PDF
    assert metadata.pdf_author is None or isinstance(metadata.pdf_author, str)
    assert metadata.pdf_title is None or isinstance(metadata.pdf_title, str)


def test_no_none_values_for_mandatory_fields(sample_pdf):
    """Test that all mandatory metadata fields are populated (no None values)."""
    text = "This is test content with multiple words for proper word counting and language detection."
    metadata = extract_metadata(str(sample_pdf), text)
    
    # Mandatory fields - none should be None
    assert metadata.filename is not None
    assert metadata.size_bytes is not None
    assert metadata.created_timestamp is not None
    assert metadata.modified_timestamp is not None
    assert metadata.sha256_hash is not None
    assert metadata.language is not None
    assert metadata.word_count is not None
    
    # Optional PDF fields may be None
    # but if they exist, they should be non-empty strings
    if metadata.pdf_author is not None:
        assert isinstance(metadata.pdf_author, str)
    if metadata.pdf_title is not None:
        assert isinstance(metadata.pdf_title, str)


def test_integration_with_ingest_module(sample_pdf):
    """Test metadata extraction integrated with the ingest module."""
    # First, extract text using the ingest module
    doc = extract_document(str(sample_pdf))
    
    # Then extract metadata
    metadata = extract_metadata(doc["path"], doc["text"])
    
    # Verify all mandatory fields are present
    assert metadata.filename is not None
    assert metadata.size_bytes > 0
    assert metadata.word_count > 0
    assert metadata.language is not None
    assert metadata.sha256_hash is not None


def test_metadata_dict_conversion(sample_pdf):
    """Test that metadata can be converted to a dictionary."""
    text = "Sample content for testing."
    metadata = extract_metadata(str(sample_pdf), text)
    
    # Convert to dict
    metadata_dict = {
        "filename": metadata.filename,
        "size_bytes": metadata.size_bytes,
        "created_timestamp": metadata.created_timestamp,
        "modified_timestamp": metadata.modified_timestamp,
        "sha256_hash": metadata.sha256_hash,
        "language": metadata.language,
        "word_count": metadata.word_count,
        "pdf_author": metadata.pdf_author,
        "pdf_title": metadata.pdf_title,
    }
    
    # Verify all keys are present and no None for mandatory fields
    assert all(k in metadata_dict for k in [
        "filename", "size_bytes", "created_timestamp", 
        "modified_timestamp", "sha256_hash", "language", "word_count"
    ])
    
    # Mandatory fields should not be None
    assert metadata_dict["filename"] is not None
    assert metadata_dict["size_bytes"] is not None
    assert metadata_dict["sha256_hash"] is not None
    assert metadata_dict["language"] is not None
    assert metadata_dict["word_count"] is not None
