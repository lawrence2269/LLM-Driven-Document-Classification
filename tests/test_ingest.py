import tempfile
from pathlib import Path

import pytest

from src.ingest.ingest import extract_document, walk_inbox


@pytest.fixture
def temp_inbox():
    """Create a temporary inbox with sample PDF files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        inbox_path = Path(tmpdir) / "inbox"
        inbox_path.mkdir()

        # Create a PDF file
        try:
            import fitz

            pdf_file = inbox_path / "sample.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text(
                (50, 50), "This is a sample PDF file with text content."
            )
            doc.save(str(pdf_file))
        except Exception as e:
            pytest.skip(f"Could not create PDF: {e}")

        yield inbox_path, pdf_file


def test_extract_pdf_yields_nonempty_text(temp_inbox):
    """Test that PDF extraction yields non-empty text."""
    inbox_path, pdf_file = temp_inbox
    result = extract_document(str(pdf_file))

    assert result["extension"] == ".pdf"
    assert len(result["text"]) > 0
    assert "sample PDF file" in result["text"]


def test_walk_inbox_returns_all_pdfs(temp_inbox):
    """Test that walk_inbox recursively finds all PDF documents."""
    inbox_path, pdf_file = temp_inbox

    # Create a subdirectory with another PDF
    subdir = inbox_path / "subdir"
    subdir.mkdir()
    
    try:
        import fitz
        nested_pdf = subdir / "nested.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Nested PDF file content.")
        doc.save(str(nested_pdf))
    except Exception as e:
        pytest.skip(f"Could not create nested PDF: {e}")

    documents = walk_inbox(str(inbox_path))

    # Should find at least 2 PDFs
    assert len(documents) >= 2

    # Verify all documents have non-empty text
    for doc in documents:
        assert len(doc["text"]) > 0
        assert "path" in doc
        assert doc["extension"] == ".pdf"


def test_walk_inbox_ignores_non_pdf_files(temp_inbox):
    """Test that walk_inbox ignores non-PDF file types."""
    inbox_path, _ = temp_inbox

    # Create non-PDF files
    txt_file = inbox_path / "file.txt"
    txt_file.write_text("Text file content")
    
    json_file = inbox_path / "file.json"
    json_file.write_text('{"key": "value"}')

    documents = walk_inbox(str(inbox_path))

    # Non-PDF files should not be included
    for doc in documents:
        assert doc["extension"] == ".pdf"
        assert not doc["path"].endswith(".txt")
        assert not doc["path"].endswith(".json")


def test_extract_document_rejects_non_pdf():
    """Test that extract_document rejects non-PDF files."""
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        txt_file = f.name
        f.write(b"Text content")
    
    try:
        with pytest.raises(ValueError, match="Only PDF files are supported"):
            extract_document(txt_file)
    finally:
        Path(txt_file).unlink()

