import fitz
from pathlib import Path
from unittest.mock import patch

from src.main.main import process_inbox
from src.classify.classify import ClassificationResult


def _make_pdf(path: Path, text: str = "Sample"):
    doc = fitz.open()
    doc.new_page()
    doc[0].insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_main_processes_inbox_dry_run(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    # Create sample PDFs
    green = inbox / "green_report.pdf"
    red = inbox / "red_contract.pdf"
    neutral = inbox / "neutral_note.pdf"
    _make_pdf(green, "This is clearly non-sensitive")
    _make_pdf(red, "Contains SSN: 123-45-6789")
    _make_pdf(neutral, "Ambiguous content")

    def fake_classify(metadata, text, **kwargs):
        name = metadata.filename.lower()
        if "green" in name:
            return ClassificationResult(classification="Green", confidence="high", rationale="Safe")
        if "red" in name:
            return ClassificationResult(classification="Red", confidence="medium", rationale="Sensitive")
        return ClassificationResult(classification="Green", confidence="low", rationale="Possibly safe")

    with patch("src.main.main.classify_document", side_effect=fake_classify):
        summary = process_inbox(inbox, dry_run=True)

    # Ensure all files were processed and no files were moved (dry-run)
    assert len(summary) == 3
    filenames = {s[0] for s in summary}
    assert filenames == {"green_report.pdf", "red_contract.pdf", "neutral_note.pdf"}

    # Check classifications match our fake classifier
    by_name = {s[0]: s[1] for s in summary}
    assert by_name["green_report.pdf"] == "Green"
    assert by_name["red_contract.pdf"] == "Red"
    assert by_name["neutral_note.pdf"] in {"Green", "Red"}

    # Files should still exist in inbox because dry-run was True
    assert green.exists() and red.exists() and neutral.exists()
