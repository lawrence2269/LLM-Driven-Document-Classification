import json
from pathlib import Path

import pytest

from src.router.router import route_document


def test_route_document_moves_file_to_green_and_writes_sidecar(tmp_path):
    source = tmp_path / "document.pdf"
    source.write_text("sample content")
    metadata = {"filename": "document.pdf", "classification": "Green"}

    moved_file, sidecar = route_document(source, "Green", metadata, root_dir=tmp_path)

    assert not source.exists()
    assert moved_file == tmp_path / "G" / "document.pdf"
    assert moved_file.exists()
    assert sidecar == tmp_path / "G" / "document.pdf.meta.json"
    assert sidecar.exists()

    loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert loaded == metadata


def test_route_document_handles_filename_collision(tmp_path):
    existing_dir = tmp_path / "G"
    existing_dir.mkdir(parents=True)
    (existing_dir / "document.pdf").write_text("existing")
    (existing_dir / "document_1.pdf").write_text("existing duplicate")

    source = tmp_path / "document.pdf"
    source.write_text("new content")
    metadata = {"filename": "document.pdf", "classification": "Green"}

    moved_file, sidecar = route_document(source, "Green", metadata, root_dir=tmp_path)

    assert moved_file == tmp_path / "G" / "document_2.pdf"
    assert moved_file.exists()
    assert sidecar == tmp_path / "G" / "document_2.pdf.meta.json"
    assert sidecar.exists()

    loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert loaded == metadata


def test_route_document_rejects_invalid_classification(tmp_path):
    source = tmp_path / "document.pdf"
    source.write_text("sample content")
    metadata = {"filename": "document.pdf"}

    with pytest.raises(ValueError, match="classification must be either 'Green' or 'Red'."):
        route_document(source, "Blue", metadata, root_dir=tmp_path)
