import json
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union


def _resolve_target_folder(classification: str) -> str:
    normalized = classification.strip().lower()
    if normalized == "green":
        return "G"
    if normalized == "red":
        return "R"
    raise ValueError("classification must be either 'Green' or 'Red'.")


def _find_unique_destination(target_dir: Path, filename: str) -> Path:
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def route_document(
    source_path: Union[str, Path],
    classification: str,
    metadata_dict: Mapping[str, Any],
    root_dir: Union[str, Path] = ".",
    *,
    create_dirs: bool = True,
) -> Tuple[Path, Path]:
    """Move a document into /G/ or /R/ and write a .meta.json sidecar.

    Args:
        source_path: path to the original document file.
        classification: label string, either "Green" or "Red".
        metadata_dict: metadata to serialize into the sidecar JSON.
        root_dir: root folder containing G/ and R/ destination directories.
        create_dirs: whether to create destination directories if missing.

    Returns:
        A tuple of (moved_file_path, sidecar_path).
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.is_dir():
        raise IsADirectoryError(f"Source path must be a file, not a directory: {source}")

    target_name = _resolve_target_folder(classification)
    root = Path(root_dir)
    destination_dir = root / target_name
    if create_dirs:
        destination_dir.mkdir(parents=True, exist_ok=True)

    destination_file = _find_unique_destination(destination_dir, source.name)
    moved_file = Path(shutil.move(str(source), str(destination_file)))

    sidecar_path = moved_file.with_name(f"{moved_file.name}.meta.json")
    with sidecar_path.open("w", encoding="utf-8") as sidecar_file:
        json.dump(metadata_dict, sidecar_file, indent=2, sort_keys=True)

    return moved_file, sidecar_path
