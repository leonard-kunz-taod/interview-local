"""Persistent, separately named artifacts for every completed pipeline stage."""

import json
from pathlib import Path

from .core import render, timestamp

DRAFT_NOTICE = "Machine-generated draft. Check uncertain passages against the audio."


def save_raw(raw: dict, directory: Path) -> None:
    """Save the unmodified speech recognition output as 01_raw.json/.txt."""
    _save_json(directory / "01_raw.json", raw)
    lines = ["RAW TRANSCRIPT — original speech recognition output", ""]
    for segment in raw["segments"]:
        time_range = f"{timestamp(segment['start'])} – {timestamp(segment['end'])}"
        lines += [f"[{time_range}]", segment["text"].strip(), ""]
    (directory / "01_raw.txt").write_text("\n".join(lines), encoding="utf-8")


def save_speakers(result: dict, directory: Path) -> None:
    """Save the speaker-attributed transcript as 02_speakers.json/.txt."""
    _save_speaker_files(result, directory / "02_speakers")


def save_speakers_with_aliases(result: dict, directory: Path) -> None:
    """Save the final speaker stage plus its compatibility aliases transcript.*."""
    save_speakers(result, directory)
    _save_speaker_files(result, directory / "transcript")


def save_readable(result: dict, directory: Path) -> None:
    """Save a readable draft as 03_readable.*, or a reviewed one as 04_review.*."""
    prefix = "04_review" if result["stage"] == "review" else "03_readable"
    _save_json(directory / f"{prefix}.json", result)
    (directory / f"{prefix}.txt").write_text(readable_text(result), encoding="utf-8")
    (directory / f"{prefix}.md").write_text(readable_markdown(result), encoding="utf-8")


def readable_text(result: dict) -> str:
    return _readable_document(result, title_marker="", heading_marker="")


def readable_markdown(result: dict) -> str:
    return _readable_document(result, title_marker="# ", heading_marker="### ")


def _readable_document(result: dict, title_marker: str, heading_marker: str) -> str:
    reviewed = result["stage"] == "review"
    lines = [title_marker + _document_title(result), "", DRAFT_NOTICE, ""]
    for p in result["paragraphs"]:
        lines += [heading_marker + _paragraph_heading(result["speakers"], p), ""]
        lines += [p["text"], ""]
        if reviewed and p["review_reasons"]:
            lines += ["Review: " + "; ".join(p["review_reasons"]), ""]
    return "\n".join(lines)


def _document_title(result: dict) -> str:
    if result["stage"] == "review":
        return "Reviewed transcript"
    return "Readable transcript — LLM draft"


def _paragraph_heading(speakers: dict, paragraph: dict) -> str:
    name = speakers.get(paragraph["speaker"], "Unassigned")
    flags = []
    if paragraph["uncertain"]:
        flags.append("UNCERTAIN")
    if paragraph.get("overlap"):
        flags.append("OVERLAPPING SPEECH")
    suffix = " · " + " · ".join(flags) if flags else ""
    time_range = f"{timestamp(paragraph['start'])}–{timestamp(paragraph['end'])}"
    return f"{name} · {time_range}{suffix}"


def _save_speaker_files(result: dict, stem: Path) -> None:
    _save_json(stem.with_suffix(".json"), result)
    stem.with_suffix(".txt").write_text(render(result), encoding="utf-8")


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so interrupted writes do not leave a partial JSON file.
    # pragma: no mutate start -- replaced onto path before returning, so its name is never observed
    temporary = path.with_suffix(path.suffix + ".tmp")
    # pragma: no mutate end
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
