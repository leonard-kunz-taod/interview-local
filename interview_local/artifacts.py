"""Persistent, separately named artifacts for every completed pipeline stage."""
import json
from pathlib import Path
from .core import render, timestamp


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so interrupted writes do not leave a partial JSON file.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_raw(raw: dict, directory: Path):
    save_json(directory / "01_raw.json", raw)
    lines = ["RAW TRANSCRIPT — original speech recognition output", ""]
    for segment in raw["segments"]:
        lines += [f"[{timestamp(segment['start'])} – {timestamp(segment['end'])}]", segment["text"].strip(), ""]
    (directory / "01_raw.txt").write_text("\n".join(lines), encoding="utf-8")


def save_speakers(result: dict, directory: Path):
    save_json(directory / "02_speakers.json", result)
    (directory / "02_speakers.txt").write_text(render(result), encoding="utf-8")


def readable_text(result: dict, markdown=False):
    reviewed = result.get("stage") == "review"
    title = "Reviewed transcript" if reviewed else "Readable transcript — LLM draft"
    lines = [("# " if markdown else "") + title, "",
             "Machine-generated draft. Check uncertain passages against the audio.", ""]
    for p in result["paragraphs"]:
        name = result["speakers"].get(p["speaker"], "Unassigned")
        flags = []
        if p["uncertain"]:
            flags.append("UNCERTAIN")
        if p.get("overlap"):
            flags.append("OVERLAPPING SPEECH")
        suffix = " · " + " · ".join(flags) if flags else ""
        heading = f"{name} · {timestamp(p['start'])}–{timestamp(p['end'])}{suffix}"
        lines += [("### " if markdown else "") + heading, "", p["text"], ""]
        if reviewed and p.get("review_reasons"):
            lines += ["Review: " + "; ".join(p["review_reasons"]), ""]
    return "\n".join(lines)


def corrected_text(paragraphs: list[dict], markdown: bool) -> str:
    """Render corrected paragraphs as a clean transcript without review markers.

    Args:
        paragraphs: The ``paragraphs`` of ``editor_document.build_editor_document``.
        markdown: Whether to use Markdown headings.

    Returns:
        The transcript text with speaker name and time range per paragraph.
    """
    lines = [("# " if markdown else "") + "Corrected transcript", ""]
    for p in paragraphs:
        heading = f"{p['speaker_name']} · {timestamp(p['start'])}–{timestamp(p['end'])}"
        lines += [("### " if markdown else "") + heading, "", p["text"], ""]
    return "\n".join(lines)


def save_readable(result: dict, directory: Path):
    prefix = "04_review" if result.get("stage") == "review" else "03_readable"
    save_json(directory / f"{prefix}.json", result)
    for suffix, markdown in (("txt", False), ("md", True)):
        (directory / f"{prefix}.{suffix}").write_text(readable_text(result, markdown), encoding="utf-8")
