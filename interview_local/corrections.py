"""Manual corrections of a readable transcript, stored beside the untouched stage files."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from .artifacts import save_json
from .llm import fingerprint, source_words, tokens
from .word_alignment import DisplayWord, align_text_to_source

CORRECTIONS_FILENAME = "corrections.json"
CORRECTIONS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CorrectedParagraph:
    """A paragraph covering the inclusive source word range first_word..last_word."""
    first_word: int
    last_word: int
    speaker: str | None
    text: str
    text_edited: bool
    speaker_edited: bool


def readable_source_words(readable: dict) -> list[dict]:
    """Return the validated source words embedded in a readable (03) transcript.

    Args:
        readable: The parsed 03_readable.json content.

    Returns:
        Source word dicts ordered by consecutive ``word_id``.
    """
    return source_words(readable["source_transcript"])


def load_corrected_paragraphs(run_dir: Path, readable: dict) -> list[CorrectedParagraph]:
    """Load the corrected paragraphs of a run.

    Without corrections.json the run has no corrections yet and its paragraphs
    are those of the readable transcript.

    Args:
        run_dir: The run's output directory.
        readable: The parsed 03_readable.json content of that run.

    Returns:
        Paragraphs covering every source word exactly once, in order.

    Raises:
        ValueError: If corrections.json belongs to a different readable
            transcript or its paragraphs do not cover the source words.
    """
    stored = _stored_corrections(run_dir, readable)
    if stored is None:
        paragraphs = [_paragraph_from_readable(p) for p in readable["paragraphs"]]
    else:
        paragraphs = [CorrectedParagraph(**p) for p in stored["paragraphs"]]
    validate_coverage(paragraphs, readable)
    return paragraphs


def load_speaker_names(run_dir: Path, readable: dict) -> dict[str, str]:
    """Load the speaker names of a run; without corrections the readable transcript's names.

    Args:
        run_dir: The run's output directory.
        readable: The parsed 03_readable.json content of that run.

    Returns:
        Display names keyed by speaker label.
    """
    stored = _stored_corrections(run_dir, readable)
    names = dict(readable["speakers"]) if stored is None else dict(stored["speakers"])
    validate_speaker_names(names, readable)
    return names


def _stored_corrections(run_dir: Path, readable: dict) -> dict | None:
    """Read corrections.json; ``None`` means the named rule "no corrections yet"."""
    path = run_dir / CORRECTIONS_FILENAME
    if not path.exists():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("schema_version") != CORRECTIONS_SCHEMA_VERSION or stored.get("stage") != "corrections":
        raise ValueError(f"{path} is not a supported corrections file.")
    if stored.get("readable_sha256") != fingerprint(readable):
        raise ValueError(f"{path} belongs to a different readable transcript; 03_readable.json has changed.")
    return stored


def _paragraph_from_readable(paragraph: dict) -> CorrectedParagraph:
    word_ids = paragraph["source_word_ids"]
    if word_ids != list(range(word_ids[0], word_ids[-1] + 1)):
        raise ValueError("A readable paragraph does not cover a contiguous source word range.")
    return CorrectedParagraph(first_word=word_ids[0], last_word=word_ids[-1], speaker=paragraph["speaker"],
                              text=paragraph["text"], text_edited=False, speaker_edited=False)


def validate_coverage(paragraphs: list[CorrectedParagraph], readable: dict) -> None:
    """Require every source word exactly once, in order, and known speakers.

    Args:
        paragraphs: The paragraphs to check.
        readable: The parsed 03_readable.json content they belong to.

    Raises:
        ValueError: On gaps, overlaps, reordering, empty text or unknown speakers.
    """
    word_count = len(readable_source_words(readable))
    cursor = 0
    for paragraph in paragraphs:
        if paragraph.first_word != cursor or paragraph.last_word < paragraph.first_word:
            raise ValueError("Corrected paragraphs do not cover the source words exactly once in order.")
        if paragraph.speaker is not None and paragraph.speaker not in readable["speakers"]:
            raise ValueError(f"Unknown speaker '{paragraph.speaker}' in corrected paragraphs.")
        if not tokens(paragraph.text):
            raise ValueError("A corrected paragraph has no words.")
        cursor = paragraph.last_word + 1
    if cursor != word_count:
        raise ValueError("Corrected paragraphs do not cover the source words exactly once in order.")


def validate_speaker_names(names: dict[str, str], readable: dict) -> None:
    """Require one non-empty name for exactly the readable transcript's speaker labels.

    Raises:
        ValueError: If labels differ or a name is empty.
    """
    if set(names) != set(readable["speakers"]):
        raise ValueError(f"Speaker names must be given for exactly {sorted(readable['speakers'])}.")
    if any(not isinstance(name, str) or not name.strip() for name in names.values()):
        raise ValueError("Speaker names must not be empty.")


def edit_paragraph_text(paragraphs: list[CorrectedParagraph], first_word: int,
                        text: str) -> list[CorrectedParagraph]:
    """Replace the text of the paragraph starting at first_word.

    Args:
        paragraphs: The current paragraphs.
        first_word: Source word id where the edited paragraph starts.
        text: The new paragraph text.

    Returns:
        A new paragraph list with the edited paragraph marked ``text_edited``.

    Raises:
        ValueError: If no paragraph starts at first_word or the text has no words.
    """
    if not tokens(text):
        raise ValueError("The paragraph text must contain at least one word.")
    position = _paragraph_position(paragraphs, first_word)
    edited = replace(paragraphs[position], text=text.strip(), text_edited=True)
    return paragraphs[:position] + [edited] + paragraphs[position + 1:]


def _paragraph_position(paragraphs: list[CorrectedParagraph], first_word: int) -> int:
    for position, paragraph in enumerate(paragraphs):
        if paragraph.first_word == first_word:
            return position
    raise ValueError(f"No paragraph starts at word {first_word}.")


def reassign_words(paragraphs: list[CorrectedParagraph], readable_words: list[dict],
                   known_speakers: set[str], first_word: int, last_word: int,
                   speaker: str) -> list[CorrectedParagraph]:
    """Assign a word range inside one paragraph to a speaker, splitting the paragraph.

    The range widens outward to whole display words; inserted display words go
    with their preceding source word. A range part without any display word is
    absorbed into the reassigned part. Neighbours with the same speaker are not merged.

    Args:
        paragraphs: The current paragraphs.
        readable_words: All source words of the readable transcript.
        known_speakers: Speaker labels of the readable transcript.
        first_word: First source word id of the range.
        last_word: Last source word id of the range (inclusive).
        speaker: The speaker label to assign.

    Returns:
        A new paragraph list with up to three paragraphs in place of the old one.

    Raises:
        ValueError: If the speaker is unknown, the range is invalid, spans more
            than one paragraph or contains no displayed word.
    """
    if speaker not in known_speakers:
        raise ValueError(f"Unknown speaker '{speaker}'.")
    position = _position_containing(paragraphs, first_word, last_word)
    paragraph = paragraphs[position]
    words = readable_words[paragraph.first_word:paragraph.last_word + 1]
    display_words = align_text_to_source(paragraph.text, words)
    anchors = _anchor_ranges(display_words, paragraph.first_word)
    begin, end = _widen_selection(anchors, first_word, last_word)
    begin, end = _absorb_textless_parts(anchors, paragraph, begin, end)
    if not any(begin <= low and high <= end for low, high in anchors):
        raise ValueError("The selection contains no displayed word.")
    parts = [
        (paragraph.first_word, begin - 1, paragraph.speaker, paragraph.speaker_edited),
        (begin, end, speaker, True),
        (end + 1, paragraph.last_word, paragraph.speaker, paragraph.speaker_edited)]
    replacement = []
    for low, high, part_speaker, speaker_edited in parts:
        if low > high:
            continue
        part_text = " ".join(word.text for word, (a, b) in zip(display_words, anchors) if low <= a and b <= high)
        replacement.append(CorrectedParagraph(first_word=low, last_word=high, speaker=part_speaker, text=part_text,
                                              text_edited=paragraph.text_edited, speaker_edited=speaker_edited))
    return paragraphs[:position] + replacement + paragraphs[position + 1:]


def _position_containing(paragraphs: list[CorrectedParagraph], first_word: int, last_word: int) -> int:
    if last_word < first_word:
        raise ValueError("The selection ends before it starts.")
    for position, paragraph in enumerate(paragraphs):
        if paragraph.first_word <= first_word <= paragraph.last_word:
            if last_word > paragraph.last_word:
                raise ValueError("A selection must lie within one paragraph.")
            return position
    raise ValueError(f"Word {first_word} is not part of the transcript.")


def _anchor_ranges(display_words: list[DisplayWord], paragraph_first_word: int) -> list[tuple[int, int]]:
    """Source id range per display word; inserted words take their preceding source word."""
    ranges = []
    preceding = paragraph_first_word
    for word in display_words:
        if word.source_word_ids:
            low, high = word.source_word_ids[0], word.source_word_ids[-1]
            preceding = high
        else:
            low = high = preceding
        ranges.append((low, high))
    return ranges


def _widen_selection(anchors: list[tuple[int, int]], first_word: int, last_word: int) -> tuple[int, int]:
    begin, end = first_word, last_word
    while True:
        touched = [(low, high) for low, high in anchors if low <= end and high >= begin]
        widened_begin = min([begin] + [low for low, _ in touched])
        widened_end = max([end] + [high for _, high in touched])
        if (widened_begin, widened_end) == (begin, end):
            return begin, end
        begin, end = widened_begin, widened_end


def _absorb_textless_parts(anchors: list[tuple[int, int]], paragraph: CorrectedParagraph,
                           begin: int, end: int) -> tuple[int, int]:
    has_text_before = any(high < begin for _, high in anchors)
    has_text_after = any(low > end for low, _ in anchors)
    return (begin if has_text_before else paragraph.first_word,
            end if has_text_after else paragraph.last_word)


def save_corrections(run_dir: Path, readable: dict, speakers: dict[str, str],
                     paragraphs: list[CorrectedParagraph]) -> None:
    """Validate and atomically write corrections.json into the run directory.

    Args:
        run_dir: The run's output directory.
        readable: The parsed 03_readable.json content the corrections refer to.
        speakers: Display names keyed by speaker label.
        paragraphs: Paragraphs covering every source word exactly once.

    Raises:
        ValueError: If paragraphs or speaker names are invalid.
    """
    validate_coverage(paragraphs, readable)
    validate_speaker_names(speakers, readable)
    save_json(run_dir / CORRECTIONS_FILENAME, dict(
        schema_version=CORRECTIONS_SCHEMA_VERSION, stage="corrections",
        readable_sha256=fingerprint(readable), speakers={label: name.strip() for label, name in speakers.items()},
        paragraphs=[asdict(p) for p in paragraphs]))
