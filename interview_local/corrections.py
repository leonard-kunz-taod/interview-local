"""Manual corrections of a readable transcript, stored beside the untouched stage files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .artifacts import save_json
from .llm import fingerprint, source_words, tokens
from .word_alignment import DisplayWord, align_text_to_source

CORRECTIONS_FILENAME = "corrections.json"
CORRECTIONS_SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
COVERAGE_MESSAGE = (
    "Corrected paragraphs do not cover the source words exactly once in order."
)
_SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {LEGACY_SCHEMA_VERSION, CORRECTIONS_SCHEMA_VERSION}
)


@dataclass(frozen=True)
class CorrectedParagraph:
    """A paragraph covering the inclusive source word range first_word..last_word."""

    first_word: int
    last_word: int
    speaker: str | None
    text: str
    text_edited: bool
    speaker_edited: bool
    checked: bool


def readable_source_words(readable: dict) -> list[dict]:
    """The validated source words embedded in a readable (03) transcript, ordered by consecutive ``word_id``."""
    return source_words(readable["source_transcript"])


def load_corrected_paragraphs(
    run_dir: Path, readable: dict
) -> list[CorrectedParagraph]:
    """Load a run's corrected paragraphs; without corrections.json, those of the readable transcript.

    Raises:
        ValueError: If corrections.json belongs to a different readable
            transcript or its paragraphs do not cover the source words.
    """
    stored = _stored_corrections(run_dir, readable)
    if stored is None:
        paragraphs = [_paragraph_from_readable(p) for p in readable["paragraphs"]]
    else:
        paragraphs = _stored_paragraphs(stored)
    validate_coverage(paragraphs, readable)
    return paragraphs


def load_speaker_names(run_dir: Path, readable: dict) -> dict[str, str]:
    """A run's display names keyed by speaker label; without corrections the readable transcript's names."""
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
    _require_supported_format(
        stored, path
    )  # pragma: no mutate - path only names the file in the error message
    if stored.get("readable_sha256") != fingerprint(readable):
        raise ValueError(
            f"{path} belongs to a different readable transcript; 03_readable.json has changed."
        )
    return stored


def _require_supported_format(stored: dict, path: Path) -> None:
    if (
        stored.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS
        or stored.get("stage") != "corrections"
    ):
        raise ValueError(f"{path} is not a supported corrections file.")


def _stored_paragraphs(stored: dict) -> list[CorrectedParagraph]:
    if stored["schema_version"] == LEGACY_SCHEMA_VERSION:
        return [_paragraph_from_legacy(p) for p in stored["paragraphs"]]
    return [CorrectedParagraph(**p) for p in stored["paragraphs"]]


def _paragraph_from_legacy(paragraph: dict) -> CorrectedParagraph:
    """Business rule: files written before review status existed have nothing checked."""
    return CorrectedParagraph(**paragraph, checked=False)


def _paragraph_from_readable(paragraph: dict) -> CorrectedParagraph:
    word_ids = paragraph["source_word_ids"]
    if word_ids != list(range(word_ids[0], word_ids[-1] + 1)):
        raise ValueError(
            "A readable paragraph does not cover a contiguous source word range."
        )
    return CorrectedParagraph(
        first_word=word_ids[0],
        last_word=word_ids[-1],
        speaker=paragraph["speaker"],
        text=paragraph["text"],
        text_edited=False,
        speaker_edited=False,
        checked=False,
    )


def validate_coverage(paragraphs: list[CorrectedParagraph], readable: dict) -> None:
    """Require every source word exactly once, in order, and known speakers.

    Raises:
        ValueError: On gaps, overlaps, reordering, empty text or unknown speakers.
    """
    word_count = len(readable_source_words(readable))
    cursor = 0
    for paragraph in paragraphs:
        _require_range_starting_at(paragraph, cursor)
        _require_known_speaker(paragraph, readable["speakers"])
        if not tokens(paragraph.text):
            raise ValueError("A corrected paragraph has no words.")
        cursor = paragraph.last_word + 1
    if cursor != word_count:
        raise ValueError(COVERAGE_MESSAGE)


def _require_range_starting_at(paragraph: CorrectedParagraph, first_word: int) -> None:
    if paragraph.first_word != first_word or paragraph.last_word < paragraph.first_word:
        raise ValueError(COVERAGE_MESSAGE)


def _require_known_speaker(
    paragraph: CorrectedParagraph, speakers: dict[str, str]
) -> None:
    """Allow the speakers of the readable transcript and no speaker (unassigned)."""
    if paragraph.speaker is not None and paragraph.speaker not in speakers:
        raise ValueError(
            f"Unknown speaker '{paragraph.speaker}' in corrected paragraphs."
        )


def validate_speaker_names(names: dict[str, str], readable: dict) -> None:
    """Require one non-empty name for exactly the readable transcript's speaker labels."""
    if set(names) != set(readable["speakers"]):
        raise ValueError(
            f"Speaker names must be given for exactly {sorted(readable['speakers'])}."
        )
    if not all(map(_is_usable_name, names.values())):
        raise ValueError("Speaker names must not be empty.")


def _is_usable_name(name: str) -> bool:
    return bool(name.strip())


def edit_paragraph_text(
    paragraphs: list[CorrectedParagraph], first_word: int, text: str
) -> list[CorrectedParagraph]:
    """Replace the text of the paragraph starting at first_word and mark it checked; returns a new list.

    Raises:
        ValueError: If no paragraph starts at first_word or the text has no words.
    """
    if not tokens(text):
        raise ValueError("The paragraph text must contain at least one word.")
    return _with_paragraph_changed(
        paragraphs, first_word, text=text.strip(), text_edited=True, checked=True
    )


def set_paragraph_checked(
    paragraphs: list[CorrectedParagraph], first_word: int, checked: bool
) -> list[CorrectedParagraph]:
    """Set the review status of the paragraph starting at first_word; returns a new list.

    Raises:
        ValueError: If no paragraph starts at first_word.
    """
    return _with_paragraph_changed(paragraphs, first_word, checked=checked)


def _with_paragraph_changed(
    paragraphs: list[CorrectedParagraph], first_word: int, **changes: object
) -> list[CorrectedParagraph]:
    """A new list in which the paragraph starting at first_word has the given field values."""
    position = _paragraph_position(paragraphs, first_word)
    changed = replace(paragraphs[position], **changes)
    return paragraphs[:position] + [changed] + paragraphs[position + 1 :]


def _paragraph_position(paragraphs: list[CorrectedParagraph], first_word: int) -> int:
    for position, paragraph in enumerate(paragraphs):
        if paragraph.first_word == first_word:
            return position
    raise ValueError(f"No paragraph starts at word {first_word}.")


def reassign_words(
    paragraphs: list[CorrectedParagraph],
    readable_words: list[dict],
    known_speakers: set[str],
    first_word: int,
    last_word: int,
    speaker: str,
) -> list[CorrectedParagraph]:
    """Assign the inclusive source word range first_word..last_word inside one paragraph to a speaker.

    The paragraph splits into up to three. The range widens outward to whole display words; inserted display words go
    with their preceding source word. A range part without any display word is
    absorbed into the reassigned part. Neighbours with the same speaker are not merged.
    Every resulting part counts as checked; the other paragraphs keep their status.

    Returns a new paragraph list.

    Raises:
        ValueError: If the speaker is unknown, the range is invalid, spans more
            than one paragraph or contains no displayed word.
    """
    if speaker not in known_speakers:
        raise ValueError(f"Unknown speaker '{speaker}'.")
    position = _position_containing(paragraphs, first_word, last_word)
    paragraph = paragraphs[position]
    words = readable_words[paragraph.first_word : paragraph.last_word + 1]
    display_words = align_text_to_source(paragraph.text, words)
    anchors = _anchor_ranges(display_words, paragraph.first_word)
    begin, end = _widen_selection(anchors, first_word, last_word)
    begin, end = _absorb_textless_parts(anchors, paragraph, begin, end)
    _require_displayed_word(anchors, begin, end)
    replacement = [
        CorrectedParagraph(
            first_word=low,
            last_word=high,
            speaker=part_speaker,
            text=_text_within(display_words, anchors, low, high),
            text_edited=paragraph.text_edited,
            speaker_edited=speaker_edited,
            checked=True,
        )
        for low, high, part_speaker, speaker_edited in _split_parts(
            paragraph, begin, end, speaker
        )
    ]
    return paragraphs[:position] + replacement + paragraphs[position + 1 :]


def _position_containing(
    paragraphs: list[CorrectedParagraph], first_word: int, last_word: int
) -> int:
    if last_word < first_word:
        raise ValueError("The selection ends before it starts.")
    position = _position_of_word(paragraphs, first_word)
    if last_word > paragraphs[position].last_word:
        raise ValueError("A selection must lie within one paragraph.")
    return position


def _position_of_word(paragraphs: list[CorrectedParagraph], word_id: int) -> int:
    for position, paragraph in enumerate(paragraphs):
        if paragraph.first_word <= word_id <= paragraph.last_word:
            return position
    raise ValueError(f"Word {word_id} is not part of the transcript.")


def _anchor_ranges(
    display_words: list[DisplayWord], paragraph_first_word: int
) -> list[tuple[int, int]]:
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


def _widen_selection(
    anchors: list[tuple[int, int]], first_word: int, last_word: int
) -> tuple[int, int]:
    selection = (first_word, last_word)
    while True:
        widened = _cover_touching_anchors(anchors, *selection)
        if widened == selection:
            return selection
        selection = widened


def _cover_touching_anchors(
    anchors: list[tuple[int, int]], begin: int, end: int
) -> tuple[int, int]:
    """Extend the range to every anchor range it touches."""
    touched = [(low, high) for low, high in anchors if low <= end and high >= begin]
    lows, highs = zip((begin, end), *touched)
    return min(lows), max(highs)


def _absorb_textless_parts(
    anchors: list[tuple[int, int]], paragraph: CorrectedParagraph, begin: int, end: int
) -> tuple[int, int]:
    lows, highs = zip(*anchors)
    has_text_before = min(highs) < begin
    has_text_after = max(lows) > end
    return (
        begin if has_text_before else paragraph.first_word,
        end if has_text_after else paragraph.last_word,
    )


def _require_displayed_word(
    anchors: list[tuple[int, int]], begin: int, end: int
) -> None:
    if not any(begin <= low <= high <= end for low, high in anchors):
        raise ValueError("The selection contains no displayed word.")


def _split_parts(
    paragraph: CorrectedParagraph, begin: int, end: int, speaker: str
) -> list[tuple[int, int, str | None, bool]]:
    """The non-empty parts before, of and after the reassigned range: (first, last, speaker, speaker_edited)."""
    parts = [
        (paragraph.first_word, begin - 1, paragraph.speaker, paragraph.speaker_edited),
        (begin, end, speaker, True),
        (end + 1, paragraph.last_word, paragraph.speaker, paragraph.speaker_edited),
    ]
    return [part for part in parts if part[0] <= part[1]]


def _text_within(
    display_words: list[DisplayWord],
    anchors: list[tuple[int, int]],
    low: int,
    high: int,
) -> str:
    return " ".join(
        word.text
        for word, (first, last) in zip(display_words, anchors)
        if low <= first <= last <= high
    )


def save_corrections(
    run_dir: Path,
    readable: dict,
    speakers: dict[str, str],
    paragraphs: list[CorrectedParagraph],
) -> None:
    """Validate, then atomically write corrections.json into the run directory."""
    validate_coverage(paragraphs, readable)
    validate_speaker_names(speakers, readable)
    save_json(
        run_dir / CORRECTIONS_FILENAME,
        {
            "schema_version": CORRECTIONS_SCHEMA_VERSION,
            "stage": "corrections",
            "readable_sha256": fingerprint(readable),
            "speakers": {label: name.strip() for label, name in speakers.items()},
            "paragraphs": [asdict(p) for p in paragraphs],
        },
    )
