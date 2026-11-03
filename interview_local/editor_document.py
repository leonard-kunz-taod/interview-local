"""The browser editor's view of a run: corrected paragraphs with review highlights."""
from __future__ import annotations

from .corrections import CorrectedParagraph, readable_source_words
from .word_alignment import DisplayWord, align_text_to_source

LOW_ASR_PROBABILITY = 0.5
SPEAKER_UNCERTAINTY_REASONS = frozenset({"overlapping_speech", "ambiguous_speaker_boundary", "no_matching_speaker"})
UNASSIGNED_NAME = "Unassigned"


def build_editor_document(readable: dict, paragraphs: list[CorrectedParagraph], speakers: dict[str, str]) -> dict:
    """Build the JSON document the browser editor renders.

    Change highlights always compare against the source (ASR) words, so after a
    manual edit they mean "differs from speech recognition".

    Args:
        readable: The parsed 03_readable.json content.
        paragraphs: The run's corrected paragraphs.
        speakers: Display names keyed by speaker label.

    Returns:
        A JSON-serializable dict with speakers and highlighted paragraphs.

    Raises:
        ValueError: If the source words carry no Whisper word probability.
    """
    words = readable_source_words(readable)
    if any("probability" not in word for word in words):
        raise ValueError("This run was transcribed before word probabilities were stored "
                         "and cannot be opened in the editor. Transcribe the audio again.")
    return dict(speakers=dict(speakers), low_asr_probability=LOW_ASR_PROBABILITY,
                paragraphs=[_editor_paragraph(readable["paragraphs"], words, p, speakers) for p in paragraphs])


def _editor_paragraph(readable_paragraphs: list[dict], words: list[dict], paragraph: CorrectedParagraph,
                      speakers: dict[str, str]) -> dict:
    paragraph_words = words[paragraph.first_word:paragraph.last_word + 1]
    start = min(w["start"] for w in paragraph_words)
    display_words = align_text_to_source(paragraph.text, paragraph_words)
    return dict(
        first_word=paragraph.first_word, last_word=paragraph.last_word, speaker=paragraph.speaker,
        speaker_name=speakers[paragraph.speaker] if paragraph.speaker is not None else UNASSIGNED_NAME,
        start=start, end=max(w["end"] for w in paragraph_words), text=paragraph.text,
        text_edited=paragraph.text_edited, speaker_edited=paragraph.speaker_edited,
        speaker_uncertainty=speaker_uncertainty(readable_paragraphs, paragraph_words, paragraph),
        words=_editor_words(display_words, words, start))


def speaker_uncertainty(readable_paragraphs: list[dict], paragraph_words: list[dict],
                        paragraph: CorrectedParagraph) -> list[str]:
    """Reasons to review a paragraph's speaker; a manual reassignment clears them.

    Args:
        readable_paragraphs: The LLM paragraphs of the parsed 03_readable.json content.
        paragraph_words: The source words of the paragraph.
        paragraph: The corrected paragraph.

    Returns:
        Sorted reason names; empty when the speaker needs no review.
    """
    if paragraph.speaker_edited:
        return []
    reasons = _word_speaker_reasons(paragraph_words)
    if paragraph.speaker is None:
        reasons.add("unassigned")
    if _readable_paragraph_containing(readable_paragraphs, paragraph.first_word)["speaker_changed"]:
        reasons.add("speaker_changed_by_llm")
    return sorted(reasons)


def _word_speaker_reasons(paragraph_words: list[dict]) -> set[str]:
    return {reason for word in paragraph_words for reason in word["uncertainty_reasons"]
            if reason in SPEAKER_UNCERTAINTY_REASONS}


def _readable_paragraph_containing(readable_paragraphs: list[dict], word_id: int) -> dict:
    for readable_paragraph in readable_paragraphs:
        if word_id in readable_paragraph["source_word_ids"]:
            return readable_paragraph
    raise ValueError(f"Word {word_id} belongs to no readable paragraph.")


def _editor_words(display_words: list[DisplayWord], words: list[dict], paragraph_start: float) -> list[dict]:
    seek_times = _seek_times(display_words, words, paragraph_start)
    return [_editor_word(word, words, seek_time) for word, seek_time in zip(display_words, seek_times)]


def _seek_times(display_words: list[DisplayWord], words: list[dict], paragraph_start: float) -> list[float]:
    """Each word's playback position: its first source word's start; for an inserted word the end of the
    preceding source word, or the paragraph start."""
    seek_times = []
    previous_end = paragraph_start
    for word in display_words:
        ids = word.source_word_ids
        seek_times.append(words[ids[0]]["start"] if ids else previous_end)
        if ids:
            previous_end = words[ids[-1]]["end"]
    return seek_times


def _editor_word(word: DisplayWord, words: list[dict], seek_time: float) -> dict:
    sources = [words[i] for i in word.source_word_ids]
    return dict(text=word.text, source_word_ids=list(word.source_word_ids), seek_time=seek_time,
                change=word.change, original_text=word.original_text,
                removed_before=word.removed_before, low_asr_confidence=low_asr_confidence(sources))


def low_asr_confidence(sources: list[dict]) -> bool:
    """Whether speech recognition was uncertain about any of these source words.

    A source word without a Whisper probability (estimated timing for a whole
    segment) is treated as low confidence: nothing vouches for its recognition.

    Args:
        sources: Source word dicts with a ``probability`` key.

    Returns:
        True if any probability is missing or below LOW_ASR_PROBABILITY.
    """
    return any(w["probability"] is None or w["probability"] < LOW_ASR_PROBABILITY for w in sources)
