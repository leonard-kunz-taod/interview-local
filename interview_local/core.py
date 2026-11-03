from __future__ import annotations

import bisect
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

AMBIGUOUS_OVERLAP_RATIO = 0.8
SEGMENT_PAUSE_SECONDS = 1.5
SEGMENT_MAX_SECONDS = 35
UNRELIABLE_TIMINGS = {"estimated", "whisper_fallback"}
SEGMENT_FIELDS = ("start", "end", "text", "speaker", "uncertain", "overlap")
SEGMENT_FLAG_LABELS = (
    ("uncertain", "Review assignment"),
    ("overlap", "Overlapping speech"),
)


@dataclass
class Word:
    start: float
    end: float
    text: str
    timing: str = "whisper"


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    exclusive: list[Turn]
    regular: list[Turn]


def merge(
    words: list[Word], turns: list[Turn], regular_turns: list[Turn] | None
) -> dict:
    """Assign words by exclusive overlap; separately retain simultaneous voices.

    Args:
        words: Recognized words with timing.
        turns: Exclusive diarization turns used for word assignment.
        regular_turns: Diarization turns that may overlap; None for legacy input
            without them, which marks overlap detection unavailable.

    Returns:
        The speaker-attributed transcript.
    """
    turns = sorted(turns, key=lambda t: (t.start, t.end))
    regular = regular_turns or []
    labels = _speaker_labels(turns, regular)
    regions = overlap_regions(regular, labels)
    attributed = _attributed_words(words, turns, regions, labels)
    return {
        "schema_version": 2,
        "language": "de",
        "speakers": {label: f"Speaker {label}" for label in labels.values()},
        "segments": _segments(attributed),
        "words": attributed,
        "diarization": _labelled(turns, labels),
        "regular_diarization": _labelled(regular, labels),
        "overlaps": regions,
        "overlap_detection_available": regular_turns is not None,
    }


def overlap_regions(turns: list[Turn], mapping: dict[str, str]) -> list[dict]:
    """Sweep regular turns; duplicate tracks from one speaker are not overlap."""
    regions = []
    for start, end, speakers in _simultaneous_intervals(turns, mapping):
        if regions and _continues_region(regions[-1], start, speakers):
            regions[-1]["end"] = end
        else:
            regions.append({"start": start, "end": end, "speakers": speakers})
    return regions


def segment_words(segment: dict, timing: str) -> list[dict]:
    """Words of an ASR segment labelled with the given timing source.

    A segment without word timestamps becomes one word with estimated timing.
    """
    if segment["words"]:
        return [dict(w, timing=timing) for w in segment["words"]]
    return [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "timing": "estimated",
        }
    ]


def timestamp(value: float) -> str:
    ms = round(value * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"


def render(result: dict) -> str:
    blocks = [_rendered_segment(s, result["speakers"]) for s in result["segments"]]
    return "\n".join(blocks) or "No speech detected."


def _rendered_segment(segment: dict, speakers: dict) -> str:
    name = speakers.get(segment["speaker"], "Unassigned")
    time_range = f"{timestamp(segment['start'])} – {timestamp(segment['end'])}"
    return f"[{time_range}] {name}{_review_flags(segment)}\n{segment['text']}\n"


def _review_flags(segment: dict) -> str:
    flags = [label for key, label in SEGMENT_FLAG_LABELS if segment.get(key)]
    return f" [{'; '.join(flags)}]" if flags else ""


def _speaker_labels(turns: list[Turn], regular_turns: list[Turn]) -> dict[str, str]:
    """Letter speakers in order of their first exclusive, then regular, turn."""
    ordered = [*turns, *sorted(regular_turns, key=lambda t: t.start)]
    ids = dict.fromkeys(t.speaker for t in ordered)
    return {speaker: chr(ord("A") + i) for i, speaker in enumerate(ids)}


def _simultaneous_intervals(
    turns: list[Turn], mapping: dict[str, str]
) -> Iterator[tuple[float, float, list[str]]]:
    """Consecutive sweep intervals in which more than one speaker is active."""
    active: Counter[str] = Counter()
    events = sorted(_speaker_events(turns).items())
    for (previous, changes), (current, _) in itertools.pairwise(events):
        active.update(changes)
        speakers = _active_labels(active, mapping)
        if len(speakers) > 1:
            yield previous, current, speakers


def _speaker_events(turns: list[Turn]) -> dict[float, Counter[str]]:
    """Per instant, how many tracks of each speaker start (+1) or end (-1)."""
    events: defaultdict[float, Counter[str]] = defaultdict(Counter)
    for turn in turns:
        # pragma: no mutate start -- a zero-length turn adds +1 and -1 at one instant, so admitting it changes no count
        if turn.end > turn.start:
            # pragma: no mutate end
            events[turn.start][turn.speaker] += 1
            events[turn.end][turn.speaker] -= 1
    return events


def _active_labels(active: Counter[str], mapping: dict[str, str]) -> list[str]:
    return sorted(mapping[s] for s, count in active.items() if count > 0)


def _continues_region(region: dict, start: float, speakers: list[str]) -> bool:
    return region["end"] == start and region["speakers"] == speakers


def _attributed_words(
    words: list[Word], turns: list[Turn], regions: list[dict], labels: dict[str, str]
) -> list[dict]:
    turn_starts = [t.start for t in turns]
    region_starts = [r["start"] for r in regions]
    attributed = []
    for word_id, word in enumerate(words):
        candidates = _ranked_candidates(word, turns, turn_starts)
        simultaneous = _simultaneous_speakers(word, regions, region_starts)
        reasons = _uncertainty_reasons(word, candidates, simultaneous)
        attributed.append(
            {
                "word_id": word_id,
                "start": word.start,
                "end": word.end,
                "text": word.text.strip(),
                "speaker": _best_speaker(candidates, labels),
                "timing": word.timing,
                "uncertain": bool(reasons),
                "uncertainty_reasons": reasons,
                "overlap": bool(simultaneous),
                "overlap_speakers": sorted(simultaneous),
            }
        )
    return attributed


def _ranked_candidates(
    word: Word, turns: list[Turn], turn_starts: list[float]
) -> list[tuple[float, Turn]]:
    """Turns touching the word, paired with their overlap, largest first."""
    # pragma: no mutate start -- exclusive turns never overlap, so no earlier turn reaches the word
    first = max(0, bisect.bisect_right(turn_starts, word.start) - 1)
    # pragma: no mutate end
    candidates = []
    # pragma: no mutate start -- exclusive turns never overlap, so no earlier turn reaches the word
    for i in range(first, len(turns)):
        # pragma: no mutate end
        turn = turns[i]
        if turn.start > word.end:
            break
        if _touches(word, turn.start, turn.end):
            overlap = max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
            candidates.append((overlap, turn))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _best_speaker(
    candidates: list[tuple[float, Turn]], labels: dict[str, str]
) -> str | None:
    """Label of the most overlapping turn; None leaves the word unassigned."""
    return labels[candidates[0][1].speaker] if candidates else None


def _simultaneous_speakers(
    word: Word, regions: list[dict], region_starts: list[float]
) -> set[str]:
    # pragma: no mutate start -- swept regions are disjoint, so no earlier region reaches the word
    first = max(0, bisect.bisect_right(region_starts, word.start) - 1)
    # pragma: no mutate end
    speakers = set()
    # pragma: no mutate start -- swept regions are disjoint, so no earlier region reaches the word
    for i in range(first, len(regions)):
        # pragma: no mutate end
        region = regions[i]
        if region["start"] > word.end:
            break
        if _touches(word, region["start"], region["end"]):
            speakers.update(region["speakers"])
    return speakers


def _touches(word: Word, start: float, end: float) -> bool:
    """A word touches an interval it overlaps or, if zero-length, lies inside."""
    overlaps = min(word.end, end) > max(word.start, start)
    contains_instant = word.start == word.end and start <= word.start < end
    return overlaps or contains_instant


def _uncertainty_reasons(
    word: Word, candidates: list[tuple[float, Turn]], simultaneous: set[str]
) -> list[str]:
    checks = (
        ("no_matching_speaker", not candidates),
        ("invalid_word_duration", word.end <= word.start),
        ("estimated_or_fallback_timing", word.timing in UNRELIABLE_TIMINGS),
        ("ambiguous_speaker_boundary", _ambiguous_boundary(candidates)),
        ("overlapping_speech", bool(simultaneous)),
    )
    return [reason for reason, applies in checks if applies]


def _ambiguous_boundary(candidates: list[tuple[float, Turn]]) -> bool:
    """The runner-up turn overlaps the word nearly as much as the best one."""
    return (
        len(candidates) > 1
        and candidates[1][0] >= candidates[0][0] * AMBIGUOUS_OVERLAP_RATIO
    )


def _segments(words: list[dict]) -> list[dict]:
    """Join consecutive words of one speaker into readable segments."""
    segments = []
    for word in _words_with_text(words):
        if segments and _continues(segments[-1], word):
            _append_word(segments[-1], word)
        else:
            segments.append(_new_segment(word))
    return segments


def _words_with_text(words: list[dict]) -> list[dict]:
    return [w for w in words if w["text"]]


def _append_word(segment: dict, word: dict) -> None:
    segment["text"] += " " + word["text"]
    segment["end"] = max(segment["end"], word["end"])
    segment["uncertain"] |= word["uncertain"]
    segment["overlap"] |= word["overlap"]
    segment["word_ids"].append(word["word_id"])


def _new_segment(word: dict) -> dict:
    segment = {key: word[key] for key in SEGMENT_FIELDS}
    segment["word_ids"] = [word["word_id"]]
    return segment


def _continues(segment: dict, word: dict) -> bool:
    return (
        segment["speaker"] == word["speaker"]
        and word["start"] - segment["end"] < SEGMENT_PAUSE_SECONDS
        and word["end"] - segment["start"] < SEGMENT_MAX_SECONDS
    )


def _labelled(turns: list[Turn], labels: dict[str, str]) -> list[dict]:
    return [
        {"start": t.start, "end": t.end, "speaker": labels[t.speaker]} for t in turns
    ]
