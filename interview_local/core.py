from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass

AMBIGUOUS_OVERLAP_RATIO = 0.8
SEGMENT_PAUSE_SECONDS = 1.5
SEGMENT_MAX_SECONDS = 35
UNRELIABLE_TIMINGS = {"estimated", "whisper_fallback"}
SEGMENT_FIELDS = ("start", "end", "text", "speaker", "uncertain", "overlap")


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
    words: list[Word], turns: list[Turn], regular_turns: list[Turn] | None = None
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


def overlap_regions(turns: list[Turn], mapping: dict) -> list[dict]:
    """Sweep regular turns; duplicate tracks from one speaker are not overlap."""
    events = defaultdict(Counter)
    for turn in turns:
        if turn.end > turn.start:
            events[turn.start][turn.speaker] += 1
            events[turn.end][turn.speaker] -= 1
    active = Counter()
    regions = []
    previous = None
    for current, changes in sorted(events.items()):
        speakers = sorted(mapping[s] for s, count in active.items() if count > 0)
        if previous is not None and current > previous and len(speakers) > 1:
            if (
                regions
                and regions[-1]["end"] == previous
                and regions[-1]["speakers"] == speakers
            ):
                regions[-1]["end"] = current
            else:
                regions.append(
                    {"start": previous, "end": current, "speakers": speakers}
                )
        active.update(changes)
        previous = current
    return regions


def segment_words(segment: dict, timing: str) -> list[dict]:
    """Words of an ASR segment labelled with the given timing source.

    A segment without word timestamps becomes one word with estimated timing.
    """
    if segment.get("words"):
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
    lines = []
    for s in result["segments"]:
        name = result["speakers"].get(s["speaker"], "Unassigned")
        flags = []
        if s.get("uncertain"):
            flags.append("Review assignment")
        if s.get("overlap"):
            flags.append("Overlapping speech")
        flag = f" [{'; '.join(flags)}]" if flags else ""
        time_range = f"{timestamp(s['start'])} – {timestamp(s['end'])}"
        lines.append(f"[{time_range}] {name}{flag}\n{s['text']}\n")
    return "\n".join(lines) or "No speech detected."


def _speaker_labels(turns: list[Turn], regular_turns: list[Turn]) -> dict:
    """Letter speakers in order of their first exclusive, then regular, turn."""
    ids = list(dict.fromkeys(t.speaker for t in turns))
    for turn in sorted(regular_turns, key=lambda t: t.start):
        if turn.speaker not in ids:
            ids.append(turn.speaker)
    return {speaker: chr(ord("A") + i) for i, speaker in enumerate(ids)}


def _attributed_words(
    words: list[Word], turns: list[Turn], regions: list[dict], labels: dict
) -> list[dict]:
    turn_starts = [t.start for t in turns]
    region_starts = [r["start"] for r in regions]
    attributed = []
    for word_id, word in enumerate(words):
        candidates = _ranked_candidates(word, turns, turn_starts)
        simultaneous = _simultaneous_speakers(word, regions, region_starts)
        reasons = _uncertainty_reasons(word, candidates, simultaneous)
        speaker = labels[candidates[0][1].speaker] if candidates else None
        attributed.append(
            {
                "word_id": word_id,
                "start": word.start,
                "end": word.end,
                "text": word.text.strip(),
                "speaker": speaker,
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
    first = max(0, bisect.bisect_right(turn_starts, word.start) - 1)
    candidates = []
    for i in range(first, len(turns)):
        turn = turns[i]
        if turn.start > word.end:
            break
        if _touches(word, turn.start, turn.end):
            overlap = max(0.0, min(word.end, turn.end) - max(word.start, turn.start))
            candidates.append((overlap, turn))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _simultaneous_speakers(
    word: Word, regions: list[dict], region_starts: list[float]
) -> set[str]:
    first = max(0, bisect.bisect_right(region_starts, word.start) - 1)
    speakers = set()
    for i in range(first, len(regions)):
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
    reasons = []
    if not candidates:
        reasons.append("no_matching_speaker")
    if word.end <= word.start:
        reasons.append("invalid_word_duration")
    if word.timing in UNRELIABLE_TIMINGS:
        reasons.append("estimated_or_fallback_timing")
    if (
        len(candidates) > 1
        and candidates[1][0] >= candidates[0][0] * AMBIGUOUS_OVERLAP_RATIO
    ):
        reasons.append("ambiguous_speaker_boundary")
    if simultaneous:
        reasons.append("overlapping_speech")
    return reasons


def _segments(words: list[dict]) -> list[dict]:
    """Join consecutive words of one speaker into readable segments."""
    segments = []
    for word in words:
        if not word["text"]:
            continue
        if segments and _continues(segments[-1], word):
            segment = segments[-1]
            segment["text"] += " " + word["text"]
            segment["end"] = max(segment["end"], word["end"])
            segment["uncertain"] |= word["uncertain"]
            segment["overlap"] |= word["overlap"]
            segment["word_ids"].append(word["word_id"])
        else:
            segment = {key: word[key] for key in SEGMENT_FIELDS}
            segment["word_ids"] = [word["word_id"]]
            segments.append(segment)
    return segments


def _continues(segment: dict, word: dict) -> bool:
    return (
        segment["speaker"] == word["speaker"]
        and word["start"] - segment["end"] < SEGMENT_PAUSE_SECONDS
        and word["end"] - segment["start"] < SEGMENT_MAX_SECONDS
    )


def _labelled(turns: list[Turn], labels: dict) -> list[dict]:
    return [
        {"start": t.start, "end": t.end, "speaker": labels[t.speaker]} for t in turns
    ]
