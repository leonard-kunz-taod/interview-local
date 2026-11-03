from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import bisect
import json
from pathlib import Path

WHISPER_TIMINGS = frozenset({"whisper", "whisper_fallback"})
AMBIGUOUS_OVERLAP_RATIO = .8
SEGMENT_MAX_PAUSE_SECONDS = 1.5
SEGMENT_MAX_SECONDS = 35


@dataclass
class Word:
    """A recognized word.

    ``probability`` is Whisper's word probability. It exists exactly for words
    with Whisper timing; aligned and estimated words carry ``None`` because no
    Whisper word corresponds to them.
    """
    start: float
    end: float
    text: str
    probability: float | None
    timing: str = "whisper"

    def __post_init__(self):
        has_whisper_timing = self.timing in WHISPER_TIMINGS
        if has_whisper_timing != (self.probability is not None):
            raise ValueError(f"Word '{self.text}' with timing '{self.timing}' has an inconsistent "
                             f"Whisper probability ({self.probability}).")

@dataclass
class Turn:
    start: float
    end: float
    speaker: str

@dataclass
class DiarizationResult:
    exclusive: list[Turn]
    regular: list[Turn]

class Transcriber(Protocol):
    def transcribe(self, audio: Path) -> dict: ...

class Diarizer(Protocol):
    def diarize(self, audio: Path) -> DiarizationResult: ...


def overlap_regions(turns: list[Turn], mapping: dict) -> list[dict]:
    """Sweep regular turns; duplicate tracks from one speaker are not overlap."""
    from collections import Counter, defaultdict
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
            if regions and regions[-1]["end"] == previous and regions[-1]["speakers"] == speakers:
                regions[-1]["end"] = current
            else:
                regions.append(dict(start=previous, end=current, speakers=speakers))
        active.update(changes)
        previous = current
    return regions


def merge(words: list[Word], turns: list[Turn], regular_turns: list[Turn] | None = None) -> dict:
    """Assign words by exclusive overlap; separately retain simultaneous voices."""
    turns = sorted(turns, key=lambda t: (t.start, t.end))
    regular = regular_turns or []
    mapping = _speaker_letters(turns, regular)
    regions = overlap_regions(regular, mapping)
    assigner = _WordAssigner(turns, regions, mapping)
    assigned = [assigner.assign(word_id, word) for word_id, word in enumerate(words)]
    return dict(schema_version=2, language="de",
        speakers={letter: f"Speaker {letter}" for letter in mapping.values()},
        segments=_group_segments(assigned), words=assigned, diarization=_lettered_turns(turns, mapping),
        regular_diarization=_lettered_turns(regular, mapping), overlaps=regions,
        overlap_detection_available=regular_turns is not None)


def _speaker_letters(turns: list[Turn], regular_turns: list[Turn]) -> dict[str, str]:
    """Letter A, B, … per diarization label: exclusive turn order first, then speakers heard only in regular turns."""
    exclusive_labels = [t.speaker for t in turns]
    regular_labels = [t.speaker for t in sorted(regular_turns, key=lambda t: t.start)]
    labels = dict.fromkeys(exclusive_labels + regular_labels)
    return {label: chr(65 + i) for i, label in enumerate(labels)}


class _WordAssigner:
    """Assigns words to the speaker of the most overlapping exclusive turn and marks simultaneous speech."""

    def __init__(self, turns: list[Turn], regions: list[dict], mapping: dict[str, str]):
        self.turns = turns
        self.turn_starts = [t.start for t in turns]
        self.regions = regions
        self.region_starts = [r["start"] for r in regions]
        self.mapping = mapping

    def assign(self, word_id: int, word: Word) -> dict:
        candidates = self._speaker_candidates(word)
        simultaneous = self._simultaneous_speakers(word)
        speaker = self.mapping[candidates[0][1].speaker] if candidates else None
        reasons = _uncertainty_reasons(word, candidates, simultaneous)
        return dict(word_id=word_id, start=word.start, end=word.end, text=word.text.strip(),
            speaker=speaker, timing=word.timing, probability=word.probability,
            uncertain=bool(reasons), uncertainty_reasons=reasons,
            overlap=bool(simultaneous), overlap_speakers=sorted(simultaneous))

    def _speaker_candidates(self, word: Word) -> list[tuple[float, Turn]]:
        """Turns touching the word with their overlap in seconds, largest overlap first."""
        nearby = _spans_near(word, self.turns, self.turn_starts)
        candidates = [(_overlap_seconds(word, t.start, t.end), t) for t in nearby if _touches(word, t.start, t.end)]
        return sorted(candidates, key=lambda item: item[0], reverse=True)

    def _simultaneous_speakers(self, word: Word) -> set[str]:
        nearby = _spans_near(word, self.regions, self.region_starts)
        return {speaker for r in nearby if _touches(word, r["start"], r["end"]) for speaker in r["speakers"]}


def _spans_near(word: Word, spans: list, starts: list[float]) -> list:
    """Spans sorted by start, from the last one starting at or before the word through the last one starting
    no later than its end."""
    first = max(0, bisect.bisect_right(starts, word.start) - 1)
    return spans[first:bisect.bisect_right(starts, word.end)]


def _overlap_seconds(word: Word, start: float, end: float) -> float:
    return max(0., min(word.end, end) - max(word.start, start))


def _touches(word: Word, start: float, end: float) -> bool:
    """Whether a time span overlaps the word or, for a zero-length word, contains its instant."""
    contains_instant = word.start == word.end and start <= word.start < end
    return _overlap_seconds(word, start, end) > 0 or contains_instant


def _uncertainty_reasons(word: Word, candidates: list[tuple[float, Turn]], simultaneous: set[str]) -> list[str]:
    checks = [("no_matching_speaker", not candidates),
              ("invalid_word_duration", word.end <= word.start),
              ("estimated_or_fallback_timing", word.timing in {"estimated", "whisper_fallback"}),
              ("ambiguous_speaker_boundary", _is_ambiguous(candidates)),
              ("overlapping_speech", bool(simultaneous))]
    return [reason for reason, applies in checks if applies]


def _is_ambiguous(candidates: list[tuple[float, Turn]]) -> bool:
    """Whether the runner-up turn overlaps the word nearly as much as the best one."""
    return len(candidates) > 1 and candidates[1][0] >= candidates[0][0] * AMBIGUOUS_OVERLAP_RATIO


def _group_segments(words: list[dict]) -> list[dict]:
    """Join consecutive non-empty words of one speaker into segments."""
    segments = []
    for word in words:
        if not word["text"]:
            continue
        if _continues_segment(segments, word):
            _extend_segment(segments[-1], word)
        else:
            segments.append(_start_segment(word))
    return segments


def _continues_segment(segments: list[dict], word: dict) -> bool:
    """Whether the word belongs to the last segment: same speaker, short pause, segment not too long."""
    return (bool(segments) and segments[-1]["speaker"] == word["speaker"]
            and word["start"] - segments[-1]["end"] < SEGMENT_MAX_PAUSE_SECONDS
            and word["end"] - segments[-1]["start"] < SEGMENT_MAX_SECONDS)


def _extend_segment(segment: dict, word: dict) -> None:
    segment["text"] += " " + word["text"]
    segment["end"] = max(segment["end"], word["end"])
    segment["uncertain"] |= word["uncertain"]
    segment["overlap"] |= word["overlap"]
    segment["word_ids"].append(word["word_id"])


def _start_segment(word: dict) -> dict:
    segment = {k: word[k] for k in ("start", "end", "text", "speaker", "uncertain", "overlap")}
    segment["word_ids"] = [word["word_id"]]
    return segment


def _lettered_turns(turns: list[Turn], mapping: dict[str, str]) -> list[dict]:
    return [dict(start=t.start, end=t.end, speaker=mapping[t.speaker]) for t in turns]


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
        lines.append(f"[{timestamp(s['start'])} – {timestamp(s['end'])}] {name}{flag}\n{s['text']}\n")
    return "\n".join(lines) or "No speech detected."


def export(result: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "transcript.txt").write_text(render(result), encoding="utf-8")
