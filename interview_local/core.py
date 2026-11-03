from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import bisect
import json
from pathlib import Path

WHISPER_TIMINGS = frozenset({"whisper", "whisper_fallback"})


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
    ids = list(dict.fromkeys(t.speaker for t in turns))
    for t in sorted(regular_turns or [], key=lambda t: t.start):
        if t.speaker not in ids:
            ids.append(t.speaker)
    mapping = {s: chr(65 + i) for i, s in enumerate(ids)}
    regions = overlap_regions(regular_turns or [], mapping)
    region_starts = [r["start"] for r in regions]
    starts = [t.start for t in turns]
    assigned = []
    for word_id, word in enumerate(words):
        begin = max(0, bisect.bisect_right(starts, word.start) - 1)
        candidates = []
        for i in range(begin, len(turns)):
            t = turns[i]
            if t.start > word.end:
                break
            overlap = max(0., min(word.end, t.end) - max(word.start, t.start))
            if overlap > 0 or (word.start == word.end and t.start <= word.start < t.end):
                candidates.append((overlap, t))
        candidates.sort(key=lambda item: item[0], reverse=True)
        speaker = mapping[candidates[0][1].speaker] if candidates else None
        simultaneous = set()
        region_index = max(0, bisect.bisect_right(region_starts, word.start) - 1)
        for i in range(region_index, len(regions)):
            r = regions[i]
            if r["start"] > word.end:
                break
            if (min(word.end, r["end"]) > max(word.start, r["start"]) or
                    (word.start == word.end and r["start"] <= word.start < r["end"])):
                simultaneous.update(r["speakers"])
        reasons = []
        if not candidates:
            reasons.append("no_matching_speaker")
        if word.end <= word.start:
            reasons.append("invalid_word_duration")
        if word.timing in {"estimated", "whisper_fallback"}:
            reasons.append("estimated_or_fallback_timing")
        if len(candidates) > 1 and candidates[1][0] >= candidates[0][0] * .8:
            reasons.append("ambiguous_speaker_boundary")
        if simultaneous:
            reasons.append("overlapping_speech")
        assigned.append(dict(word_id=word_id, start=word.start, end=word.end, text=word.text.strip(),
            speaker=speaker, timing=word.timing, probability=word.probability,
            uncertain=bool(reasons), uncertainty_reasons=reasons,
            overlap=bool(simultaneous), overlap_speakers=sorted(simultaneous)))
    segments = []
    for w in assigned:
        if not w["text"]:
            continue
        if (segments and segments[-1]["speaker"] == w["speaker"]
                and w["start"] - segments[-1]["end"] < 1.5
                and w["end"] - segments[-1]["start"] < 35):
            s = segments[-1]
            s["text"] += " " + w["text"]
            s["end"] = max(s["end"], w["end"])
            s["uncertain"] |= w["uncertain"]
            s["overlap"] |= w["overlap"]
            s["word_ids"].append(w["word_id"])
        else:
            segment = {k: w[k] for k in ("start", "end", "text", "speaker", "uncertain", "overlap")}
            segment["word_ids"] = [w["word_id"]]
            segments.append(segment)
    def serialize(ts):
        return [dict(start=t.start, end=t.end, speaker=mapping[t.speaker]) for t in ts]
    return dict(schema_version=2, language="de", speakers={mapping[s]: f"Speaker {mapping[s]}" for s in ids},
        segments=segments, words=assigned, diarization=serialize(turns),
        regular_diarization=serialize(regular_turns or []), overlaps=regions,
        overlap_detection_available=regular_turns is not None)


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
