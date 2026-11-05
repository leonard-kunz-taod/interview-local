"""Reproducible evaluation against the official CALLHOME TSV reference."""

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

from .llm import source_words, tokens
from .word_alignment import align_text_to_source

DER_COLLAR_SECONDS = 0.25
NORMALIZATION = (
    "Lowercase Unicode words/numbers; punctuation and noise "
    "labels removed; no number expansion"
)
CAVEAT = (
    "Single sample with dialect and overlapping speech; WER ordering "
    "is ambiguous during overlap."
)
DER_CONDITIONS = (
    "250-ms collar; overlap included; utterance-level reference "
    "intervals; optimal global label mapping"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("hypothesis", type=Path)
    parser.add_argument(
        "--offset",
        type=float,
        required=True,
        help="Original-recording offset to subtract from reference timestamps",
    )
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score(args.reference, args.hypothesis, args.offset, args.start, args.end)
    # pragma: no mutate start -- every report key and text is ASCII, so escaping non-ASCII changes nothing
    result = json.dumps(report, ensure_ascii=False, indent=2)
    # pragma: no mutate end
    print(result)
    if args.output:
        # pragma: no mutate start -- the report is ASCII, which every platform default and codec spelling writes identically
        args.output.write_text(result, encoding="utf-8")
        # pragma: no mutate end


def score(
    reference_file: Path, hypothesis_file: Path, offset: float, start: float, end: float
) -> dict:
    """WER, and with diarization DER and cpWER, inside one time window.

    Args:
        reference_file: CALLHOME TSV with Beg, End, Speaker and Text columns.
        hypothesis_file: Raw, speaker or readable transcript JSON produced by the app.
        offset: Original-recording offset subtracted from reference timestamps.
        start: Window start in seconds.
        end: Window end in seconds.

    Raises:
        ValueError: The window holds no reference words.
    """
    rows = read_reference(reference_file)
    hypothesis = json.loads(hypothesis_file.read_text(encoding="utf-8"))
    return score_transcript(rows, hypothesis, offset, start, end)


def read_reference(reference_file: Path) -> list[dict[str, str]]:
    """The rows of a CALLHOME TSV with Beg, End, Speaker and Text columns."""
    with reference_file.open(encoding="utf-8") as table:
        return list(csv.DictReader(table, delimiter="\t"))


def score_transcript(
    reference_rows: list[dict[str, str]],
    hypothesis: dict,
    offset: float,
    start: float,
    end: float,
) -> dict:
    """WER inside one time window; DER with diarization; cpWER with speakers.

    A readable transcript gets WER and cpWER but no DER: its paragraphs carry
    speakers, not diarization turns.

    Args:
        reference_rows: CALLHOME TSV rows, as returned by ``read_reference``.
        hypothesis: Raw, speaker or readable transcript produced by the app.
        offset: Original-recording offset subtracted from reference timestamps.
        start: Window start in seconds.
        end: Window end in seconds.

    Raises:
        ValueError: The window holds no reference words.
    """
    reference = _reference_utterances(reference_rows, offset, start, end)
    hyp = _hypothesis_words(hypothesis, start, end)
    report = _word_errors(reference, hyp)
    report.update(
        {
            "reference_offset_seconds": offset,
            "evaluated_start": start,
            "evaluated_end": end,
            "normalization": NORMALIZATION,
            "caveat": CAVEAT,
        }
    )
    if _is_readable(hypothesis):
        _add_cpwer(report, reference, hyp)
        return report
    if "diarization" not in hypothesis:
        return report
    report["der"] = _diarization_error(reference, hypothesis["diarization"], start, end)
    report["der_conditions"] = DER_CONDITIONS
    _add_cpwer(report, reference, hyp)
    return report


def _is_readable(transcript: dict) -> bool:
    return transcript.get("stage") == "readable"


def _add_cpwer(report: dict, reference: list[dict], hyp: list[dict]) -> None:
    """cpWER, when reference and hypothesis both have exactly two speakers."""
    ref_ids, hyp_ids = _speaker_ids(reference), _speaker_ids(hyp)
    if len(ref_ids) == len(hyp_ids) == 2:
        errors = _speaker_permuted_errors(reference, hyp, ref_ids, hyp_ids)
        report["cpwer"] = errors / report["reference_words"]


def normalize(text: str) -> list[str]:
    """Lowercase word tokens without punctuation, noise labels or markup."""
    text = re.sub(r"\{[^}]*\}|\[\[[^]]*\]\]|\[[^]]*\]", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    # pragma: no mutate start -- re.UNICODE is already the default for str patterns
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
    # pragma: no mutate end


def distance(reference: list[str], hypothesis: list[str]) -> int:
    """Word-level Levenshtein distance."""
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ref != hyp))
            )
        previous = current
    return previous[-1]


def _reference_utterances(
    rows: list[dict], offset: float, start: float, end: float
) -> list[dict]:
    """Reference utterances shifted by offset and clipped to the window."""
    reference = []
    for row in rows:
        begin, finish = float(row["Beg"]) - offset, float(row["End"]) - offset
        if finish > start and begin < end:
            reference.append(
                {
                    "start": max(start, begin),
                    "end": min(end, finish),
                    "text": row["Text"],
                    "speaker": row["Speaker"],
                }
            )
    return reference


def _hypothesis_words(hypothesis: dict, start: float, end: float) -> list[dict]:
    """Hypothesis words whose midpoint lies inside the window."""
    words = _transcript_words(hypothesis)
    return [w for w in words if start <= (w["start"] + w["end"]) / 2 < end]


def _transcript_words(hypothesis: dict) -> list[dict]:
    """The words of a readable transcript's paragraphs, the word list of a speaker
    transcript, or the segment words of a raw one."""
    if _is_readable(hypothesis):
        return _readable_words(hypothesis)
    if "words" in hypothesis:
        return hypothesis["words"]
    return _segment_words(hypothesis)


def _readable_words(readable: dict) -> list[dict]:
    """Paragraph words, timed by the source words they were aligned to."""
    words = source_words(readable["source_transcript"])
    return [
        word
        for paragraph in readable["paragraphs"]
        for word in _paragraph_words(paragraph, words)
    ]


def _paragraph_words(paragraph: dict, words: list[dict]) -> list[dict]:
    """Display words of one paragraph with the times of their source words.

    A paragraph text without word tokens contributes no words.
    """
    if not tokens(paragraph["text"]):
        return []
    selected = [words[i] for i in paragraph["source_word_ids"]]
    return _timed_display_words(paragraph, selected)


def _timed_display_words(paragraph: dict, selected: list[dict]) -> list[dict]:
    """The paragraph's display words, timed by the selected source words.

    A word the editor inserted takes the times of the preceding source word, or
    of the paragraph's first source word when nothing precedes it.
    """
    by_id = {w["word_id"]: w for w in selected}
    preceding = selected[0]
    timed = []
    for display_word in align_text_to_source(paragraph["text"], selected):
        ids = display_word.source_word_ids
        first, last = (by_id[ids[0]], by_id[ids[-1]]) if ids else (preceding, preceding)
        preceding = last
        timed.append(
            {
                "text": display_word.text,
                "start": first["start"],
                "end": last["end"],
                "speaker": paragraph["speaker"],
            }
        )
    return timed


def _segment_words(raw: dict) -> list[dict]:
    """The words of a raw transcript's segments."""
    return [w for s in raw["segments"] for w in s["words"]]


def _word_errors(reference: list[dict], hyp: list[dict]) -> dict:
    """WER of the time-ordered reference against the hypothesis words.

    Raises:
        ValueError: The window holds no reference words.
    """
    ordered = sorted(reference, key=lambda r: r["start"])
    ref_tokens = normalize(" ".join(r["text"] for r in ordered))
    hyp_tokens = normalize(" ".join(w["text"] for w in hyp))
    if not ref_tokens:
        raise ValueError("No reference words in the selected time window")
    edits = distance(ref_tokens, hyp_tokens)
    return {
        "reference_words": len(ref_tokens),
        "hypothesis_words": len(hyp_tokens),
        "word_errors": edits,
        "wer": edits / len(ref_tokens),
    }


def _diarization_error(
    reference: list[dict], turns: list[dict], start: float, end: float
) -> dict:
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = _reference_annotation(reference)
    predicted = _clipped_annotation(turns, start, end)
    # pragma: no mutate start -- skip_overlap=False is pyannote's default and is only read for truthiness
    metric = DiarizationErrorRate(collar=DER_COLLAR_SECONDS, skip_overlap=False)
    # pragma: no mutate end
    # pragma: no mutate start -- reference and hypothesis are already clipped to this window
    detail = metric(ref, predicted, uem=Timeline([Segment(start, end)]), detailed=True)
    # pragma: no mutate end
    return {str(k): float(v) for k, v in detail.items()}


def _reference_annotation(reference: list[dict]):
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for i, r in enumerate(reference):
        annotation[Segment(r["start"], r["end"]), i] = r["speaker"]
    return annotation


def _clipped_annotation(turns: list[dict], start: float, end: float):
    """Hypothesis turns that reach into the window, clipped to it."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation()
    for i, t in enumerate(turns):
        # pragma: no mutate start -- the metric's uem crops evaluation to this window anyway
        if t["end"] > start and t["start"] < end:
            # pragma: no mutate end
            clipped = Segment(max(start, t["start"]), min(end, t["end"]))
            annotation[clipped, i] = t["speaker"]
    return annotation


def _speaker_ids(items: list[dict]) -> list:
    """Sorted distinct speakers of utterances or words; unassigned ones excluded."""
    return sorted({x.get("speaker") for x in items if x.get("speaker") is not None})


def _speaker_permuted_errors(reference, hyp, ref_ids, hyp_ids) -> int:
    """Word errors under the best speaker mapping; unassigned words all count."""
    ref_tokens = [_speaker_tokens(reference, s) for s in ref_ids]
    hyp_tokens = [_speaker_tokens(hyp, s) for s in hyp_ids]
    best = min(
        _paired_distance(ref_tokens, mapped)
        for mapped in itertools.permutations(hyp_tokens)
    )
    return best + _unassigned_word_count(hyp)


def _speaker_tokens(items: list[dict], speaker) -> list[str]:
    """Normalized tokens of everything one speaker said, in order."""
    return normalize(" ".join(x["text"] for x in items if x.get("speaker") == speaker))


def _paired_distance(references: list[list[str]], hypotheses) -> int:
    return sum(distance(r, h) for r, h in zip(references, hypotheses))


def _unassigned_word_count(hyp: list[dict]) -> int:
    return sum(len(normalize(w["text"])) for w in hyp if w.get("speaker") is None)


if __name__ == "__main__":
    main()
