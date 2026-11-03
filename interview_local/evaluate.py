"""Reproducible evaluation against the official CALLHOME TSV reference."""

import argparse
import csv
import itertools
import json
import re
from pathlib import Path

DER_COLLAR_SECONDS = 0.25


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
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result, encoding="utf-8")


def score(reference_file, hypothesis_file, offset, start, end) -> dict:
    """WER, and with diarization DER and cpWER, inside one time window.

    Args:
        reference_file: CALLHOME TSV with Beg, End, Speaker and Text columns.
        hypothesis_file: Raw or speaker transcript JSON produced by the app.
        offset: Original-recording offset subtracted from reference timestamps.
        start: Window start in seconds.
        end: Window end in seconds.

    Raises:
        ValueError: The window holds no reference words.
    """
    with Path(reference_file).open(encoding="utf-8") as table:
        rows = list(csv.DictReader(table, delimiter="\t"))
    reference = _reference_utterances(rows, offset, start, end)
    hypothesis = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))
    hyp = _hypothesis_words(hypothesis, start, end)
    ordered = sorted(reference, key=lambda r: r["start"])
    ref_tokens = normalize(" ".join(r["text"] for r in ordered))
    hyp_tokens = normalize(" ".join(w["text"] for w in hyp))
    if not ref_tokens:
        raise ValueError("No reference words in the selected time window")
    edits = distance(ref_tokens, hyp_tokens)
    report = {
        "reference_words": len(ref_tokens),
        "hypothesis_words": len(hyp_tokens),
        "word_errors": edits,
        "wer": edits / len(ref_tokens),
        "reference_offset_seconds": offset,
        "evaluated_start": start,
        "evaluated_end": end,
        "normalization": "Lowercase Unicode words/numbers; punctuation and noise "
        "labels removed; no number expansion",
        "caveat": "Single sample with dialect and overlapping speech; WER ordering "
        "is ambiguous during overlap.",
    }
    if hypothesis.get("diarization") is not None:
        report["der"] = _diarization_error(
            reference, hypothesis["diarization"], start, end
        )
        report["der_conditions"] = (
            "250-ms collar; overlap included; utterance-level reference "
            "intervals; optimal global label mapping"
        )
        ref_ids = sorted({r["speaker"] for r in reference})
        hyp_ids = sorted(
            {w.get("speaker") for w in hyp if w.get("speaker") is not None}
        )
        if len(ref_ids) == 2 and len(hyp_ids) == 2:
            errors = _speaker_permuted_errors(reference, hyp, ref_ids, hyp_ids)
            report["cpwer"] = errors / len(ref_tokens)
    return report


def normalize(text: str) -> list[str]:
    """Lowercase word tokens without punctuation, noise labels or markup."""
    text = re.sub(r"\{[^}]*\}|\[\[[^]]*\]\]|\[[^]]*\]", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


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


def _reference_utterances(rows: list[dict], offset, start, end) -> list[dict]:
    """Reference utterances shifted by offset and clipped to the window."""
    reference = []
    for row in rows:
        begin, finish = float(row["Beg"]) - offset, float(row["End"]) - offset
        if finish > start and begin < end:
            # Choose evaluation boundaries that include complete reference utterances.
            reference.append(
                {
                    "start": max(start, begin),
                    "end": min(end, finish),
                    "text": row["Text"],
                    "speaker": row["Speaker"],
                }
            )
    return reference


def _hypothesis_words(hypothesis: dict, start, end) -> list[dict]:
    """Hypothesis words whose midpoint lies inside the window."""
    words = hypothesis.get("words")
    if words is None:
        words = [w for s in hypothesis["segments"] for w in s.get("words", [])]
    return [w for w in words if start <= (w["start"] + w["end"]) / 2 < end]


def _diarization_error(reference: list[dict], turns: list[dict], start, end) -> dict:
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = Annotation()
    for i, r in enumerate(reference):
        ref[Segment(r["start"], r["end"]), i] = r["speaker"]
    predicted = Annotation()
    for i, t in enumerate(turns):
        if t["end"] > start and t["start"] < end:
            clipped = Segment(max(start, t["start"]), min(end, t["end"]))
            predicted[clipped, i] = t["speaker"]
    metric = DiarizationErrorRate(collar=DER_COLLAR_SECONDS, skip_overlap=False)
    detail = metric(ref, predicted, uem=Timeline([Segment(start, end)]), detailed=True)
    return {str(k): float(v) for k, v in detail.items()}


def _speaker_permuted_errors(reference, hyp, ref_ids, hyp_ids) -> int:
    """Word errors under the best speaker mapping; unassigned words all count."""
    ref_by = {
        s: normalize(" ".join(r["text"] for r in reference if r["speaker"] == s))
        for s in ref_ids
    }
    hyp_by = {
        s: normalize(" ".join(w["text"] for w in hyp if w.get("speaker") == s))
        for s in hyp_ids
    }
    unassigned = sum(len(normalize(w["text"])) for w in hyp if w.get("speaker") is None)
    return min(
        sum(distance(ref_by[r], hyp_by[h]) for r, h in zip(ref_ids, perm)) + unassigned
        for perm in itertools.permutations(hyp_ids)
    )


if __name__ == "__main__":
    main()
