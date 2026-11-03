"""Reproducible evaluation against the official CALLHOME TSV reference."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import re


def normalize(text):
    text = re.sub(r"\{[^}]*\}|\[\[[^]]*\]\]|\[[^]]*\]", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    return re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1,
                               previous[j-1] + (ref != hyp)))
        previous = current
    return previous[-1]


def score(reference_file, hypothesis_file, offset, start, end):
    rows = list(csv.DictReader(Path(reference_file).open(encoding="utf-8"), delimiter="\t"))
    reference = []
    for row in rows:
        begin, finish = float(row["Beg"]) - offset, float(row["End"]) - offset
        if finish > start and begin < end:
            # Choose evaluation boundaries that include complete reference utterances.
            reference.append(dict(start=max(start, begin), end=min(end, finish),
                                  text=row["Text"], speaker=row["Speaker"]))
    hypothesis = json.loads(Path(hypothesis_file).read_text(encoding="utf-8"))
    words = hypothesis.get("words")
    if words is not None:
        hyp = [w for w in words if start <= (w["start"] + w["end"]) / 2 < end]
    else:
        hyp = [w for s in hypothesis["segments"] for w in s.get("words", [])
               if start <= (w["start"] + w["end"]) / 2 < end]
    ref_tokens = normalize(" ".join(r["text"] for r in sorted(reference, key=lambda r: r["start"])))
    hyp_tokens = normalize(" ".join(w["text"] for w in hyp))
    if not ref_tokens:
        raise ValueError("No reference words in the selected time window")
    edits = distance(ref_tokens, hyp_tokens)
    report = dict(reference_words=len(ref_tokens), hypothesis_words=len(hyp_tokens),
                  word_errors=edits, wer=edits/len(ref_tokens), reference_offset_seconds=offset,
                  evaluated_start=start, evaluated_end=end,
                  normalization="Lowercase Unicode words/numbers; punctuation and noise labels removed; no number expansion",
                  caveat="Single sample with dialect and overlapping speech; WER ordering is ambiguous during overlap.")
    if hypothesis.get("diarization") is not None:
        from pyannote.core import Annotation, Segment, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate
        ref = Annotation()
        for i, r in enumerate(reference):
            ref[Segment(r["start"], r["end"]), i] = r["speaker"]
        predicted = Annotation()
        for i, t in enumerate(hypothesis["diarization"]):
            if t["end"] > start and t["start"] < end:
                predicted[Segment(max(start,t["start"]),min(end,t["end"])), i] = t["speaker"]
        metric = DiarizationErrorRate(collar=.25, skip_overlap=False)
        detail = metric(ref, predicted, uem=Timeline([Segment(start,end)]), detailed=True)
        report["der"] = {str(k): float(v) for k,v in detail.items()}
        report["der_conditions"] = "250-ms collar; overlap included; utterance-level reference intervals; optimal global label mapping"
        ref_ids = sorted({r["speaker"] for r in reference})
        hyp_ids = sorted({w.get("speaker") for w in hyp if w.get("speaker") is not None})
        if len(ref_ids) == 2 and len(hyp_ids) == 2:
            ref_by = {s: normalize(" ".join(r["text"] for r in reference if r["speaker"] == s)) for s in ref_ids}
            hyp_by = {s: normalize(" ".join(w["text"] for w in hyp if w.get("speaker") == s)) for s in hyp_ids}
            unassigned = sum(len(normalize(w["text"])) for w in hyp if w.get("speaker") is None)
            best = min(sum(distance(ref_by[r],hyp_by[h]) for r,h in zip(ref_ids,perm)) + unassigned
                       for perm in itertools.permutations(hyp_ids))
            report["cpwer"] = best / len(ref_tokens)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference", type=Path)
    p.add_argument("hypothesis", type=Path)
    p.add_argument("--offset", type=float, required=True, help="Original-recording offset to subtract from reference timestamps")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    report = score(args.reference,args.hypothesis,args.offset,args.start,args.end)
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result,encoding="utf-8")

if __name__ == "__main__":
    main()
