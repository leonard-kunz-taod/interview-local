from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from .core import Word, Turn, merge

STEPS = {"decode": "1/5 Decode audio (WAV, MP3 or M4A)",
         "asr": "2/5 Transcribe with Whisper large-v3 (CPU / INT8)",
         "align": "3/5 Align German words (CPU / Float32)",
         "diarize": "4/5 Identify two speakers (CPU / Float32)",
         "check-access": "Checking local community-1 model / Hugging Face access if needed"}

def worker(stage, source, target, status=print, token=None, offline=False):
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYANNOTE_METRICS_ENABLED"] = "0"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    tail = []
    with subprocess.Popen([sys.executable, "-m", "interview_local.worker", stage, str(source), str(target)],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as proc:
        try:
            for line in proc.stdout:
                line = line.strip()
                if env.get("HF_TOKEN"):
                    line = line.replace(env["HF_TOKEN"], "[Token redacted]")
                tail.append(line)
                tail = tail[-18:]
                if line.startswith(("Transcribed", "Aligned", "Loading",
                                    "Stage ", "Peak process RAM")):
                    status(line)
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if code:
        detail = "\n".join(tail)
        if stage == "decode":
            raise RuntimeError("The file could not be read as audio. Check that it plays in an audio player "
                               "and is not DRM-protected, or try a WAV file.\n"
                               f"Technical details:\n{detail}")
        raise RuntimeError(f"Stage '{stage}' failed.\n"
            "For 401/403: renew your HF login/read token and accept the community-1 conditions.\n"
            "For download errors, check your connection or model cache. If memory is low, close other apps.\n"
            f"Technical details:\n{detail}")

def process(source: Path, status=print, token=None, offline=False, alignment=False,
            output_dir: Path | None = None, raw_only=False) -> dict:
    from .artifacts import save_raw, save_speakers
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ValueError("The audio file was not found.")
    if source.suffix.lower() not in {".wav", ".mp3", ".m4a"}:
        raise ValueError("Please select a WAV, MP3 or M4A file.")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if any((output_dir / name).exists() for name in
               ("01_raw.json", "01_raw.txt", "02_speakers.json", "02_speakers.txt")):
            raise ValueError("Completed stage files already exist. Select a new output directory.")
    with tempfile.TemporaryDirectory(prefix="interview-local-") as directory:
        work = Path(directory)
        audio = work / "original.wav"
        for stage, src, dst in [("decode", source, audio), ("asr", audio, work / "asr.json")]:
            status(STEPS[stage])
            worker(stage, src, dst, status, token, offline)
        transcript = json.loads((work / "asr.json").read_text(encoding="utf-8"))
        raw = dict(schema_version=2, stage="raw", source=source.name, **transcript)
        if output_dir is not None:
            save_raw(raw, output_dir)
            status(f"Original speech recognition output saved: {output_dir / '01_raw.txt'}")
        if raw_only:
            return raw
        status(STEPS["check-access"])
        worker("check-access", source, work / "access", status, token, offline)
        if alignment:
            status(STEPS["align"])
            worker("align", audio, work / "words.json", status, token, offline)
            raw_words = json.loads((work / "words.json").read_text(encoding="utf-8"))
        else:
            status("3/5 Use Whisper word timestamps (no additional alignment)")
            from .backends import fallback_words
            raw_words = [dict(w) for segment in transcript["segments"] for w in fallback_words(segment)]
            for word in raw_words:
                if word["timing"] == "whisper_fallback":
                    word["timing"] = "whisper"
        status(STEPS["diarize"])
        worker("diarize", audio, work / "turns.json", status, token, offline)
        words = [Word(**w) for w in raw_words]
        diarization = json.loads((work / "turns.json").read_text(encoding="utf-8"))
        # Legacy fixtures and earlier cached outputs contain exclusive turns only.
        if isinstance(diarization, list):
            turns = [Turn(**t) for t in diarization]
            regular = None
        else:
            turns = [Turn(**t) for t in diarization["exclusive"]]
            regular = [Turn(**t) for t in diarization["regular"]]
        status("5/5 Merge words and speaker turns, including overlap markers")
        result = merge(words, turns, regular)
        result["stage"] = "speakers"
        result["raw_transcript"] = raw
        from collections import Counter
        result["metadata"] = dict(word_timing_counts=dict(Counter(w.timing for w in words)),
            source=source.name, transcription="faster-whisper/large-v3",
            diarization="pyannote/speaker-diarization-community-1", alignment=alignment,
            device="cpu", compute_type="int8 (ASR), float32 (diarization/alignment)",
            requested_speakers=2, detected_speakers=len(result["speakers"]),
            warnings=["Automatic transcript: review text and speaker changes before use."])
        if len(result["speakers"]) != 2:
            result["metadata"]["warnings"].append("The model did not detect exactly two speakers despite the requested count.")
        if output_dir is not None:
            save_speakers(result, output_dir)
        return result

def prepare(status=print, token=None, alignment=False):
    stages = ["prepare-diarize", "prepare-asr"]
    if alignment:
        stages.append("prepare-align")
    for stage in stages:
        status(f"Model download / loading check: {stage}")
        worker(stage, Path("."), Path("."), status, token)
