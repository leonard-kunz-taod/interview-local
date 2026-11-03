"""Sequential speech stages, each in its own worker process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .artifacts import save_raw, save_speakers
from .core import Turn, Word, merge, segment_words
from .models import DIARIZATION_ID, WHISPER_MODEL

STEPS = {
    "decode": "1/5 Decode audio (WAV, MP3 or M4A)",
    "asr": "2/5 Transcribe with Whisper large-v3 (CPU / INT8)",
    "align": "3/5 Align German words (CPU / Float32)",
    "diarize": "4/5 Identify two speakers (CPU / Float32)",
}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a"}
COMPLETED_STAGE_FILES = (
    "01_raw.json",
    "01_raw.txt",
    "02_speakers.json",
    "02_speakers.txt",
)
FORWARDED_OUTPUT = ("Transcribed", "Aligned", "Loading", "Stage ", "Peak process RAM")
FAILURE_DETAIL_LINES = 18
WORKSPACE_PREFIX = "interview-local-"
REQUESTED_SPEAKERS = 2


def transcribe(source: Path, status: Callable, offline: bool, output_dir: Path) -> dict:
    """Stage 1 only: speech recognition without diarization, LLM or HF account.

    Args:
        source: WAV, MP3 or M4A recording.
        status: Receives progress messages.
        offline: Use local model caches only.
        output_dir: Directory for the 01_raw.* files.

    Returns:
        The raw speech recognition transcript.
    """
    source = _checked_audio(source)
    output_dir = _open_output(output_dir)
    with tempfile.TemporaryDirectory(prefix=WORKSPACE_PREFIX) as directory:
        run = _Run(Path(directory), status, None, offline, output_dir)
        return _recognize_speech(source, run)


def process(
    source: Path,
    status=print,
    token=None,
    offline=False,
    alignment=False,
    output_dir: Path | None = None,
) -> dict:
    """Speech recognition, then two-speaker attribution, saving each stage.

    Args:
        source: WAV, MP3 or M4A recording.
        status: Receives progress messages.
        token: Hugging Face read token for community-1, if needed.
        offline: Use local model caches only.
        alignment: Run the optional German word alignment.
        output_dir: Directory for the stage files; None keeps nothing on disk.

    Returns:
        The speaker-attributed transcript (stage 2).
    """
    source = _checked_audio(source)
    if output_dir is not None:
        output_dir = _open_output(output_dir)
    with tempfile.TemporaryDirectory(prefix=WORKSPACE_PREFIX) as directory:
        run = _Run(Path(directory), status, token, offline, output_dir)
        raw = _recognize_speech(source, run)
        status("Checking local community-1 model / Hugging Face access if needed")
        run.stage("check-access", source, run.work / "access")
        timed_words = _aligned_words(run) if alignment else _whisper_words(raw, run)
        turns, regular = _diarization(run)
        words = [Word(**w) for w in timed_words]
        status("5/5 Merge words and speaker turns, including overlap markers")
        result = merge(words, turns, regular)
        result["stage"] = "speakers"
        result["raw_transcript"] = raw
        result["metadata"] = _metadata(
            source, words, alignment, len(result["speakers"])
        )
        if output_dir is not None:
            save_speakers(result, output_dir)
        return result


def prepare(status=print, token=None, alignment=False) -> None:
    """Download or verify the speech models, one worker process per model."""
    stages = ["prepare-diarize", "prepare-asr"]
    if alignment:
        stages.append("prepare-align")
    for stage in stages:
        status(f"Model download / loading check: {stage}")
        worker(stage, Path("."), Path("."), status, token)


def worker(stage, source, target, status=print, token=None, offline=False) -> None:
    """Run one stage in a fresh process so its model memory is released on exit.

    Raises:
        RuntimeError: The stage failed; the message ends with its last output lines.
    """
    env = _worker_environment(token, offline)
    secret = env.get("HF_TOKEN")
    tail = []
    command = [sys.executable, "-m", "interview_local.worker", stage]
    with subprocess.Popen(
        [*command, str(source), str(target)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proc:
        try:
            for line in proc.stdout:
                line = line.strip()
                if secret:
                    line = line.replace(secret, "[Token redacted]")
                tail = [*tail, line][-FAILURE_DETAIL_LINES:]
                if line.startswith(FORWARDED_OUTPUT):
                    status(line)
            code = proc.wait()
        except BaseException:
            _terminate(proc)
            raise
    if code:
        detail = "\n".join(tail)
        raise RuntimeError(
            f"Stage '{stage}' failed.\n"
            "For 401/403: renew your HF login/read token and accept the "
            "community-1 conditions.\n"
            "For download errors, check your connection or model cache. "
            "If memory is low, close other apps.\n"
            f"Technical details:\n{detail}"
        )


@dataclass
class _Run:
    """One processing run: its temporary workspace and worker settings."""

    work: Path
    status: Callable
    token: str | None
    offline: bool
    output_dir: Path | None

    @property
    def audio(self) -> Path:
        return self.work / "original.wav"

    def stage(self, stage: str, source: Path, target: Path) -> None:
        worker(stage, source, target, self.status, self.token, self.offline)


def _checked_audio(source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ValueError("The audio file was not found.")
    if source.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError("Please select a WAV, MP3 or M4A file.")
    return source


def _open_output(output_dir: Path) -> Path:
    """Create the output directory; refuse one that already holds stage files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in COMPLETED_STAGE_FILES):
        raise ValueError(
            "Completed stage files already exist. Select a new output directory."
        )
    return output_dir


def _recognize_speech(source: Path, run: _Run) -> dict:
    """Decode and transcribe; the raw stage is saved before later stages run."""
    asr_output = run.work / "asr.json"
    for stage, src, dst in [
        ("decode", source, run.audio),
        ("asr", run.audio, asr_output),
    ]:
        run.status(STEPS[stage])
        run.stage(stage, src, dst)
    transcript = _read_json(asr_output)
    raw = dict(schema_version=2, stage="raw", source=source.name, **transcript)
    if run.output_dir is not None:
        save_raw(raw, run.output_dir)
        saved = run.output_dir / "01_raw.txt"
        run.status(f"Original speech recognition output saved: {saved}")
    return raw


def _aligned_words(run: _Run) -> list[dict]:
    run.status(STEPS["align"])
    run.stage("align", run.audio, run.work / "words.json")
    return _read_json(run.work / "words.json")


def _whisper_words(raw: dict, run: _Run) -> list[dict]:
    run.status("3/5 Use Whisper word timestamps (no additional alignment)")
    return [w for s in raw["segments"] for w in segment_words(s, "whisper")]


def _diarization(run: _Run) -> tuple[list[Turn], list[Turn] | None]:
    """Exclusive turns for assignment and regular turns for overlap detection."""
    run.status(STEPS["diarize"])
    run.stage("diarize", run.audio, run.work / "turns.json")
    diarization = _read_json(run.work / "turns.json")
    # Legacy fixtures and earlier cached outputs contain exclusive turns only.
    if isinstance(diarization, list):
        return [Turn(**t) for t in diarization], None
    exclusive = [Turn(**t) for t in diarization["exclusive"]]
    return exclusive, [Turn(**t) for t in diarization["regular"]]


def _metadata(source: Path, words: list[Word], alignment: bool, detected: int) -> dict:
    warnings = ["Automatic transcript: review text and speaker changes before use."]
    if detected != REQUESTED_SPEAKERS:
        warnings.append(
            "The model did not detect exactly two speakers despite the requested count."
        )
    return {
        "word_timing_counts": dict(Counter(w.timing for w in words)),
        "source": source.name,
        "transcription": f"faster-whisper/{WHISPER_MODEL}",
        "diarization": DIARIZATION_ID,
        "alignment": alignment,
        "device": "cpu",
        "compute_type": "int8 (ASR), float32 (diarization/alignment)",
        "requested_speakers": REQUESTED_SPEAKERS,
        "detected_speakers": detected,
        "warnings": warnings,
    }


def _worker_environment(token: str | None, offline: bool) -> dict:
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYANNOTE_METRICS_ENABLED"] = "0"
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
