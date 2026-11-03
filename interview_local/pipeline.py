from __future__ import annotations
from collections import Counter, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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

OFFLINE_ENVIRONMENT = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
TELEMETRY_OFF_ENVIRONMENT = {"HF_HUB_DISABLE_TELEMETRY": "1", "PYANNOTE_METRICS_ENABLED": "0"}
PROGRESS_PREFIXES = ("Transcribed", "Aligned", "Loading", "Stage ", "Peak process RAM")
FAILURE_DETAIL_LINES = 18
STOP_TIMEOUT_SECONDS = 10
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a"}
STAGE_FILENAMES = ("01_raw.json", "01_raw.txt", "02_speakers.json", "02_speakers.txt")
DECODED_AUDIO = "original.wav"
REQUESTED_SPEAKERS = 2


def worker(stage: str, source: Path, target: Path, status: Callable[[str], None], token: str | None,
           offline: bool = False) -> None:
    env = worker_environment(token)
    if offline:
        env.update(OFFLINE_ENVIRONMENT)
    with subprocess.Popen([sys.executable, "-m", "interview_local.worker", stage, str(source), str(target)],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace") as proc:
        try:
            tail = relay_output(proc.stdout, env.get("HF_TOKEN"), status)
            code = proc.wait()
        except BaseException:
            stop(proc)
            raise
    if code:
        raise stage_failure(stage, "\n".join(tail))


def worker_environment(token: str | None) -> dict[str, str]:
    """The model worker's environment: this process's, with the HF token if given and telemetry off."""
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
    env.update(TELEMETRY_OFF_ENVIRONMENT)
    return env


def relay_output(lines: Iterable[str], secret: str | None, status: Callable[[str], None]) -> list[str]:
    """Pass progress lines on to status and return the last output lines, with the secret redacted."""
    tail = deque(maxlen=FAILURE_DETAIL_LINES)
    for line in lines:
        line = redact(line.strip(), secret)
        tail.append(line)
        if line.startswith(PROGRESS_PREFIXES):
            status(line)
    return list(tail)


def redact(line: str, secret: str | None) -> str:
    return line.replace(secret, "[Token redacted]") if secret else line


def stop(proc: subprocess.Popen) -> None:
    """Terminate the worker; kill it if it does not exit in time."""
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def stage_failure(stage: str, detail: str) -> RuntimeError:
    if stage == "decode":
        return RuntimeError("The file could not be read as audio. Check that it plays in an audio player "
                            "and is not DRM-protected, or try a WAV file.\n"
                            f"Technical details:\n{detail}")
    return RuntimeError(f"Stage '{stage}' failed.\n"
        "For 401/403: renew your HF login/read token and accept the community-1 conditions.\n"
        "For download errors, check your connection or model cache. If memory is low, close other apps.\n"
        f"Technical details:\n{detail}")


@dataclass(frozen=True)
class StageRun:
    """One processing run: its temporary work directory, artifact directory and worker settings."""
    work: Path
    output_dir: Path
    status: Callable[[str], None]
    token: str | None
    offline: bool

    def execute(self, stage: str, source: Path, target: Path) -> None:
        self.status(STEPS[stage])
        worker(stage, source, target, self.status, self.token, self.offline)

    def read_json(self, name: str):
        return json.loads((self.work / name).read_text(encoding="utf-8"))


def process(source: Path, output_dir: Path, status: Callable[[str], None] = print, token: str | None = None,
            offline: bool = False, alignment: bool = False, raw_only: bool = False) -> dict:
    source = audio_source(source)
    output_dir = claim_output_dir(output_dir)
    with tempfile.TemporaryDirectory(prefix="interview-local-") as directory:  # pragma: no mutate - the prefix only names the temporary directory, which is removed afterwards
        run = StageRun(Path(directory), output_dir, status, token, offline)
        raw = recognize_speech(run, source)
        if raw_only:
            return raw
        run.execute("check-access", source, run.work / "access")  # pragma: no mutate - the check-access stage reads neither its source nor its target (worker.run)
        words = align_words(run) if alignment else whisper_words(run, raw)
        return attribute_speakers(run, raw, words, alignment)


def audio_source(source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ValueError("The audio file was not found.")
    if source.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError("Please select a WAV, MP3 or M4A file.")
    return source


def claim_output_dir(output_dir: Path) -> Path:
    """Create the output directory; refuse one that already holds completed stage files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in STAGE_FILENAMES):
        raise ValueError("Completed stage files already exist. Select a new output directory.")
    return output_dir


def recognize_speech(run: StageRun, source: Path) -> dict:
    """Decode and transcribe the audio and save the raw transcript."""
    from .artifacts import save_raw
    audio = run.work / DECODED_AUDIO
    run.execute("decode", source, audio)
    run.execute("asr", audio, run.work / "asr.json")
    raw = dict(schema_version=2, stage="raw", source=source.name, **run.read_json("asr.json"))
    save_raw(raw, run.output_dir)
    run.status(f"Original speech recognition output saved: {run.output_dir / '01_raw.txt'}")
    return raw


def align_words(run: StageRun) -> list[dict]:
    run.execute("align", run.work / DECODED_AUDIO, run.work / "words.json")
    return run.read_json("words.json")


def whisper_words(run: StageRun, raw: dict) -> list[dict]:
    """Words with Whisper's own timestamps; segments without words become one estimated word."""
    run.status("3/5 Use Whisper word timestamps (no additional alignment)")
    from .backends import fallback_words
    return [as_whisper_timing(w) for segment in raw["segments"] for w in fallback_words(segment)]


def as_whisper_timing(word: dict) -> dict:
    """Without alignment, Whisper timing is the primary timing, not a fallback."""
    return dict(word, timing="whisper") if word["timing"] == "whisper_fallback" else dict(word)


def attribute_speakers(run: StageRun, raw: dict, raw_words: list[dict], alignment: bool) -> dict:
    """Diarize, merge words and speaker turns, and save the speaker transcript."""
    from .artifacts import save_speakers
    run.execute("diarize", run.work / DECODED_AUDIO, run.work / "turns.json")
    words = [Word(**w) for w in raw_words]
    turns, regular = diarization_turns(run.read_json("turns.json"))
    run.status("5/5 Merge words and speaker turns, including overlap markers")
    result = merge(words, turns, regular)
    result["stage"] = "speakers"
    result["raw_transcript"] = raw
    result["metadata"] = speaker_metadata(words, result["speakers"], raw["source"], alignment)
    save_speakers(result, run.output_dir)
    return result


def diarization_turns(diarization: dict[str, list[dict]]) -> tuple[list[Turn], list[Turn]]:
    """Exclusive and regular turns of the diarize stage's output."""
    return as_turns(diarization["exclusive"]), as_turns(diarization["regular"])


def as_turns(items: list[dict]) -> list[Turn]:
    return [Turn(**t) for t in items]


def speaker_metadata(words: list[Word], speakers: dict, source_name: str, alignment: bool) -> dict:
    warnings = ["Automatic transcript: review text and speaker changes before use."]  # pragma: no mutate - warning wording shown to the user
    if len(speakers) != REQUESTED_SPEAKERS:
        warnings.append("The model did not detect exactly two speakers despite the requested count.")
    return dict(word_timing_counts=dict(Counter(w.timing for w in words)),
        source=source_name, transcription="faster-whisper/large-v3",
        diarization="pyannote/speaker-diarization-community-1", alignment=alignment,
        device="cpu", compute_type="int8 (ASR), float32 (diarization/alignment)",
        requested_speakers=REQUESTED_SPEAKERS, detected_speakers=len(speakers), warnings=warnings)

def prepare(status=print, token=None, alignment=False):
    stages = ["prepare-diarize", "prepare-asr"]
    if alignment:
        stages.append("prepare-align")
    for stage in stages:
        status(f"Model download / loading check: {stage}")
        worker(stage, Path("."), Path("."), status, token)
