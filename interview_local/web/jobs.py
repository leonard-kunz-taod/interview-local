"""Background transcription of one uploaded recording into its run directory."""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import traceback
import uuid

from ..artifacts import save_readable, save_speakers
from ..llm import OllamaClient, enhance
from ..pipeline import STEPS, process

PHASE_SPEECH_RECOGNITION = "speech_recognition"
PHASE_SPEAKER_RECOGNITION = "speaker_recognition"
PHASE_LLM = "llm"


@dataclass(frozen=True)
class JobStatus:
    """A snapshot of a job's progress for the browser."""
    job_id: str
    run_id: str
    state: str
    phase: str
    detail: str
    error: str | None


class TranscriptionJob:
    """Runs speech recognition, speaker recognition and the LLM draft in a daemon thread."""

    def __init__(self, run_dir: Path, audio: Path, speaker_names: dict[str, str]):
        self.job_id = uuid.uuid4().hex
        self.run_dir = run_dir
        self.audio = audio
        self.speaker_names = dict(speaker_names)
        self._lock = threading.Lock()
        self._state = "running"
        self._phase = PHASE_SPEECH_RECOGNITION
        self._detail = "Starting"
        self._error: str | None = None

    def start(self) -> None:
        """Start processing in a daemon thread."""
        threading.Thread(target=self._run, name=f"transcription-{self.job_id}", daemon=True).start()

    def status(self) -> JobStatus:
        """Return the current progress snapshot."""
        with self._lock:
            return JobStatus(job_id=self.job_id, run_id=self.run_dir.name, state=self._state,
                             phase=self._phase, detail=self._detail, error=self._error)

    @property
    def running(self) -> bool:
        """Whether the job has neither finished nor failed."""
        with self._lock:
            return self._state == "running"

    def _run(self) -> None:
        try:
            self._transcribe()
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                self._state = "failed"
                self._error = str(exc)
            return
        with self._lock:
            self._state = "done"
            self._detail = "Done"

    def _transcribe(self) -> None:
        client = OllamaClient()
        client.ready()
        token = diarization_token()
        result = process(self.audio, status=self._record_status, token=token, output_dir=self.run_dir)
        result["speakers"].update(self.speaker_names)
        save_speakers(result, self.run_dir)
        self._enter_phase(PHASE_LLM)
        save_readable(enhance(result, client=client, status=self._record_status), self.run_dir)

    def _record_status(self, message: str) -> None:
        if message == STEPS["check-access"]:
            self._enter_phase(PHASE_SPEAKER_RECOGNITION)
        with self._lock:
            self._detail = message

    def _enter_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase


def diarization_token() -> str | None:
    """Return the HF token for speaker recognition without prompting.

    Returns:
        ``None`` when the local community-1 folder is used, else the token from
        HF_TOKEN or the cached Hugging Face login.

    Raises:
        RuntimeError: If neither the local model nor any HF credential exists.
    """
    from ..models import local_diarization_model
    if local_diarization_model():
        return None
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    from huggingface_hub import get_token
    cached = get_token()
    if cached:
        return cached
    raise RuntimeError("Speaker recognition needs the local community-1 folder (models/community-1), "
                       "the HF_TOKEN environment variable or a cached Hugging Face login "
                       "(.venv/bin/hf auth login). The web UI does not ask for tokens.")
