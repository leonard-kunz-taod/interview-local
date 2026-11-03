import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from interview_local.web.jobs import TranscriptionJob
from run_fixtures import ScriptedClient

JOB_TIMEOUT_SECONDS = 10
ASR_OUTPUT = {"segments": [dict(start=0, end=1, text="Guten Tag", words=[
    dict(start=0, end=.5, text="Guten", timing="whisper", probability=.9),
    dict(start=.5, end=1, text="Tag", timing="whisper", probability=.4)])], "language": "de"}
DIARIZATION_OUTPUT = {"exclusive": [dict(start=0, end=1, speaker="s0")],
                      "regular": [dict(start=0, end=1, speaker="s0")]}


def speech_models(failing_stage=None):
    """Stand-in for the model worker processes: writes the outputs the real stages produce."""
    def worker(stage, source, target, *args):
        if stage == failing_stage:
            raise RuntimeError(f"Stage '{stage}' failed: model unavailable")
        if stage == "decode":
            target.touch()
        elif stage == "asr":
            target.write_text(json.dumps(ASR_OUTPUT))
        elif stage == "diarize":
            target.write_text(json.dumps(DIARIZATION_OUTPUT))
    return worker


def local_llm():
    return ScriptedClient([(0, 1, "A", "Guten Tag.")])


def finished(job):
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while job.running and time.monotonic() < deadline:
        time.sleep(.02)
    return job.status()


class TranscriptionJobTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.directory.name) / "run"
        self.run_dir.mkdir()
        self.audio = self.run_dir / "source.wav"
        self.audio.touch()

    def tearDown(self):
        self.directory.cleanup()

    def run_job(self, failing_stage=None):
        with (patch("interview_local.pipeline.worker", speech_models(failing_stage)),
              patch("interview_local.web.jobs.OllamaClient", local_llm),
              patch.dict(os.environ, {"HF_TOKEN": "hf_test"})):
            job = TranscriptionJob(self.run_dir, self.audio, {"A": "Anna", "B": "Bernd"})
            job.start()
            return finished(job)

    def test_successful_job_writes_all_stages_with_names(self):
        status = self.run_job()
        self.assertEqual((status.state, status.phase, status.error), ("done", "llm", None))
        readable = json.loads((self.run_dir / "03_readable.json").read_text(encoding="utf-8"))
        self.assertEqual(readable["speakers"]["A"], "Anna")
        self.assertEqual([w["probability"] for w in readable["source_transcript"]["words"]], [.9, .4])

    def test_speech_recognition_failure_is_reported_in_its_phase(self):
        status = self.run_job(failing_stage="asr")
        self.assertEqual((status.state, status.phase), ("failed", "speech_recognition"))
        self.assertIn("model unavailable", status.error)

    def test_speaker_recognition_failure_is_reported_in_its_phase(self):
        status = self.run_job(failing_stage="diarize")
        self.assertEqual((status.state, status.phase), ("failed", "speaker_recognition"))
        self.assertIn("model unavailable", status.error)


if __name__ == "__main__":
    unittest.main()
