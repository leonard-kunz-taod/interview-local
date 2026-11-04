import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from run_fixtures import gated_speech_models, scripted_ollama, speech_models
from support.fake_ollama import FakeOllama, served_by

from interview_local.pipeline import STEPS
from interview_local.web.jobs import TranscriptionJob, diarization_token

JOB_TIMEOUT_SECONDS = 10
HF_TOKEN = "hf_test"


def local_llm():
    return scripted_ollama([(0, 1, "A", "Guten Tag.")])


def failing_llm():
    """A local LLM that is reachable but fails while drafting."""
    return FakeOllama(failures={"/api/chat": b"model unavailable"})


def finished(job):
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    while job.running and time.monotonic() < deadline:
        time.sleep(0.02)
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

    def run_job(self, failing_stage=None, llm=local_llm, models=None):
        with (
            patch(
                "interview_local.pipeline.worker",
                models or speech_models(failing_stage),
            ),
            served_by(llm()),
            patch("interview_local.models.local_diarization_model", lambda: None),
            patch.dict(os.environ, {"HF_TOKEN": HF_TOKEN}),
        ):
            job = TranscriptionJob(
                self.run_dir, self.audio, {"A": "Anna", "B": "Bernd"}
            )
            job.start()
            return finished(job)

    def saved_readable(self):
        return json.loads(
            (self.run_dir / "03_readable.json").read_text(encoding="utf-8")
        )

    def test_successful_job_finishes_in_the_llm_phase(self):
        status = self.run_job()
        self.assertEqual(
            (status.state, status.phase, status.error), ("done", "llm", None)
        )

    def test_successful_job_saves_the_entered_speaker_names(self):
        self.run_job()
        self.assertEqual(self.saved_readable()["speakers"], {"A": "Anna", "B": "Bernd"})

    def test_successful_job_keeps_the_whisper_word_probabilities(self):
        self.run_job()
        self.assertEqual(
            [
                w["probability"]
                for w in self.saved_readable()["source_transcript"]["words"]
            ],
            [0.9, 0.4],
        )

    def test_speech_recognition_failure_is_reported_in_its_phase(self):
        status = self.run_job(failing_stage="asr")
        self.assertEqual(
            (status.state, status.phase, "model unavailable" in status.error),
            ("failed", "speech_recognition", True),
        )

    def test_speaker_recognition_failure_is_reported_in_its_phase(self):
        status = self.run_job(failing_stage="diarize")
        self.assertEqual(
            (status.state, status.phase, "model unavailable" in status.error),
            ("failed", "speaker_recognition", True),
        )

    def test_llm_failure_is_reported_in_its_phase(self):
        status = self.run_job(llm=failing_llm)
        self.assertEqual(
            (status.state, status.phase, "model unavailable" in status.error),
            ("failed", "llm", True),
        )

    def test_failed_job_shows_the_step_it_failed_in(self):
        status = self.run_job(failing_stage="diarize")
        self.assertEqual(status.detail, STEPS["diarize"])

    def test_failed_llm_draft_shows_the_llm_progress(self):
        status = self.run_job(llm=failing_llm)
        self.assertTrue(status.detail.startswith("LLM readability"), status.detail)

    def test_speaker_recognition_gets_the_hf_token_of_the_environment(self):
        status = self.run_job(models=gated_speech_models(HF_TOKEN))
        self.assertEqual((status.state, status.error), ("done", None))

    def test_status_names_the_run(self):
        job = TranscriptionJob(self.run_dir, self.audio, {"A": "Anna", "B": "Bernd"})
        self.assertEqual(job.status().run_id, self.run_dir.name)


class DiarizationTokenTests(unittest.TestCase):
    def token_with(self, local_model, environment, cached_login):
        with (
            patch(
                "interview_local.models.local_diarization_model", lambda: local_model
            ),
            patch.dict(os.environ, environment, clear=True),
            patch("huggingface_hub.get_token", lambda: cached_login),
        ):
            return diarization_token()

    def test_local_model_needs_no_token(self):
        self.assertIsNone(
            self.token_with(
                Path("models/community-1"), {"HF_TOKEN": "hf_env"}, "hf_cached"
            )
        )

    def test_environment_token_is_preferred_over_the_cached_login(self):
        self.assertEqual(
            self.token_with(None, {"HF_TOKEN": "hf_env"}, "hf_cached"), "hf_env"
        )

    def test_cached_login_is_used_without_environment_token(self):
        self.assertEqual(self.token_with(None, {}, "hf_cached"), "hf_cached")

    def test_missing_credentials_fail_instead_of_prompting(self):
        with self.assertRaises(RuntimeError):
            self.token_with(None, {}, None)


if __name__ == "__main__":
    unittest.main()
