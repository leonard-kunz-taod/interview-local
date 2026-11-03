import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from test_models import without_local_model

from interview_local.core import DiarizationResult, Turn
from interview_local.pipeline import worker
from interview_local.worker import main, run

TOKEN = "hf_secretToken123"
TRANSCRIPT = {"segments": [{"start": 0, "end": 1, "text": "Hallo"}], "language": "de"}
ALIGNED = [{"start": 0, "end": 1, "text": "Hallo", "timing": "aligned"}]


def stereo_recording(path):
    sf.write(path, np.zeros((8000, 2)), 8000)


def silent(_message):
    pass


class FakeTranscriber:
    loaded = 0

    def __init__(self):
        FakeTranscriber.loaded += 1

    def transcribe(self, audio):
        return TRANSCRIPT


class FakeDiarizer:
    def diarize(self, audio):
        turns = [Turn(0, 1, "s")]
        return DiarizationResult(turns, turns)


class FakeAligner:
    def align(self, audio, transcript):
        return [dict(ALIGNED[0], text=transcript["segments"][0]["text"])]


class WorkerProcessTests(unittest.TestCase):
    def test_decoded_audio_is_16khz_mono_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            target = Path(directory) / "decoded.wav"
            stereo_recording(source)
            worker("decode", source, target, status=lambda _: None)
            info = sf.info(target)
        self.assertEqual(
            (info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16")
        )

    def test_completed_stage_reports_its_duration(self):
        messages = []
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            stereo_recording(source)
            worker("decode", source, Path(directory) / "out.wav", messages.append)
        self.assertTrue(
            any(m.startswith("Stage decode completed in") for m in messages)
        )

    def test_unknown_stage_fails_with_the_worker_detail(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "Unknown processing stage"),
        ):
            worker("unknown", Path(directory), Path(directory), lambda _: None)

    def test_token_is_redacted_from_failure_details(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / f"{TOKEN}.wav"
            with self.assertRaises(RuntimeError) as failure:
                worker("decode", missing, missing, lambda _: None, TOKEN)
        self.assertNotIn(TOKEN, str(failure.exception))

    def test_cancelling_during_a_stage_stops_the_worker(self):
        def cancel(_message):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            stereo_recording(source)
            with self.assertRaises(KeyboardInterrupt):
                worker("decode", source, Path(directory) / "out.wav", cancel)


class StageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.source = self.directory / "original.wav"
        self.target = self.directory / "result.json"

    def saved_result(self):
        return json.loads(self.target.read_text(encoding="utf-8"))

    def test_unknown_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown processing stage"):
            run("unknown", self.source, self.target)

    def test_decode_writes_16khz_mono_pcm(self):
        stereo_recording(self.source)
        decoded = self.directory / "decoded.wav"
        run("decode", self.source, decoded)
        info = sf.info(decoded)
        self.assertEqual(
            (info.samplerate, info.channels, info.frames), (16000, 1, 16000)
        )

    def test_recording_without_samples_is_rejected(self):
        sf.write(self.source, np.zeros((0, 1)), 8000)
        with self.assertRaisesRegex(ValueError, "no audio track"):
            run("decode", self.source, self.directory / "decoded.wav")

    def test_speech_recognition_saves_the_transcript(self):
        with patch(
            "interview_local.backends.FasterWhisperTranscriber", FakeTranscriber
        ):
            run("asr", self.source, self.target)
        self.assertEqual(self.saved_result(), TRANSCRIPT)

    def test_diarization_saves_exclusive_and_regular_turns(self):
        with patch("interview_local.backends.CommunityDiarizer", FakeDiarizer):
            run("diarize", self.source, self.target)
        turn = {"start": 0, "end": 1, "speaker": "s"}
        self.assertEqual(self.saved_result(), {"exclusive": [turn], "regular": [turn]})

    def test_alignment_aligns_the_transcript_next_to_the_audio(self):
        (self.directory / "asr.json").write_text(json.dumps(TRANSCRIPT))
        with patch("interview_local.backends.GermanWhisperXAligner", FakeAligner):
            run("align", self.source, self.target)
        self.assertEqual(self.saved_result(), ALIGNED)

    def test_preparing_a_model_loads_it_without_saving_a_result(self):
        loaded = FakeTranscriber.loaded
        with patch(
            "interview_local.backends.FasterWhisperTranscriber", FakeTranscriber
        ):
            run("prepare-asr", self.source, self.target)
        self.assertEqual(
            (FakeTranscriber.loaded - loaded, self.target.exists()), (1, False)
        )

    def test_preparing_diarization_saves_no_result(self):
        with patch("interview_local.backends.CommunityDiarizer", FakeDiarizer):
            run("prepare-diarize", self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_preparing_alignment_fetches_language_data(self):
        with (
            patch("interview_local.backends.GermanWhisperXAligner", FakeAligner),
            patch("nltk.download", return_value=True),
        ):
            run("prepare-align", self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_missing_language_data_download_fails_alignment_preparation(self):
        with (
            patch("interview_local.backends.GermanWhisperXAligner", FakeAligner),
            patch("nltk.download", return_value=False),
            self.assertRaisesRegex(RuntimeError, "NLTK language data"),
        ):
            run("prepare-align", self.source, self.target)

    def test_access_check_with_a_local_model_contacts_no_hub(self):
        (self.directory / "config.yaml").touch()
        refuse = AssertionError("The Hugging Face Hub was contacted.")
        with (
            patch.dict(
                os.environ, {"INTERVIEW_DIARIZATION_MODEL": str(self.directory)}
            ),
            patch("huggingface_hub.hf_hub_download", side_effect=refuse),
        ):
            run("check-access", self.source, self.target)
        self.assertFalse(self.target.exists())

    def test_access_check_without_a_local_model_fails_when_the_hub_refuses(self):
        refusal = RuntimeError("401 Client Error")
        with (
            without_local_model(),
            patch("huggingface_hub.hf_hub_download", side_effect=refusal),
            self.assertRaisesRegex(RuntimeError, "401"),
        ):
            run("check-access", self.source, self.target)

    def test_access_check_without_a_local_model_passes_when_the_hub_grants(self):
        with (
            without_local_model(),
            patch("huggingface_hub.hf_hub_download", return_value="config.yaml"),
        ):
            run("check-access", self.source, self.target)
        self.assertFalse(self.target.exists())


class MainTests(unittest.TestCase):
    def run_main(self, *arguments):
        """Exit code and combined output of the worker entry point."""
        printed = io.StringIO()
        code = 0
        with (
            patch("sys.argv", ["worker", *map(str, arguments)]),
            contextlib.redirect_stdout(printed),
            contextlib.redirect_stderr(printed),
        ):
            try:
                main()
            except SystemExit as exit_:
                code = exit_.code
        return code, printed.getvalue()

    def test_completed_stage_reports_duration_and_peak_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            stereo_recording(source)
            code, output = self.run_main("decode", source, Path(directory) / "o.wav")
        lines = output.splitlines()
        self.assertEqual(
            (code, lines[0].startswith("Stage decode completed in"), lines[1][:23]),
            (0, True, "Peak process RAM (RSS):"),
        )

    def test_failed_stage_exits_with_the_error_type_and_message(self):
        with tempfile.TemporaryDirectory() as directory:
            code, output = self.run_main("unknown", directory, directory)
        self.assertEqual(
            (code, output), (1, "ERROR (ValueError): Unknown processing stage\n")
        )

    def test_environment_token_is_redacted_from_the_error(self):
        secret = "plain-secret-value"
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"HF_TOKEN": secret}),
        ):
            missing = Path(directory) / f"{secret}.wav"
            _, output = self.run_main("decode", missing, missing)
        self.assertEqual(
            (secret in output, "[Token redacted]" in output), (False, True)
        )

    def test_token_shaped_text_is_redacted_from_the_error(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"HF_TOKEN": ""}),
        ):
            missing = Path(directory) / f"{TOKEN}.wav"
            _, output = self.run_main("decode", missing, missing)
        self.assertEqual((TOKEN in output, "[Token redacted]" in output), (False, True))


if __name__ == "__main__":
    unittest.main()
