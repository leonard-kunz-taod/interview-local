# German fixtures intentionally exercise non-ASCII speech.
import contextlib
import io
import json
import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
from support.model_folders import without_local_model
from support.platform_encoding import windows_default_encoding
from support.platform_files import case_sensitive_file_names

from interview_local.core import DiarizationResult, Turn
from interview_local.models import DIARIZATION_ID
from interview_local.pipeline import worker
from interview_local.worker import SAMPLE_RATE, main, run

TOKEN = "hf_secretToken123"
GRANTED_TOKEN = "hf_granted"
TRANSCRIPT = {"segments": [{"start": 0, "end": 1, "text": "Hallo"}], "language": "de"}
ALIGNED = [{"start": 0, "end": 1, "text": "Hallo", "timing": "aligned"}]
PCM_16_FULL_SCALE = 32768


def stereo_recording(path):
    sf.write(path, np.zeros((8000, 2)), 8000)


def silent(_message):
    pass


class FakeTranscriber:
    """Whisper stand-in; like the real model it reads the audio it is given."""

    loaded = 0

    def __init__(self):
        FakeTranscriber.loaded += 1

    def transcribe(self, audio):
        Path(audio).read_bytes()
        return TRANSCRIPT


class FakeDiarizer:
    def diarize(self, audio):
        Path(audio).read_bytes()
        turns = [Turn(0, 1, "s")]
        return DiarizationResult(turns, turns)


class FakeAligner:
    def align(self, audio, transcript):
        Path(audio).read_bytes()
        return [dict(ALIGNED[0], text=transcript["segments"][0]["text"])]


class GatedHub:
    """The Hugging Face Hub serving the gated community-1 configuration."""

    def __init__(self):
        self.downloaded = []

    def __call__(self, repo_id, filename, *, token, local_files_only):
        if local_files_only:
            raise FileNotFoundError("Not in the local cache.")
        if (repo_id, filename) != (DIARIZATION_ID, "config.yaml"):
            raise FileNotFoundError(f"{repo_id} has no {filename}.")
        if token != GRANTED_TOKEN:
            raise PermissionError("401 Client Error: gated repository.")
        self.downloaded.append(filename)
        return filename


class NltkIndex:
    """nltk.download: True for a package it can fetch; it talks unless quiet."""

    def __init__(self, available=("punkt_tab",)):
        self.available = available

    def __call__(self, package, quiet=False):
        if not quiet:
            print(f"[nltk_data] Downloading package {package}")
        return package in self.available


class WorkerProcessTests(unittest.TestCase):
    def test_decoded_audio_is_16khz_mono_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            target = Path(directory) / "decoded.wav"
            stereo_recording(source)
            worker("decode", source, target, silent, None, False)
            info = sf.info(target)
        self.assertEqual(
            (info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16")
        )

    def test_completed_stage_reports_its_duration(self):
        messages = []
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            stereo_recording(source)
            target = Path(directory) / "out.wav"
            worker("decode", source, target, messages.append, None, False)
        self.assertTrue(
            any(m.startswith("Stage decode completed in") for m in messages)
        )

    def test_unknown_stage_fails_with_the_worker_detail(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "Unknown processing stage"),
        ):
            worker("unknown", Path(directory), Path(directory), silent, None, False)

    def test_token_is_redacted_from_failure_details(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / f"{TOKEN}.wav"
            with self.assertRaises(RuntimeError) as failure:
                worker("decode", missing, missing, silent, TOKEN, False)
        self.assertNotIn(TOKEN, str(failure.exception))

    def test_token_without_the_hub_prefix_is_redacted_from_failure_details(self):
        secret = "plain-secret-value"
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / f"{secret}.wav"
            with self.assertRaises(RuntimeError) as failure:
                worker("decode", missing, missing, silent, secret, False)
        self.assertNotIn(secret, str(failure.exception))

    def test_cancelling_during_a_stage_stops_the_worker(self):
        def cancel(_message):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            stereo_recording(source)
            with self.assertRaises(KeyboardInterrupt):
                target = Path(directory) / "out.wav"
                worker("decode", source, target, cancel, None, False)


class StageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.source = self.directory / "original.wav"
        self.source.touch()
        self.target = self.directory / "result.json"

    def saved_result(self):
        return json.loads(self.target.read_text(encoding="utf-8"))

    def save_transcript(self, transcript):
        text = json.dumps(transcript, ensure_ascii=False)
        (self.directory / "asr.json").write_text(text, encoding="utf-8")

    def check_access(self, environment, hub):
        with (
            without_local_model(),
            patch.dict(os.environ, environment),
            patch("huggingface_hub.hf_hub_download", hub),
        ):
            run("check-access", self.source, self.target)

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

    def test_decoding_keeps_the_waveform_of_a_floating_point_recording(self):
        seconds = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        tone = 0.6 * np.sin(2 * np.pi * 440 * seconds)
        sf.write(self.source, np.stack([tone, tone], axis=1), SAMPLE_RATE, "FLOAT")
        decoded = self.directory / "decoded.wav"
        run("decode", self.source, decoded)
        samples, _ = sf.read(decoded, dtype="int16")
        expected = np.round(tone * PCM_16_FULL_SCALE)
        self.assertLessEqual(np.abs(samples - expected).max(), 1)

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
        self.save_transcript(TRANSCRIPT)
        with patch("interview_local.backends.GermanWhisperXAligner", FakeAligner):
            run("align", self.source, self.target)
        self.assertEqual(self.saved_result(), ALIGNED)

    def test_alignment_finds_its_transcript_on_a_case_sensitive_file_system(self):
        self.save_transcript(TRANSCRIPT)
        with (
            case_sensitive_file_names(),
            patch("interview_local.backends.GermanWhisperXAligner", FakeAligner),
        ):
            run("align", self.source, self.target)
        self.assertEqual(self.saved_result(), ALIGNED)

    def test_german_alignment_is_kept_where_the_platform_default_is_not_utf8(self):
        german = {"segments": [{"start": 0, "end": 1, "text": "Grüße"}]}
        self.save_transcript(german)
        with (
            windows_default_encoding(),
            patch("interview_local.backends.GermanWhisperXAligner", FakeAligner),
        ):
            run("align", self.source, self.target)
        self.assertEqual(self.saved_result(), [dict(ALIGNED[0], text="Grüße")])

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

    def test_preparing_alignment_fetches_its_language_data_quietly(self):
        printed = io.StringIO()
        with (
            patch("interview_local.backends.GermanWhisperXAligner", FakeAligner),
            patch("nltk.download", NltkIndex()),
            contextlib.redirect_stdout(printed),
        ):
            run("prepare-align", self.source, self.target)
        self.assertEqual((printed.getvalue(), self.target.exists()), ("", False))

    def test_missing_language_data_download_fails_alignment_preparation(self):
        with (
            patch("interview_local.backends.GermanWhisperXAligner", FakeAligner),
            patch("nltk.download", NltkIndex(available=())),
            contextlib.redirect_stdout(io.StringIO()),
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

    def test_access_check_fetches_the_gated_configuration_with_the_token(self):
        hub = GatedHub()
        self.check_access({"HF_TOKEN": GRANTED_TOKEN, "HF_HUB_OFFLINE": "0"}, hub)
        self.assertEqual(hub.downloaded, ["config.yaml"])

    def test_access_check_fails_when_the_hub_refuses_the_token(self):
        with self.assertRaisesRegex(PermissionError, "401"):
            self.check_access({"HF_TOKEN": TOKEN, "HF_HUB_OFFLINE": "0"}, GatedHub())

    def test_offline_access_check_only_consults_the_local_cache(self):
        environment = {"HF_TOKEN": GRANTED_TOKEN, "HF_HUB_OFFLINE": "1"}
        with self.assertRaisesRegex(FileNotFoundError, "local cache"):
            self.check_access(environment, GatedHub())

    def test_access_check_without_a_token_uses_the_cached_login(self):
        with self.assertRaisesRegex(PermissionError, "401"):
            self.check_access({"HF_TOKEN": "", "HF_HUB_OFFLINE": "0"}, GatedHub())


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

    def peak_memory_line(self, system, max_rss):
        usage = SimpleNamespace(ru_maxrss=max_rss)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"INTERVIEW_DIARIZATION_MODEL": directory}),
            patch("platform.system", return_value=system),
            patch("resource.getrusage", return_value=usage),
        ):
            (Path(directory) / "config.yaml").touch()
            _, output = self.run_main("check-access", directory, directory)
        return output.splitlines()[-1]

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

    def test_reported_duration_does_not_exceed_the_stage_run(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            stereo_recording(source)
            started = time.monotonic()
            _, output = self.run_main("decode", source, Path(directory) / "o.wav")
            elapsed = time.monotonic() - started
        reported = float(re.search(r"completed in ([\d.]+) s", output).group(1))
        self.assertLessEqual(reported, round(elapsed, 1))

    def test_macos_peak_memory_is_converted_from_bytes(self):
        line = self.peak_memory_line("Darwin", max_rss=3 * 2**30)
        self.assertEqual(line, "Peak process RAM (RSS): 3.00 GiB")

    def test_linux_peak_memory_is_converted_from_kibibytes(self):
        line = self.peak_memory_line("Linux", max_rss=3 * 2**20)
        self.assertEqual(line, "Peak process RAM (RSS): 3.00 GiB")

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
