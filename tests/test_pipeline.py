import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from interview_local.pipeline import process, worker
from run_fixtures import ASR_OUTPUT, STAGE_OUTPUTS, speech_models

INTERRUPT_SECONDS = 5
SPEECH_MODELS = speech_models()

class PipelineTests(unittest.TestCase):
    def test_orchestration_sequential_and_cleanup(self):
        calls = []
        temp = []
        def worker(stage, source, target, *args):
            calls.append(stage)
            temp.append(target.parent)
            if stage == "check-access":
                return
            if stage == "decode":
                target.touch()
            else:
                content = {"asr": {"segments": []},
                    "align": [{"start": 0, "end": 1, "text": "Hallo", "timing": "aligned", "probability": None}],
                    "diarize": [{"start": 0, "end": 1, "speaker": "s0"}]}[stage]
                target.write_text(json.dumps(content))
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/"test.m4a"
            source.touch()
            with patch("interview_local.pipeline.worker", worker):
                result = process(source, status=lambda _: None, alignment=True)
        self.assertEqual(calls, ["decode", "asr", "check-access", "align", "diarize"])
        self.assertEqual(result["segments"][0]["speaker"], "A")
        self.assertTrue(all(not p.exists() for p in temp))
        self.assertEqual(result["metadata"]["detected_speakers"], 1)

    def test_error_cleanup_and_invalid_input(self):
        with self.assertRaises(ValueError):
            process(Path("/missing.wav"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"broken.wav"
            p.touch()
            with patch("interview_local.pipeline.worker", side_effect=RuntimeError("Broken")):
                with self.assertRaises(RuntimeError):
                    process(p, status=lambda _: None)

class DecodeFailureTests(unittest.TestCase):
    def test_unreadable_audio_gets_its_own_message(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d)/"notes.wav"
            fake.write_text("This is not audio.")
            with self.assertRaises(RuntimeError) as raised:
                worker("decode", fake, Path(d)/"decoded.wav", status=lambda _: None)
        message = str(raised.exception)
        names_audio_cause_without_login_advice = ("could not be read as audio" in message, "401/403" in message)
        self.assertEqual(names_audio_cause_without_login_advice, (True, False))

def scripted_stage(script):
    """Replace the model worker command with a Python script; the environment is passed on unchanged."""
    real_popen = subprocess.Popen

    def popen(command, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)
    return patch("interview_local.pipeline.subprocess.Popen", popen)


def speech_models_transcribing(transcript):
    """Stand-in for the model worker processes whose speech recognition returns the given transcript."""
    outputs = dict(STAGE_OUTPUTS, asr=json.dumps(transcript))

    def worker(stage, source, target, *args):
        if stage in outputs:
            target.write_text(outputs[stage])
    return worker


def interrupting_status(line):
    raise KeyboardInterrupt


class WorkerTests(unittest.TestCase):
    def failure_of(self, script, stage="asr", **options):
        with scripted_stage(script), self.assertRaises(RuntimeError) as raised:
            worker(stage, Path("in"), Path("out"), status=lambda _: None, **options)
        return str(raised.exception)

    def test_progress_lines_are_reported_and_other_output_is_not(self):
        reported = []
        with scripted_stage("print('Loading model'); print('noise'); print('Transcribed through 5 seconds')"):
            worker("asr", Path("in"), Path("out"), status=reported.append)
        self.assertEqual(reported, ["Loading model", "Transcribed through 5 seconds"])

    def test_token_is_redacted_from_failure_details(self):
        message = self.failure_of("import os; print('token', os.environ['HF_TOKEN']); raise SystemExit(1)",
                                  token="hf_secret")
        self.assertEqual(("hf_secret" in message, "token [Token redacted]" in message), (False, True))

    def test_offline_run_keeps_the_child_offline(self):
        message = self.failure_of("import os; print('offline', os.environ['HF_HUB_OFFLINE'], "
                                  "os.environ['TRANSFORMERS_OFFLINE']); raise SystemExit(1)", offline=True)
        self.assertIn("offline 1 1", message)

    def test_child_telemetry_is_disabled(self):
        message = self.failure_of("import os; print('telemetry', os.environ['HF_HUB_DISABLE_TELEMETRY'], "
                                  "os.environ['PYANNOTE_METRICS_ENABLED']); raise SystemExit(1)")
        self.assertIn("telemetry 1 0", message)

    def test_failure_details_keep_the_last_eighteen_lines(self):
        message = self.failure_of("[print(f'line {i}') for i in range(30)]; raise SystemExit(1)")
        self.assertEqual(("line 11" in message, "line 12" in message, "line 29" in message), (False, True, True))

    def test_failed_model_stage_gives_login_advice(self):
        self.assertIn("401/403", self.failure_of("raise SystemExit(1)", stage="diarize"))

    def test_successful_stage_returns_quietly(self):
        with scripted_stage("print('done')"):
            self.assertIsNone(worker("asr", Path("in"), Path("out"), status=lambda _: None))

    def test_interrupt_stops_the_child_and_propagates(self):
        started = time.monotonic()
        with (scripted_stage("import time; print('Loading', flush=True); time.sleep(60)"),
              self.assertRaises(KeyboardInterrupt)):
            worker("asr", Path("in"), Path("out"), status=interrupting_status)
        self.assertLess(time.monotonic() - started, INTERRUPT_SECONDS)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.audio = self.root / "interview.wav"
        self.audio.touch()
        self.output = self.root / "output"
        self.messages = []

    def tearDown(self):
        self.directory.cleanup()

    def run_process(self, stages=SPEECH_MODELS, **options):
        with patch("interview_local.pipeline.worker", stages):
            return process(self.audio, status=self.messages.append, output_dir=self.output, **options)

    def test_unsupported_format_is_rejected(self):
        flac = self.root / "interview.flac"
        flac.touch()
        with self.assertRaises(ValueError):
            process(flac)

    def test_existing_stage_files_are_never_overwritten(self):
        self.output.mkdir()
        (self.output / "02_speakers.txt").touch()
        with self.assertRaises(ValueError):
            self.run_process()

    def test_raw_only_stops_after_speech_recognition(self):
        raw = self.run_process(raw_only=True)
        saved = sorted(p.name for p in self.output.iterdir())
        self.assertEqual((raw["stage"], raw["source"], saved), ("raw", "interview.wav", ["01_raw.json", "01_raw.txt"]))

    def test_speaker_transcript_is_saved_beside_the_raw_transcript(self):
        self.run_process()
        self.assertEqual(sorted(p.name for p in self.output.iterdir()),
                         ["01_raw.json", "01_raw.txt", "02_speakers.json", "02_speakers.txt"])

    def test_without_alignment_whisper_words_keep_their_timing(self):
        result = self.run_process()
        self.assertEqual(([w["timing"] for w in result["words"]], result["metadata"]["word_timing_counts"]),
                         (["whisper", "whisper"], {"whisper": 2}))

    def test_without_alignment_the_whisper_timestamps_are_announced(self):
        self.run_process()
        self.assertIn("3/5 Use Whisper word timestamps (no additional alignment)", self.messages)

    def test_segment_without_words_gets_estimated_timing(self):
        transcript = {"segments": [{"start": 0, "end": 1, "text": "Guten Tag", "words": []}], "language": "de"}
        result = self.run_process(stages=speech_models_transcribing(transcript))
        self.assertEqual([(w["text"], w["timing"]) for w in result["words"]], [("Guten Tag", "estimated")])

    def test_regular_diarization_enables_overlap_detection(self):
        self.assertTrue(self.run_process()["overlap_detection_available"])

    def test_fewer_than_two_speakers_are_warned_about(self):
        warnings = self.run_process()["metadata"]["warnings"]
        self.assertIn("The model did not detect exactly two speakers despite the requested count.", warnings)

    def test_speaker_transcript_embeds_the_raw_transcript(self):
        result = self.run_process()
        self.assertEqual((result["stage"], result["raw_transcript"]["segments"]),
                         ("speakers", ASR_OUTPUT["segments"]))


if __name__ == "__main__":
    unittest.main()
