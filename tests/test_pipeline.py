import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from interview_local.pipeline import process, worker
from run_fixtures import ASR_OUTPUT, STAGE_OUTPUTS, speech_models, uncached_speech_models

INTERRUPT_SECONDS = 5
STOP_TIMEOUT_FOR_TESTS = .5
DECODED_SAMPLE_RATE = 16000
SPEECH_MODELS = speech_models()

ALIGNED_STAGE_OUTPUTS = {
    "asr": {"segments": []},
    "align": [{"start": 0, "end": 1, "text": "Hallo", "timing": "aligned", "probability": None}],
    "diarize": {"exclusive": [{"start": 0, "end": 1, "speaker": "s0"}], "regular": []}}


def aligning_speech_models(stages_run, work_dirs):
    """Stand-in for the model workers of an aligned run; records each started stage and its work directory."""
    def worker(stage, source, target, *args):
        stages_run.append(stage)
        work_dirs.append(target.parent)
        if stage == "decode":
            target.touch()
        elif stage in ALIGNED_STAGE_OUTPUTS:
            target.write_text(json.dumps(ALIGNED_STAGE_OUTPUTS[stage]))
    return worker


class PipelineTests(unittest.TestCase):
    def aligned_run(self):
        """Process a recording with alignment; return (result, stages run, work directories)."""
        stages_run, work_dirs = [], []
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "test.m4a"
            source.touch()
            with patch("interview_local.pipeline.worker", aligning_speech_models(stages_run, work_dirs)):
                result = process(source, Path(d) / "output", status=lambda _: None, alignment=True)
        return result, stages_run, work_dirs

    def test_model_stages_run_one_after_another(self):
        _, stages_run, _ = self.aligned_run()
        self.assertEqual(stages_run, ["decode", "asr", "check-access", "align", "diarize"])

    def test_aligned_words_are_attributed_to_speakers(self):
        result, _, _ = self.aligned_run()
        self.assertEqual(result["segments"][0]["speaker"], "A")

    def test_temporary_work_directory_is_removed(self):
        _, _, work_dirs = self.aligned_run()
        self.assertEqual([p.exists() for p in work_dirs], [False] * len(work_dirs))

    def test_missing_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            process(Path(d) / "missing.wav", Path(d) / "output")

    def test_failing_model_stage_fails_the_run(self):
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "broken.wav"
            audio.touch()
            with patch("interview_local.pipeline.worker", side_effect=RuntimeError("Broken")), \
                    self.assertRaises(RuntimeError):
                process(audio, Path(d) / "output", status=lambda _: None)

class DecodeFailureTests(unittest.TestCase):
    def test_unreadable_audio_gets_its_own_message(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d)/"notes.wav"
            fake.write_text("This is not audio.")
            with self.assertRaises(RuntimeError) as raised:
                worker("decode", fake, Path(d)/"decoded.wav", status=lambda _: None, token=None)
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
    def failure_of(self, script: str, stage: str = "asr", token: str | None = None, offline: bool = False) -> str:
        with scripted_stage(script), self.assertRaises(RuntimeError) as raised:
            worker(stage, Path("in"), Path("out"), status=lambda _: None, token=token, offline=offline)
        return str(raised.exception)

    def test_progress_lines_are_reported_and_other_output_is_not(self):
        reported = []
        with scripted_stage("print('Loading model'); print('noise'); print('Transcribed through 5 seconds')"):
            worker("asr", Path("in"), Path("out"), status=reported.append, token=None)
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
        last_eighteen_lines = "\n".join(f"line {i}" for i in range(12, 30))
        self.assertEqual(("line 11" in message, message.endswith(last_eighteen_lines)), (False, True))

    def test_failure_details_include_the_error_output(self):
        message = self.failure_of("import sys; print('model file corrupt', file=sys.stderr); raise SystemExit(1)")
        self.assertIn("model file corrupt", message)

    def test_undecodable_output_is_replaced_instead_of_failing_the_relay(self):
        message = self.failure_of("import sys; sys.stdout.buffer.write(b'bad \\xff byte\\n'); raise SystemExit(1)")
        self.assertIn("bad � byte", message)

    def test_stage_without_offline_flag_lets_the_child_download(self):
        with (patch.dict(os.environ), scripted_stage("import os; print('offline', os.environ.get('HF_HUB_OFFLINE')); "
                                                     "raise SystemExit(1)"),
              self.assertRaises(RuntimeError) as raised):
            os.environ.pop("HF_HUB_OFFLINE", None)
            worker("prepare-asr", Path("in"), Path("out"), lambda _: None, None)
        self.assertIn("offline None", str(raised.exception))

    def test_interrupt_kills_a_child_that_ignores_termination(self):
        started = time.monotonic()
        script = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  "print('Loading', flush=True); time.sleep(60)")
        with (patch("interview_local.pipeline.STOP_TIMEOUT_SECONDS", STOP_TIMEOUT_FOR_TESTS),
              scripted_stage(script), self.assertRaises(KeyboardInterrupt)):
            worker("asr", Path("in"), Path("out"), status=interrupting_status, token=None)
        self.assertLess(time.monotonic() - started, INTERRUPT_SECONDS)

    def test_decode_stage_converts_a_recording_into_the_working_wav(self):
        with tempfile.TemporaryDirectory() as d:
            recording, decoded = Path(d) / "interview.wav", Path(d) / "original.wav"
            sf.write(recording, np.zeros(DECODED_SAMPLE_RATE // 10, dtype="float32"), DECODED_SAMPLE_RATE)
            worker("decode", recording, decoded, status=lambda _: None, token=None)
            info = sf.info(decoded)
        self.assertEqual((info.samplerate, info.channels, info.frames), (DECODED_SAMPLE_RATE, 1, DECODED_SAMPLE_RATE // 10))

    def test_failed_model_stage_gives_login_advice(self):
        self.assertIn("401/403", self.failure_of("raise SystemExit(1)", stage="diarize"))

    def test_successful_stage_returns_quietly(self):
        with scripted_stage("print('done')"):
            self.assertIsNone(worker("asr", Path("in"), Path("out"), status=lambda _: None, token=None))

    def test_interrupt_stops_the_child_and_propagates(self):
        started = time.monotonic()
        with (scripted_stage("import time; print('Loading', flush=True); time.sleep(60)"),
              self.assertRaises(KeyboardInterrupt)):
            worker("asr", Path("in"), Path("out"), status=interrupting_status, token=None)
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
            return process(self.audio, self.output, status=self.messages.append, **options)

    def test_unsupported_format_is_rejected(self):
        flac = self.root / "interview.flac"
        flac.touch()
        with self.assertRaises(ValueError):
            process(flac, self.output)

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

    def test_raw_transcript_names_its_schema_stage_and_source(self):
        raw = self.run_process(raw_only=True)
        self.assertEqual({key: raw[key] for key in ("schema_version", "stage", "source", "language")},
                         {"schema_version": 2, "stage": "raw", "source": "interview.wav", "language": "de"})

    def test_speaker_transcript_records_its_provenance(self):
        metadata = dict(self.run_process()["metadata"])
        del metadata["warnings"]
        self.assertEqual(metadata, {"word_timing_counts": {"whisper": 2}, "source": "interview.wav",
                                        "transcription": "faster-whisper/large-v3",
                                        "diarization": "pyannote/speaker-diarization-community-1", "alignment": False,
                                        "device": "cpu", "compute_type": "int8 (ASR), float32 (diarization/alignment)",
                                        "requested_speakers": 2, "detected_speakers": 1})

    def test_missing_output_directory_is_created_with_its_parents(self):
        self.output = self.root / "transcripts" / "interview-01"
        self.run_process(raw_only=True)
        self.assertTrue((self.output / "01_raw.json").is_file())

    def test_processing_without_the_offline_option_may_download_models(self):
        result = self.run_process(stages=uncached_speech_models())
        self.assertEqual(result["stage"], "speakers")

    def test_offline_processing_uses_only_cached_models(self):
        with self.assertRaisesRegex(RuntimeError, "not in the local cache"):
            self.run_process(stages=uncached_speech_models(), offline=True)

    def test_speaker_transcript_embeds_the_raw_transcript(self):
        result = self.run_process()
        self.assertEqual((result["stage"], result["raw_transcript"]["segments"]),
                         ("speakers", ASR_OUTPUT["segments"]))


if __name__ == "__main__":
    unittest.main()
