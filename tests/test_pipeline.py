# German fixtures intentionally exercise non-ASCII speech and paths.
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support.fake_process import FakeProcess
from support.fake_worker import OUTPUTS, TURNS, WORDS, FakeWorker
from support.platform_encoding import windows_default_encoding
from support.platform_files import case_sensitive_file_names

from interview_local.models import DIARIZATION_ID, WHISPER_MODEL
from interview_local.pipeline import (
    FAILURE_DETAIL_LINES,
    STEPS,
    WORKSPACE_PREFIX,
    prepare,
    process,
    transcribe,
    worker,
)
from interview_local.worker import REDACTED

RAW = {"segments": [{"start": 0, "end": 1, "text": "Original", "words": []}]}
TERMINATION_GRACE_SECONDS = 10


def silent(_message):
    pass


def processed(audio, output_dir, alignment=False):
    """Process audio online without a token, reporting nothing."""
    return process(audio, silent, None, False, alignment, output_dir)


class ProcessTests(unittest.TestCase):
    def run_process(self, outputs, alignment=False, offline=False, status=silent):
        worker = FakeWorker(outputs)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "test.m4a"
            audio.touch()
            output = Path(directory) / "output"
            with patch("interview_local.pipeline.worker", worker):
                result = process(audio, status, None, offline, alignment, output)
        return worker, result

    def test_speech_stages_run_sequentially_in_separate_workers(self):
        worker, _ = self.run_process(OUTPUTS, alignment=True)
        self.assertEqual(
            worker.stages, ["decode", "asr", "check-access", "align", "diarize"]
        )

    def test_temporary_workspace_is_removed_after_processing(self):
        worker, _ = self.run_process(OUTPUTS, alignment=True)
        self.assertFalse(any(path.exists() for path in worker.workspaces))

    def test_temporary_workspace_is_named_for_the_app(self):
        worker, _ = self.run_process(OUTPUTS, alignment=True)
        self.assertTrue(worker.workspaces[0].name.startswith(WORKSPACE_PREFIX))

    def test_offline_processing_runs_every_worker_offline(self):
        worker, _ = self.run_process(OUTPUTS, alignment=True, offline=True)
        self.assertEqual(worker.offline, {True})

    def test_words_are_attributed_to_the_diarized_speaker(self):
        _, result = self.run_process(OUTPUTS, alignment=True)
        self.assertEqual([w["speaker"] for w in result["words"]], ["A"])

    def test_speaker_stage_embeds_the_raw_transcript_it_came_from(self):
        _, result = self.run_process(OUTPUTS, alignment=True)
        self.assertEqual(
            (result["stage"], result["raw_transcript"]),
            (
                "speakers",
                {
                    "schema_version": 2,
                    "stage": "raw",
                    "source": "test.m4a",
                    "segments": [],
                },
            ),
        )

    def test_metadata_describes_the_run(self):
        _, result = self.run_process(OUTPUTS, alignment=True)
        self.assertEqual(
            result["metadata"],
            {
                "word_timing_counts": {"aligned": 1},
                "source": "test.m4a",
                "transcription": f"faster-whisper/{WHISPER_MODEL}",
                "diarization": DIARIZATION_ID,
                "alignment": True,
                "device": "cpu",
                "compute_type": "int8 (ASR), float32 (diarization/alignment)",
                "requested_speakers": 2,
                "detected_speakers": 1,
                "warnings": [
                    "Automatic transcript: review text and speaker changes before use.",
                    (
                        "The model did not detect exactly two speakers despite the "
                        "requested count."
                    ),
                ],
            },
        )

    def test_two_detected_speakers_add_no_speaker_warning(self):
        two_speakers = [
            {"start": 0, "end": 0.5, "speaker": "s0"},
            {"start": 0.5, "end": 1, "speaker": "s1"},
        ]
        diarization = {"exclusive": two_speakers, "regular": two_speakers}
        _, result = self.run_process(dict(OUTPUTS, diarize=diarization))
        self.assertEqual(
            result["metadata"]["warnings"],
            ["Automatic transcript: review text and speaker changes before use."],
        )

    def test_progress_reports_every_stage_and_where_the_raw_stage_was_saved(self):
        messages = []
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "test.wav"
            audio.touch()
            output = Path(directory) / "output"
            with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
                process(audio, messages.append, None, False, True, output)
            saved_raw = output / "01_raw.txt"
        steps = [m for m in messages if m in STEPS.values() or str(saved_raw) in m]
        self.assertEqual(
            steps,
            [
                STEPS["decode"],
                STEPS["asr"],
                f"Original speech recognition output saved: {saved_raw}",
                STEPS["align"],
                STEPS["diarize"],
            ],
        )

    def test_fewer_than_two_detected_speakers_adds_a_warning(self):
        _, result = self.run_process(OUTPUTS, alignment=True)
        self.assertEqual(
            result["metadata"]["warnings"],
            [
                "Automatic transcript: review text and speaker changes before use.",
                (
                    "The model did not detect exactly two speakers despite the "
                    "requested count."
                ),
            ],
        )

    def test_without_alignment_whisper_word_timestamps_are_used(self):
        words = [
            {"start": 0, "end": 0.4, "text": "Hallo", "timing": "whisper"},
            {"start": 0.4, "end": 1, "text": "Welt", "timing": "whisper"},
        ]
        segment = {"start": 0, "end": 1, "text": " Hallo Welt", "words": words}
        outputs = dict(OUTPUTS, asr={"segments": [segment]})
        _, result = self.run_process(outputs)
        timings = [w["timing"] for w in result["words"]]
        self.assertEqual(timings, ["whisper", "whisper"])

    def test_segment_without_word_timestamps_becomes_one_estimated_word(self):
        segment = {"start": 0, "end": 1, "text": " Hallo Welt", "words": []}
        outputs = dict(OUTPUTS, asr={"segments": [segment]})
        _, result = self.run_process(outputs)
        words = [(w["text"], w["timing"]) for w in result["words"]]
        self.assertEqual(words, [("Hallo Welt", "estimated")])

    def test_legacy_exclusive_turns_disable_overlap_detection(self):
        outputs = dict(OUTPUTS, diarize=TURNS)
        _, result = self.run_process(outputs, alignment=True)
        self.assertFalse(result["overlap_detection_available"])

    def test_regular_turns_enable_overlap_detection(self):
        _, result = self.run_process(OUTPUTS, alignment=True)
        self.assertTrue(result["overlap_detection_available"])

    def test_speaker_stage_is_saved_to_the_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "test.wav"
            audio.touch()
            output = Path(directory) / "output"
            with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
                processed(audio, output, alignment=True)
            saved = sorted(os.listdir(output))
        self.assertEqual(
            saved, ["01_raw.json", "01_raw.txt", "02_speakers.json", "02_speakers.txt"]
        )

    def test_stage_results_are_found_on_a_case_sensitive_file_system(self):
        with case_sensitive_file_names():
            _, result = self.run_process(OUTPUTS, alignment=True)
        self.assertEqual([w["speaker"] for w in result["words"]], ["A"])

    def test_german_worker_results_are_read_where_the_platform_default_is_not(self):
        aligned = [dict(WORDS[0], text="Grüße")]
        with windows_default_encoding():
            _, result = self.run_process(dict(OUTPUTS, align=aligned), alignment=True)
        self.assertEqual([w["text"] for w in result["words"]], ["Grüße"])

    def test_missing_parent_folders_of_the_output_directory_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "test.wav"
            audio.touch()
            output = Path(directory) / "interviews" / "first"
            with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
                processed(audio, output)
            saved = sorted(os.listdir(output))
        self.assertEqual(
            saved, ["01_raw.json", "01_raw.txt", "02_speakers.json", "02_speakers.txt"]
        )


class TranscribeTests(unittest.TestCase):
    def transcribed(self, offline=False):
        worker = FakeWorker(OUTPUTS)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "test.wav"
            audio.touch()
            with patch("interview_local.pipeline.worker", worker):
                raw = transcribe(audio, silent, offline, Path(directory) / "output")
        return worker, raw

    def test_raw_transcription_stops_after_speech_recognition(self):
        worker, _ = self.transcribed()
        self.assertEqual(worker.stages, ["decode", "asr"])

    def test_raw_transcription_returns_the_raw_stage(self):
        _, raw = self.transcribed()
        self.assertEqual(
            raw,
            {"schema_version": 2, "stage": "raw", "source": "test.wav", "segments": []},
        )

    def test_offline_transcription_runs_every_worker_offline(self):
        worker, _ = self.transcribed(offline=True)
        self.assertEqual(worker.offline, {True})

    def test_transcription_workspace_is_named_for_the_app(self):
        worker, _ = self.transcribed()
        self.assertTrue(worker.workspaces[0].name.startswith(WORKSPACE_PREFIX))


class PrepareTests(unittest.TestCase):
    def prepared(self, alignment, token=None):
        worker = FakeWorker({})
        with patch("interview_local.pipeline.worker", worker):
            prepare(silent, token, alignment)
        return worker

    def test_preparation_loads_diarization_and_speech_models(self):
        self.assertEqual(
            self.prepared(alignment=False).stages, ["prepare-diarize", "prepare-asr"]
        )

    def test_preparation_with_alignment_also_loads_the_alignment_model(self):
        self.assertEqual(
            self.prepared(alignment=True).stages,
            ["prepare-diarize", "prepare-asr", "prepare-align"],
        )

    def test_preparation_hands_the_token_to_every_worker(self):
        self.assertEqual(self.prepared(False, token="hf_token").tokens, {"hf_token"})

    def test_preparation_always_runs_online(self):
        self.assertEqual(self.prepared(alignment=True).offline, {False})


class InputTests(unittest.TestCase):
    def test_missing_audio_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.wav"
            with self.assertRaises(ValueError):
                processed(missing, Path(directory) / "output")

    def test_unsupported_audio_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.ogg"
            audio.touch()
            with self.assertRaises(ValueError):
                processed(audio, Path(directory) / "output")

    def test_output_directory_with_completed_stages_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.touch()
            (Path(directory) / "01_raw.json").touch()
            with self.assertRaises(ValueError):
                processed(audio, Path(directory))

    def test_worker_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "broken.wav"
            audio.touch()
            with (
                patch(
                    "interview_local.pipeline.worker",
                    side_effect=RuntimeError("Broken"),
                ),
                self.assertRaises(RuntimeError),
            ):
                processed(audio, Path(directory) / "output")


class PreservationTests(unittest.TestCase):
    def fail_after_raw_stage(self, output):
        def worker(stage, source, target, *args):
            if stage == "decode":
                target.touch()
            elif stage == "asr":
                target.write_text(json.dumps(RAW))
            elif stage == "check-access":
                raise RuntimeError("Access denied")

        audio = output.parent / "audio.wav"
        audio.touch()
        with (
            patch("interview_local.pipeline.worker", worker),
            self.assertRaises(RuntimeError),
        ):
            processed(audio, output)

    def test_raw_stage_files_survive_a_later_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            self.fail_after_raw_stage(output)
            saved = sorted(os.listdir(output))
        self.assertEqual(saved, ["01_raw.json", "01_raw.txt"])

    def test_preserved_raw_json_holds_the_original_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            self.fail_after_raw_stage(output)
            raw = json.loads((output / "01_raw.json").read_text())
        self.assertEqual(raw["segments"], RAW["segments"])


class WorkerLaunchTests(unittest.TestCase):
    """The child process a stage runs in, with its output scripted."""

    def launched(self, child, token=None, offline=False, status=silent):
        with (
            patch.dict(os.environ, clear=True),
            patch("subprocess.Popen", child),
        ):
            worker("decode", Path("in.wav"), Path("out.wav"), status, token, offline)
        return child

    def failure(self, output):
        with self.assertRaises(RuntimeError) as failure:
            self.launched(FakeProcess(output, exit_code=1))
        return str(failure.exception)

    def test_stage_runs_as_the_worker_module_with_its_paths(self):
        child = self.launched(FakeProcess())
        self.assertEqual(
            child.commands,
            [
                [
                    sys.executable,
                    "-m",
                    "interview_local.worker",
                    "decode",
                    "in.wav",
                    "out.wav",
                ]
            ],
        )

    def test_offline_worker_is_told_not_to_download(self):
        child = self.launched(FakeProcess(), offline=True)
        self.assertEqual(child.environments[0].get("HF_HUB_OFFLINE"), "1")

    def test_online_worker_may_download(self):
        child = self.launched(FakeProcess(), offline=False)
        self.assertNotIn("HF_HUB_OFFLINE", child.environments[0])

    def test_token_is_handed_to_the_worker(self):
        child = self.launched(FakeProcess(), token="hf_entered")
        self.assertEqual(child.environments[0].get("HF_TOKEN"), "hf_entered")

    def test_token_in_relayed_progress_is_redacted(self):
        messages = []
        output = b"Loading model with plain-secret\n"
        self.launched(FakeProcess(output), token="plain-secret", status=messages.append)
        self.assertEqual(messages, [f"Loading model with {REDACTED}"])

    def test_failure_detail_ends_with_the_last_output_lines(self):
        lines = [f"line {i}" for i in range(FAILURE_DETAIL_LINES + 2)]
        message = self.failure("\n".join(lines).encode())
        expected_tail = "\n".join(lines[-FAILURE_DETAIL_LINES:])
        self.assertTrue(message.endswith(f"Technical details:\n{expected_tail}"))

    def test_undecodable_output_is_relayed_with_replacement_characters(self):
        messages = []
        self.launched(FakeProcess(b"Loading \xff model\n"), status=messages.append)
        self.assertEqual(messages, ["Loading � model"])

    def test_output_is_read_as_utf8_where_the_platform_default_is_not(self):
        messages = []
        with patch("subprocess._text_encoding", return_value="cp1252"):
            output = "Loading Grüße\n".encode()
            self.launched(FakeProcess(output), status=messages.append)
        self.assertEqual(messages, ["Loading Grüße"])

    def test_cancelled_worker_ignoring_termination_is_killed_after_a_grace_period(
        self,
    ):
        def cancel(_message):
            raise KeyboardInterrupt

        child = FakeProcess(b"Loading\n", ignores_termination=True)
        with self.assertRaises(KeyboardInterrupt):
            self.launched(child, status=cancel)
        self.assertEqual(
            (child.terminated, child.waits[0], child.killed),
            (True, TERMINATION_GRACE_SECONDS, True),
        )


if __name__ == "__main__":
    unittest.main()
