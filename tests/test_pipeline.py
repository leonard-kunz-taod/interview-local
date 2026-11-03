import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interview_local.pipeline import prepare, process, transcribe

WORDS = [{"start": 0, "end": 1, "text": "Hallo", "timing": "aligned"}]
TURNS = [{"start": 0, "end": 1, "speaker": "s0"}]
DIARIZATION = {"exclusive": TURNS, "regular": TURNS}
OUTPUTS = {"asr": {"segments": []}, "align": WORDS, "diarize": DIARIZATION}
RAW = {"segments": [{"start": 0, "end": 1, "text": "Original", "words": []}]}


class FakeWorker:
    """Writes canned stage results instead of running speech models."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.stages = []
        self.workspaces = []
        self.tokens = set()

    def __call__(self, stage, source, target, status=None, token=None, offline=False):
        self.stages.append(stage)
        self.workspaces.append(target.parent)
        self.tokens.add(token)
        if stage == "decode":
            target.touch()
        elif stage in self.outputs:
            target.write_text(json.dumps(self.outputs[stage]))


def silent(_message):
    pass


class ProcessTests(unittest.TestCase):
    def run_process(self, outputs, **options):
        worker = FakeWorker(outputs)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "test.m4a"
            audio.touch()
            with patch("interview_local.pipeline.worker", worker):
                result = process(audio, status=silent, **options)
        return worker, result

    def test_speech_stages_run_sequentially_in_separate_workers(self):
        worker, _ = self.run_process(OUTPUTS, alignment=True)
        self.assertEqual(
            worker.stages, ["decode", "asr", "check-access", "align", "diarize"]
        )

    def test_temporary_workspace_is_removed_after_processing(self):
        worker, _ = self.run_process(OUTPUTS, alignment=True)
        self.assertFalse(any(path.exists() for path in worker.workspaces))

    def test_words_are_attributed_to_the_diarized_speaker(self):
        _, result = self.run_process(OUTPUTS, alignment=True)
        self.assertEqual([w["speaker"] for w in result["words"]], ["A"])

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
                process(audio, status=silent, alignment=True, output_dir=output)
            saved = sorted(os.listdir(output))
        self.assertEqual(
            saved, ["01_raw.json", "01_raw.txt", "02_speakers.json", "02_speakers.txt"]
        )


class TranscribeTests(unittest.TestCase):
    def test_raw_transcription_stops_after_speech_recognition(self):
        worker = FakeWorker(OUTPUTS)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "test.wav"
            audio.touch()
            with patch("interview_local.pipeline.worker", worker):
                transcribe(audio, silent, False, Path(directory) / "output")
        self.assertEqual(worker.stages, ["decode", "asr"])

    def test_raw_transcription_returns_the_raw_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "test.wav"
            audio.touch()
            with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
                raw = transcribe(audio, silent, False, Path(directory) / "output")
        self.assertEqual(
            raw,
            {"schema_version": 2, "stage": "raw", "source": "test.wav", "segments": []},
        )


class PrepareTests(unittest.TestCase):
    def prepared_stages(self, **options):
        worker = FakeWorker({})
        with patch("interview_local.pipeline.worker", worker):
            prepare(status=silent, **options)
        return worker.stages

    def test_preparation_loads_diarization_and_speech_models(self):
        self.assertEqual(self.prepared_stages(), ["prepare-diarize", "prepare-asr"])

    def test_preparation_with_alignment_also_loads_the_alignment_model(self):
        self.assertEqual(
            self.prepared_stages(alignment=True),
            ["prepare-diarize", "prepare-asr", "prepare-align"],
        )


class InputTests(unittest.TestCase):
    def test_missing_audio_file_is_rejected(self):
        with self.assertRaises(ValueError):
            process(Path("/missing.wav"))

    def test_unsupported_audio_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.ogg"
            audio.touch()
            with self.assertRaises(ValueError):
                process(audio, status=silent)

    def test_output_directory_with_completed_stages_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.touch()
            (Path(directory) / "01_raw.json").touch()
            with self.assertRaises(ValueError):
                process(audio, status=silent, output_dir=Path(directory))

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
                process(audio, status=silent)


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
            process(audio, output_dir=output, status=silent)

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


if __name__ == "__main__":
    unittest.main()
