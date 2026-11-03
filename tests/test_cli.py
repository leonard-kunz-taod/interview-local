# German fixtures intentionally exercise non-ASCII speech and speaker names.
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fake_ollama import FakeClient
from test_pipeline import OUTPUTS, FakeWorker

from interview_local.cli import main
from interview_local.core import Turn, Word, merge
from interview_local.llm import enhance

SPEAKER_FILES = [
    "02_speakers.json",
    "02_speakers.txt",
    "transcript.json",
    "transcript.txt",
]


def run_cli(*arguments):
    output = io.StringIO()
    with (
        patch("sys.argv", ["interview-local", *map(str, arguments)]),
        contextlib.redirect_stdout(output),
        contextlib.redirect_stderr(output),
    ):
        code = main()
    return code, output.getvalue()


def saved_stage(directory, name, value):
    path = Path(directory) / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def speaker_transcript():
    return merge(
        [Word(0, 0.3, "Hallo"), Word(0.3, 0.7, "zusammen.")], [Turn(0, 1, "s")]
    )


class ArgumentTests(unittest.TestCase):
    def test_audio_combined_with_a_saved_stage_command_is_rejected(self):
        code, output = run_cli("audio.wav", "--rename", "speakers.json")
        self.assertEqual(
            (code, output),
            (
                1,
                (
                    "Error: Supply audio or a saved-stage command, not both.\n"
                    "Completed earlier stages remain on disk, if any.\n"
                ),
            ),
        )

    def test_offline_preparation_is_rejected(self):
        code, _ = run_cli("--prepare", "--offline")
        self.assertEqual(code, 1)


class RenameTests(unittest.TestCase):
    def test_rename_saves_the_speaker_stage_and_its_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            output = Path(directory) / "renamed"
            with patch("builtins.input", side_effect=["Änne", "Bernd"]):
                run_cli("--rename", source, "--output", output)
            saved = sorted(os.listdir(output))
        self.assertEqual(saved, SPEAKER_FILES)

    def test_rename_applies_the_entered_names(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            output = Path(directory) / "renamed"
            with patch("builtins.input", side_effect=["Änne", "Bernd"]):
                run_cli("--rename", source, "--output", output)
            saved = json.loads((output / "02_speakers.json").read_text("utf-8"))
        self.assertEqual(saved["speakers"], {"A": "Änne", "B": "Bernd"})

    def test_empty_name_entry_keeps_the_default_name(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            output = Path(directory) / "renamed"
            with patch("builtins.input", side_effect=["", "Bernd"]):
                run_cli("--rename", source, "--output", output)
            saved = json.loads((output / "transcript.json").read_text("utf-8"))
        self.assertEqual(saved["speakers"], {"A": "Speaker A", "B": "Bernd"})

    def test_rename_refuses_a_non_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            with patch("builtins.input", side_effect=["Änne", "Bernd"]):
                code, _ = run_cli("--rename", source, "--output", directory)
        self.assertEqual(code, 1)


class SavedStageTests(unittest.TestCase):
    def test_enhance_saves_only_the_readable_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            output = Path(directory) / "readable"
            with patch("interview_local.llm.OllamaClient", FakeClient):
                run_cli("--enhance", source, "--output", output)
            saved = sorted(os.listdir(output))
        self.assertEqual(
            saved, ["03_readable.json", "03_readable.md", "03_readable.txt"]
        )

    def test_enhance_requires_a_speaker_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", {"segments": []})
            code, _ = run_cli("--enhance", source, "--output", Path(directory) / "o")
        self.assertEqual(code, 1)

    def test_review_saves_only_the_review_stage(self):
        draft = enhance(
            speaker_transcript(), client=FakeClient(), status=lambda _: None
        )
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "03_readable.json", draft)
            output = Path(directory) / "review"
            with patch("interview_local.llm.OllamaClient", FakeClient):
                run_cli("--review", source, "--output", output)
            saved = sorted(os.listdir(output))
        self.assertEqual(saved, ["04_review.json", "04_review.md", "04_review.txt"])


class AudioTests(unittest.TestCase):
    def run_audio(self, *options):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "interview.wav"
            audio.touch()
            output = Path(directory) / "run"
            with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
                code, _ = run_cli(audio, "--offline", "--output", output, *options)
            return code, sorted(os.listdir(output))

    def test_raw_only_saves_only_the_raw_stage(self):
        self.assertEqual(
            self.run_audio("--raw-only"), (0, ["01_raw.json", "01_raw.txt"])
        )

    def test_run_without_llm_saves_raw_and_speaker_stages(self):
        self.assertEqual(
            self.run_audio("--no-llm"),
            (0, ["01_raw.json", "01_raw.txt", *SPEAKER_FILES]),
        )

    def test_full_run_adds_the_readable_stage(self):
        with patch("interview_local.llm.OllamaClient", FakeClient):
            outcome = self.run_audio()
        self.assertEqual(
            outcome,
            (
                0,
                [
                    "01_raw.json",
                    "01_raw.txt",
                    *SPEAKER_FILES[:2],
                    "03_readable.json",
                    "03_readable.md",
                    "03_readable.txt",
                    *SPEAKER_FILES[2:],
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
