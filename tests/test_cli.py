# German fixtures intentionally exercise non-ASCII speech and speaker names.
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fake_ollama import FakeClient
from test_models import bundled_local_model, without_local_model
from test_pipeline import OUTPUTS, FakeWorker

from interview_local.cli import main
from interview_local.core import Turn, Word, merge
from interview_local.llm import enhance

TOKEN = "hf_environmentToken"
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
        [Word(0, 0.3, "Hallo"), Word(0.3, 0.7, "zusammen.")], [Turn(0, 1, "s")], []
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

    def test_rename_requires_a_speaker_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", {"words": []})
            code, output = run_cli(
                "--rename", source, "--output", Path(directory) / "o"
            )
        self.assertEqual(
            (code, output.splitlines()[0]),
            (1, "Error: Renaming requires a speaker transcript JSON."),
        )

    def test_cancelled_run_keeps_completed_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                code, output = run_cli("--rename", source, "--output", directory)
        self.assertEqual(
            (code, output),
            (130, "\nProcessing cancelled. Completed stage files remain on disk.\n"),
        )

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
        with patch("interview_local.llm.OllamaClient", FakeClient):
            draft = enhance(speaker_transcript(), lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "03_readable.json", draft)
            output = Path(directory) / "review"
            with patch("interview_local.llm.OllamaClient", FakeClient):
                run_cli("--review", source, "--output", output)
            saved = sorted(os.listdir(output))
        self.assertEqual(saved, ["04_review.json", "04_review.md", "04_review.txt"])


class UnreachableClient(FakeClient):
    def ready(self):
        raise RuntimeError("Cannot reach local Ollama.")


class DoctorTests(unittest.TestCase):
    def doctor(self, client=FakeClient, system="Darwin", machine="arm64"):
        memory = SimpleNamespace(total=16 * 2**30, available=8 * 2**30)
        with (
            patch("interview_local.llm.OllamaClient", client),
            patch("platform.system", return_value=system),
            patch("platform.machine", return_value=machine),
            patch("psutil.virtual_memory", return_value=memory),
            patch("torch.cuda.is_available", return_value=False),
        ):
            return run_cli("--doctor")

    def test_doctor_reports_the_local_llm(self):
        code, output = self.doctor()
        self.assertEqual(
            (code, output.splitlines()[-1]),
            (0, "Local LLM: {'model': 'fake-local-test'}"),
        )

    def test_unreachable_llm_is_reported_without_failing(self):
        code, output = self.doctor(UnreachableClient)
        self.assertEqual(
            (code, output.splitlines()[-1]), (0, "Cannot reach local Ollama.")
        )

    def test_doctor_reports_total_and_available_ram(self):
        _, output = self.doctor()
        self.assertIn("RAM: 16.0 GiB total; 8.0 GiB available\n", output)

    def test_unreadable_ram_is_reported(self):
        with (
            patch("interview_local.llm.OllamaClient", FakeClient),
            patch("psutil.virtual_memory", side_effect=OSError),
        ):
            _, output = run_cli("--doctor")
        self.assertIn("Could not read RAM information automatically.\n", output)

    def test_apple_silicon_notes_shared_memory(self):
        _, output = self.doctor(system="Darwin", machine="arm64")
        self.assertIn("there is no additional dedicated VRAM", output)

    def test_other_systems_get_no_shared_memory_note(self):
        _, output = self.doctor(system="Linux", machine="x86_64")
        self.assertNotIn("there is no additional dedicated VRAM", output)

    def test_cuda_gpu_memory_is_reported(self):
        device = SimpleNamespace(total_memory=8 * 2**30)
        with (
            patch("interview_local.llm.OllamaClient", FakeClient),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=device),
        ):
            _, output = run_cli("--doctor")
        self.assertIn("Dedicated GPU memory: 8.0 GiB\n", output)


class GuiFlagTests(unittest.TestCase):
    def test_gui_flag_opens_the_window(self):
        opened = []
        window = types.ModuleType("interview_local.gui")
        window.launch = lambda: opened.append(True)
        with patch.dict(sys.modules, {"interview_local.gui": window}):
            code, _ = run_cli("--gui")
        self.assertEqual((code, opened), (0, [True]))

    def test_missing_tk_falls_back_to_the_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            output = Path(directory) / "renamed"
            with (
                patch.dict(sys.modules, {"interview_local.gui": None}),
                patch("builtins.input", side_effect=["Änne", "Bernd"]),
            ):
                code, printed = run_cli("--gui", "--rename", source, "--output", output)
            saved = sorted(os.listdir(output))
        self.assertEqual(
            (code, printed.splitlines()[0], saved),
            (
                0,
                "Tk is not installed. Falling back to the terminal dialog.",
                SPEAKER_FILES,
            ),
        )


class PrepareTests(unittest.TestCase):
    def test_speech_preparation_loads_every_speech_model(self):
        worker = FakeWorker({})
        with bundled_local_model(), patch("interview_local.pipeline.worker", worker):
            code, output = run_cli("--prepare", "--align")
        self.assertEqual(
            (code, worker.stages, output.splitlines()[-1]),
            (
                0,
                ["prepare-diarize", "prepare-asr", "prepare-align"],
                "Preparation complete.",
            ),
        )

    def test_llm_preparation_downloads_only_the_language_model(self):
        client = FakeClient()
        worker = FakeWorker({})
        with (
            patch("interview_local.llm.OllamaClient", lambda: client),
            patch("interview_local.pipeline.worker", worker),
        ):
            code, _ = run_cli("--prepare-llm")
        self.assertEqual((code, client.prepared, worker.stages), (0, True, []))


class TokenTests(unittest.TestCase):
    def worker_tokens(self, environment, cached=None, entered=""):
        """Tokens the speech workers receive, and the printed output."""
        worker = FakeWorker(OUTPUTS)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "interview.wav"
            audio.touch()
            with (
                patch.dict(os.environ, environment),
                patch("interview_local.pipeline.worker", worker),
                patch("huggingface_hub.get_token", return_value=cached),
                patch("getpass.getpass", return_value=entered),
            ):
                _, output = run_cli(
                    audio, "--no-llm", "--output", Path(directory) / "run"
                )
        return worker.tokens, output

    def test_local_model_needs_no_token(self):
        with bundled_local_model():
            tokens, _ = self.worker_tokens({"HF_TOKEN": TOKEN})
        self.assertEqual(tokens, {None})

    def test_environment_token_is_used_for_the_hub(self):
        with without_local_model():
            tokens, _ = self.worker_tokens({"HF_TOKEN": TOKEN})
        self.assertEqual(tokens, {TOKEN})

    def test_cached_login_is_used_without_an_environment_token(self):
        with without_local_model():
            tokens, _ = self.worker_tokens({"HF_TOKEN": ""}, cached="hf_cached")
        self.assertEqual(tokens, {"hf_cached"})

    def test_entered_token_is_used_without_any_saved_token(self):
        with without_local_model():
            tokens, output = self.worker_tokens({"HF_TOKEN": ""}, entered=" hf_typed ")
        self.assertEqual(
            (tokens, "speaker-diarization-community-1 conditions" in output),
            ({"hf_typed"}, True),
        )

    def test_skipped_token_entry_runs_without_a_token(self):
        with without_local_model():
            tokens, _ = self.worker_tokens({"HF_TOKEN": ""}, entered="")
        self.assertEqual(tokens, {None})

    def test_offline_run_asks_for_no_token(self):
        worker = FakeWorker(OUTPUTS)
        with (
            tempfile.TemporaryDirectory() as directory,
            without_local_model(),
            patch.dict(os.environ, {"HF_TOKEN": TOKEN}),
            patch("interview_local.pipeline.worker", worker),
        ):
            audio = Path(directory) / "interview.wav"
            audio.touch()
            run_cli(audio, "--offline", "--no-llm", "--output", Path(directory) / "o")
        self.assertEqual(worker.tokens, {None})


class InteractiveAudioTests(unittest.TestCase):
    def run_interactive(self, file_name, entered_path, *names):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / file_name
            audio.touch()
            output = Path(directory) / "run"
            entries = [entered_path(audio), *names]
            with (
                patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)),
                patch("builtins.input", side_effect=entries),
            ):
                code, _ = run_cli("--offline", "--no-llm", "--output", output)
            saved = json.loads((output / "02_speakers.json").read_text("utf-8"))
        return code, saved["speakers"]

    def test_quoted_dragged_path_is_processed_and_names_are_asked(self):
        outcome = self.run_interactive(
            "interview.wav", lambda audio: f"'{audio}' ", "Änne", ""
        )
        self.assertEqual(outcome, (0, {"A": "Änne", "B": "Speaker B"}))

    def test_unquoted_path_with_spaces_is_taken_literally(self):
        outcome = self.run_interactive("my interview.wav", str, "", "")
        self.assertEqual(outcome, (0, {"A": "Speaker A", "B": "Speaker B"}))


class AudioTests(unittest.TestCase):
    def run_audio(self, *options):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "interview.wav"
            audio.touch()
            output = Path(directory) / "run"
            with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
                code, _ = run_cli(audio, "--offline", "--output", output, *options)
            return code, sorted(os.listdir(output))

    def test_missing_audio_file_is_reported(self):
        code, output = run_cli("/missing/interview.wav", "--offline", "--no-llm")
        self.assertEqual(
            (code, output.splitlines()[0]),
            (1, "Error: Audio file not found. Enter its full path."),
        )

    def test_names_from_the_command_line_are_saved_without_asking(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "interview.wav"
            audio.touch()
            output = Path(directory) / "run"
            with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
                run_cli(
                    audio,
                    "--offline",
                    "--no-llm",
                    "--output",
                    output,
                    "--name-a",
                    "Änne",
                    "--name-b",
                    "Bernd",
                )
            saved = json.loads((output / "02_speakers.json").read_text("utf-8"))
        self.assertEqual(saved["speakers"], {"A": "Änne", "B": "Bernd"})

    def test_unreachable_llm_stops_before_speech_processing(self):
        worker = FakeWorker(OUTPUTS)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "interview.wav"
            audio.touch()
            with (
                patch("interview_local.llm.OllamaClient", UnreachableClient),
                patch("interview_local.pipeline.worker", worker),
            ):
                code, _ = run_cli(audio, "--offline")
        self.assertEqual((code, worker.stages), (1, []))

    def test_full_run_points_to_the_separate_review(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "interview.wav"
            audio.touch()
            with (
                patch("interview_local.llm.OllamaClient", FakeClient),
                patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)),
            ):
                _, output = run_cli(
                    audio, "--offline", "--output", Path(directory) / "o"
                )
        self.assertIn(
            "For the separate comparison pass, run --review on 03_readable.json.",
            output,
        )

    def test_run_without_llm_prints_the_speaker_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "interview.wav"
            audio.touch()
            with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
                _, output = run_cli(
                    audio, "--offline", "--no-llm", "--output", Path(directory) / "o"
                )
        self.assertIn("No speech detected.\n", output)

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
