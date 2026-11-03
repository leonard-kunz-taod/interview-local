# German fixtures intentionally exercise non-ASCII speech and speaker names.
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from support.fake_ollama import IDENTITY, FakeOllama, served_by
from support.fake_worker import OUTPUTS, SPOKEN_OUTPUTS, FakeWorker
from support.model_folders import bundled_local_model, without_local_model
from support.platform_encoding import windows_default_encoding
from support.platform_files import case_sensitive_file_names

from interview_local import cli
from interview_local.artifacts import readable_text
from interview_local.cli import main
from interview_local.core import Turn, Word, merge, render
from interview_local.llm import MODEL, enhance

TOKEN = "hf_environmentToken"
SPEAKER_FILES = [
    "02_speakers.json",
    "02_speakers.txt",
    "transcript.json",
    "transcript.txt",
]
TIMESTAMPED_FOLDER = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}"


def run_cli(*arguments):
    code, stdout, stderr = run_cli_streams(*arguments)
    return code, stdout + stderr


def run_cli_streams(*arguments):
    """Exit code, standard output and standard error of one command line."""
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        patch("sys.argv", ["interview-local", *map(str, arguments)]),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        code = main()
    return code, stdout.getvalue(), stderr.getvalue()


def error_line(output):
    return next(line for line in output.splitlines() if line.startswith("Error:"))


def saved_stage(directory, name, value):
    path = Path(directory) / name
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def saved_json(path):
    return json.loads(path.read_text("utf-8"))


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

    def test_errors_are_written_to_standard_error(self):
        code, stdout, stderr = run_cli_streams("audio.wav", "--rename", "x.json")
        self.assertEqual(
            (code, stdout, stderr.splitlines()[0]),
            (1, "", "Error: Supply audio or a saved-stage command, not both."),
        )

    def test_offline_preparation_is_rejected(self):
        code, _ = run_cli("--prepare", "--offline")
        self.assertEqual(code, 1)


class RenameTests(unittest.TestCase):
    def renamed(self, transcript, names, *options):
        """Exit code, printed output and the saved speaker stage of a rename."""
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", transcript)
            output = Path(directory) / "renamed"
            with patch("builtins.input", side_effect=names):
                code, printed = run_cli(
                    "--rename", source, "--output", output, *options
                )
            saved_file = output / "02_speakers.json"
            saved = saved_json(saved_file) if saved_file.exists() else None
        return code, printed, saved

    def test_rename_saves_the_speaker_stage_and_its_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            output = Path(directory) / "renamed"
            with patch("builtins.input", side_effect=["Änne", "Bernd"]):
                run_cli("--rename", source, "--output", output)
            saved = sorted(os.listdir(output))
        self.assertEqual(saved, SPEAKER_FILES)

    def test_rename_applies_the_entered_names(self):
        _, _, saved = self.renamed(speaker_transcript(), ["Änne", "Bernd"])
        self.assertEqual(saved["speakers"], {"A": "Änne", "B": "Bernd"})

    def test_empty_name_entry_keeps_the_default_name(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            output = Path(directory) / "renamed"
            with patch("builtins.input", side_effect=["", "Bernd"]):
                run_cli("--rename", source, "--output", output)
            saved = json.loads((output / "transcript.json").read_text("utf-8"))
        self.assertEqual(saved["speakers"], {"A": "Speaker A", "B": "Bernd"})

    def test_empty_name_entries_keep_both_default_names(self):
        _, _, saved = self.renamed(speaker_transcript(), ["", ""])
        self.assertEqual(saved["speakers"], {"A": "Speaker A", "B": "Speaker B"})

    def test_rename_prints_the_renamed_transcript(self):
        _, printed, saved = self.renamed(speaker_transcript(), ["Änne", "Bernd"])
        self.assertIn(render(saved), printed)

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

    def test_rename_requires_segments_as_well_as_speakers(self):
        transcript = {"speakers": {"A": "Speaker A", "B": "Speaker B"}}
        code, printed, _ = self.renamed(transcript, ["Änne", "Bernd"])
        self.assertEqual(
            (code, error_line(printed)),
            (1, "Error: Renaming requires a speaker transcript JSON."),
        )

    def test_german_transcript_is_read_where_the_platform_default_is_not_utf8(self):
        transcript = merge([Word(0, 0.5, "Grüße")], [Turn(0, 1, "s")], [])
        with windows_default_encoding():
            _, _, saved = self.renamed(transcript, ["", ""])
        self.assertEqual(saved["words"][0]["text"], "Grüße")

    def test_cancelled_run_keeps_completed_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                code, output = run_cli("--rename", source, "--output", directory)
        self.assertEqual(
            (code, output),
            (130, "\nProcessing cancelled. Completed stage files remain on disk.\n"),
        )

    def test_cancellation_is_written_to_standard_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            with patch("builtins.input", side_effect=KeyboardInterrupt):
                code, stdout, stderr = run_cli_streams(
                    "--rename", source, "--output", Path(directory) / "o"
                )
        self.assertEqual(
            (code, stdout, stderr),
            (
                130,
                "",
                "\nProcessing cancelled. Completed stage files remain on disk.\n",
            ),
        )

    def test_rename_refuses_a_non_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            with patch("builtins.input", side_effect=["Änne", "Bernd"]):
                code, output = run_cli("--rename", source, "--output", directory)
        self.assertEqual(
            (code, error_line(output)),
            (
                1,
                "Error: Output directory is not empty. Choose a new --output directory.",
            ),
        )


class SavedStageTests(unittest.TestCase):
    def test_enhance_saves_only_the_readable_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            output = Path(directory) / "readable"
            with served_by(FakeOllama()):
                code, _ = run_cli("--enhance", source, "--output", output)
            saved = sorted(os.listdir(output))
        self.assertEqual(
            (code, saved),
            (0, ["03_readable.json", "03_readable.md", "03_readable.txt"]),
        )

    def test_enhance_prints_the_readable_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", speaker_transcript())
            output = Path(directory) / "readable"
            with served_by(FakeOllama()):
                _, printed = run_cli("--enhance", source, "--output", output)
            draft = saved_json(output / "03_readable.json")
        self.assertIn(readable_text(draft), printed)

    def test_enhance_requires_a_speaker_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "in.json", {"segments": []})
            code, _ = run_cli("--enhance", source, "--output", Path(directory) / "o")
        self.assertEqual(code, 1)

    def test_enhance_requires_speakers_as_well_as_words(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = {"words": speaker_transcript()["words"]}
            source = saved_stage(directory, "in.json", transcript)
            with served_by(FakeOllama()):
                code, output = run_cli(
                    "--enhance", source, "--output", Path(directory) / "o"
                )
        self.assertEqual(
            (code, error_line(output)),
            (1, "Error: Enhancement requires a stage-2 speaker transcript JSON."),
        )

    def test_review_saves_only_the_review_stage(self):
        with served_by(FakeOllama()):
            draft = enhance(speaker_transcript(), lambda _: None)
        with tempfile.TemporaryDirectory() as directory:
            source = saved_stage(directory, "03_readable.json", draft)
            output = Path(directory) / "review"
            with served_by(FakeOllama()):
                code, _ = run_cli("--review", source, "--output", output)
            saved = sorted(os.listdir(output))
        self.assertEqual(
            (code, saved), (0, ["04_review.json", "04_review.md", "04_review.txt"])
        )


class DoctorTests(unittest.TestCase):
    def doctor(self, service=None, system="Darwin", machine="arm64"):
        memory = SimpleNamespace(total=16 * 2**30, available=8 * 2**30)
        with (
            served_by(service or FakeOllama()),
            patch("platform.system", return_value=system),
            patch("platform.machine", return_value=machine),
            patch("psutil.virtual_memory", return_value=memory),
            patch("torch.cuda.is_available", return_value=False),
        ):
            return run_cli("--doctor")

    def test_doctor_reports_the_local_llm(self):
        code, output = self.doctor()
        self.assertEqual((code, output.splitlines()[-1]), (0, f"Local LLM: {IDENTITY}"))

    def test_unreachable_llm_is_reported_without_failing(self):
        code, output = self.doctor(FakeOllama(reachable=False))
        self.assertEqual(
            (code, output.splitlines()[-1].startswith("Cannot reach local Ollama.")),
            (0, True),
        )

    def test_doctor_reports_total_and_available_ram(self):
        _, output = self.doctor()
        self.assertIn("RAM: 16.0 GiB total; 8.0 GiB available\n", output)

    def test_unreadable_ram_is_reported(self):
        with (
            served_by(FakeOllama()),
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

    def test_intel_mac_gets_no_shared_memory_note(self):
        _, output = self.doctor(system="Darwin", machine="x86_64")
        self.assertNotIn("there is no additional dedicated VRAM", output)

    def test_memory_of_the_first_cuda_gpu_is_reported(self):
        first_gpu = SimpleNamespace(total_memory=8 * 2**30)
        with (
            served_by(FakeOllama()),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", {0: first_gpu}.__getitem__),
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

    def test_speech_preparation_hands_the_environment_token_to_every_worker(self):
        worker = FakeWorker({})
        with (
            without_local_model(),
            patch.dict(os.environ, {"HF_TOKEN": TOKEN}),
            patch("interview_local.pipeline.worker", worker),
        ):
            run_cli("--prepare")
        self.assertEqual(worker.tokens, {TOKEN})

    def test_llm_preparation_downloads_only_the_language_model(self):
        service = FakeOllama(installed=())
        worker = FakeWorker({})
        with served_by(service), patch("interview_local.pipeline.worker", worker):
            code, _ = run_cli("--prepare-llm")
        self.assertEqual(
            (code, MODEL in service.installed, worker.stages), (0, True, [])
        )


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
                code, printed = run_cli("--offline", "--no-llm", "--output", output)
            saved_file = output / "02_speakers.json"
            speakers = (
                saved_json(saved_file)["speakers"] if saved_file.exists() else None
            )
        return code, speakers, printed

    def test_quoted_dragged_path_is_processed_and_names_are_asked(self):
        code, speakers, _ = self.run_interactive(
            "interview.wav", lambda audio: f"'{audio}' ", "Änne", ""
        )
        self.assertEqual((code, speakers), (0, {"A": "Änne", "B": "Speaker B"}))

    def test_unquoted_path_with_spaces_is_taken_literally(self):
        code, speakers, _ = self.run_interactive("my interview.wav", str, "", "")
        self.assertEqual((code, speakers), (0, {"A": "Speaker A", "B": "Speaker B"}))

    def windows_entry(self, entered_path):
        """The error a Windows terminal reports for a file named take\\X."""
        windows = SimpleNamespace(name="nt", environ=os.environ)
        with patch.object(cli, "os", windows):
            _, _, printed = self.run_interactive("take\\X", entered_path, "", "")
        return error_line(printed)

    def test_windows_path_is_used_exactly_as_entered_inside_its_quotes(self):
        self.assertEqual(
            self.windows_entry(lambda audio: f'"{audio}"'),
            "Error: Please select a WAV, MP3 or M4A file.",
        )

    def test_windows_path_keeps_its_backslashes(self):
        self.assertEqual(
            self.windows_entry(str), "Error: Please select a WAV, MP3 or M4A file."
        )


class AudioTests(unittest.TestCase):
    def run_audio(self, *options, outputs=OUTPUTS):
        """Exit code, saved files, printed output and the fake speech workers."""
        worker = FakeWorker(outputs)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "interview.wav"
            audio.touch()
            output = Path(directory) / "run"
            with patch("interview_local.pipeline.worker", worker):
                code, printed = run_cli(
                    audio, "--offline", "--output", output, *options
                )
            saved = sorted(os.listdir(output)) if output.exists() else []
            speakers_file = output / "02_speakers.json"
            speakers = saved_json(speakers_file) if speakers_file.exists() else None
        return SimpleNamespace(
            code=code, saved=saved, printed=printed, worker=worker, speakers=speakers
        )

    def test_missing_audio_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "interview.wav"
            code, output = run_cli(
                missing, "--offline", "--no-llm", "--output", Path(directory) / "o"
            )
        self.assertEqual(
            (code, output.splitlines()[0]),
            (1, "Error: Audio file not found. Enter its full path."),
        )

    def test_names_from_the_command_line_are_saved_without_asking(self):
        run = self.run_audio("--no-llm", "--name-a", "Änne", "--name-b", "Bernd")
        self.assertEqual(run.speakers["speakers"], {"A": "Änne", "B": "Bernd"})

    def test_unreachable_llm_stops_before_speech_processing(self):
        worker = FakeWorker(OUTPUTS)
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "interview.wav"
            audio.touch()
            with (
                served_by(FakeOllama(reachable=False)),
                patch("interview_local.pipeline.worker", worker),
            ):
                code, _ = run_cli(audio, "--offline", "--output", Path(directory) / "o")
        self.assertEqual((code, worker.stages), (1, []))

    def test_full_run_points_to_the_separate_review(self):
        with served_by(FakeOllama()):
            run = self.run_audio()
        self.assertIn(
            "For the separate comparison pass, run --review on 03_readable.json.",
            run.printed,
        )

    def test_full_run_reports_the_editor_progress(self):
        with served_by(FakeOllama()):
            run = self.run_audio(outputs=SPOKEN_OUTPUTS)
        self.assertIn("LLM readability: words 1–1 / 1\n", run.printed)

    def test_run_without_llm_prints_the_speaker_transcript(self):
        self.assertIn("No speech detected.\n", self.run_audio("--no-llm").printed)

    def test_offline_run_starts_every_worker_offline(self):
        self.assertEqual(self.run_audio("--no-llm").worker.offline, {True})

    def test_alignment_flag_aligns_the_words(self):
        self.assertIn("align", self.run_audio("--no-llm", "--align").worker.stages)

    def test_no_align_flag_runs_without_alignment(self):
        run = self.run_audio("--no-llm", "--no-align")
        self.assertEqual((run.code, "align" in run.worker.stages), (0, False))

    def test_run_records_that_alignment_was_not_used(self):
        metadata = self.run_audio("--no-llm").speakers["metadata"]
        self.assertIs(metadata["alignment"], False)

    def test_raw_only_saves_only_the_raw_stage(self):
        run = self.run_audio("--raw-only")
        self.assertEqual((run.code, run.saved), (0, ["01_raw.json", "01_raw.txt"]))

    def test_raw_only_prints_the_raw_transcript(self):
        run = self.run_audio("--raw-only", outputs=SPOKEN_OUTPUTS)
        self.assertIn("[00:00:00.000 – 00:00:01.000]\nGrüße\n", run.printed)

    def test_raw_only_runs_offline_when_asked(self):
        self.assertEqual(self.run_audio("--raw-only").worker.offline, {True})

    def test_raw_transcript_is_printed_on_a_case_sensitive_file_system(self):
        with case_sensitive_file_names():
            run = self.run_audio("--raw-only", outputs=SPOKEN_OUTPUTS)
        self.assertIn("\nGrüße\n", run.printed)

    def test_german_raw_transcript_is_printed_where_the_default_is_not_utf8(self):
        with windows_default_encoding():
            run = self.run_audio("--raw-only", outputs=SPOKEN_OUTPUTS)
        self.assertIn("\nGrüße\n", run.printed)

    def test_run_without_llm_saves_raw_and_speaker_stages(self):
        run = self.run_audio("--no-llm")
        self.assertEqual(
            (run.code, run.saved), (0, ["01_raw.json", "01_raw.txt", *SPEAKER_FILES])
        )

    def test_full_run_adds_the_readable_stage(self):
        with served_by(FakeOllama()):
            run = self.run_audio()
        self.assertEqual(
            (run.code, run.saved),
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


class OutputDirectoryTests(unittest.TestCase):
    def run_into(self, directory, output):
        audio = Path(directory) / "interview.wav"
        audio.touch()
        with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
            return run_cli(audio, "--offline", "--no-llm", "--output", output)

    def test_default_output_is_a_new_timestamped_folder_under_transcripts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.chdir(directory):
            audio = Path(directory) / "interview.wav"
            audio.touch()
            with patch("interview_local.pipeline.worker", FakeWorker(OUTPUTS)):
                run_cli(audio, "--offline", "--no-llm")
            runs = os.listdir(Path(directory) / "transcripts")
            created = sorted(os.listdir(directory))
        self.assertEqual(
            (created, len(runs), re.fullmatch(TIMESTAMPED_FOLDER, runs[0]) is not None),
            (["interview.wav", "transcripts"], 1, True),
        )

    def test_existing_empty_output_directory_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            code, _ = self.run_into(directory, output)
            saved = sorted(os.listdir(output))
        self.assertEqual(
            (code, saved), (0, ["01_raw.json", "01_raw.txt", *SPEAKER_FILES])
        )

    def test_missing_parents_of_the_output_directory_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "interviews" / "first"
            code, _ = self.run_into(directory, output)
            saved = sorted(os.listdir(output))
        self.assertEqual(
            (code, saved), (0, ["01_raw.json", "01_raw.txt", *SPEAKER_FILES])
        )


if __name__ == "__main__":
    unittest.main()
