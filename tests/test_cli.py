from contextlib import chdir, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from interview_local.artifacts import readable_text
from interview_local.cli import main
from interview_local.core import merge
from run_fixtures import (INTERVIEW_TURNS, TWO_PARAGRAPHS, ScriptedClient, gated_speech_models, interview_words,
                          readable_transcript, speech_models, uncached_speech_models)

SPEECH_PARAGRAPHS = [(0, 1, "A", "Guten Tag.")]
HF_TOKEN = "hf_test"
WINDOWS_OS = SimpleNamespace(**{**vars(os), "name": "nt"})


class PreparingClient(ScriptedClient):
    """A local LLM that downloads its model instantly."""

    def prepare(self, status=print):
        status("Ministral downloaded")


class StoppedServer:
    """A web server stand-in that needs no port and is stopped by Ctrl+C at once."""

    server_address = ("127.0.0.1", 8765)

    def __init__(self, runs_root):
        self.runs_root = runs_root

    def serve_forever(self):
        raise KeyboardInterrupt

    def server_close(self):
        pass


def cancelled_input(prompt):
    raise KeyboardInterrupt


class UnreachableClient(ScriptedClient):
    """A local LLM service that is not running."""

    def ready(self):
        raise RuntimeError("Ollama is not running")


def unanswered_token_prompt(prompt):
    raise AssertionError("The run asked for an HF token.")


def run_cli(*arguments, answers=(), paragraphs=SPEECH_PARAGRAPHS, models=None, llm=PreparingClient):
    """Run the command line with stand-ins for the speech models and the local LLM.

    The HF token comes from HF_TOKEN; there is no local diarization model, and the stand-in for the gated
    speaker-recognition model accepts only that token.

    Returns:
        The exit code, standard output and standard error.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    typed = iter(answers)
    with (patch.object(sys, "argv", ["interview-local", *map(str, arguments)]),
          patch("builtins.input", lambda prompt: next(typed)),
          patch("interview_local.pipeline.worker", models or gated_speech_models(HF_TOKEN)),
          patch("interview_local.llm.OllamaClient", lambda: llm(paragraphs)),
          patch("interview_local.models.local_diarization_model", lambda: None),
          patch.dict(os.environ, {"HF_TOKEN": HF_TOKEN}),
          redirect_stdout(stdout), redirect_stderr(stderr)):
        code = main()
    return code, stdout.getvalue(), stderr.getvalue()


def run_cli_without_credentials(*arguments, models):
    """Run the command line without HF_TOKEN or a cached login; asking for a token fails the run."""
    stdout, stderr = io.StringIO(), io.StringIO()
    environment = {key: value for key, value in os.environ.items() if key != "HF_TOKEN"}
    with (patch.object(sys, "argv", ["interview-local", *map(str, arguments)]),
          patch("interview_local.pipeline.worker", models),
          patch("interview_local.models.local_diarization_model", lambda: None),
          patch("huggingface_hub.get_token", lambda: None),
          patch("interview_local.cli.getpass.getpass", unanswered_token_prompt),
          patch.dict(os.environ, environment, clear=True),
          redirect_stdout(stdout), redirect_stderr(stderr)):
        code = main()
    return code, stdout.getvalue(), stderr.getvalue()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def speaker_transcript():
    return merge(interview_words(), INTERVIEW_TURNS, [])


class CommandLineTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.output = self.root / "output"
        self.audio = self.root / "interview.wav"
        self.audio.touch()

    def tearDown(self):
        self.directory.cleanup()

    def saved(self, name):
        return json.loads((self.output / name).read_text(encoding="utf-8"))


class OptionTests(CommandLineTestCase):
    def test_web_ui_serves_the_resolved_runs_directory(self):
        with (patch("interview_local.web.server.EditorServer", StoppedServer),
              patch("interview_local.web.server.webbrowser.open")):
            code, printed, _ = run_cli("--web", "--runs-dir", self.root / "runs")
        self.assertEqual((code, f"Runs are saved in {(self.root / 'runs').resolve()}" in printed), (0, True))

    def test_web_ui_serves_transcripts_in_the_working_directory_by_default(self):
        with (patch("interview_local.web.server.EditorServer", StoppedServer),
              patch("interview_local.web.server.webbrowser.open"), chdir(self.root)):
            _, printed, _ = run_cli("--web")
        self.assertIn(f"Runs are saved in {(self.root / 'transcripts').resolve()}", printed)

    def test_audio_together_with_a_saved_stage_command_is_rejected(self):
        code, _, error = run_cli(self.audio, "--rename", self.root / "speakers.json")
        self.assertEqual((code, "Supply audio or a saved-stage command" in error), (1, True))

    def test_preparation_offline_is_rejected(self):
        code, _, error = run_cli("--offline", "--prepare-llm")
        self.assertEqual((code, "do not combine it with --offline" in error), (1, True))

    def test_doctor_reports_and_exits(self):
        with patch("interview_local.cli.doctor", lambda: print("Doctor report")):
            code, printed, _ = run_cli("--doctor")
        self.assertEqual((code, printed), (0, "Doctor report\n"))

    def test_gui_launches_when_tk_is_available(self):
        gui = SimpleNamespace(launch=lambda: print("GUI closed"))
        with patch.dict(sys.modules, {"interview_local.gui": gui}):
            code, printed, _ = run_cli("--gui")
        self.assertEqual((code, printed), (0, "GUI closed\n"))

    def test_gui_without_tk_falls_back_to_the_terminal(self):
        with patch.dict(sys.modules, {"interview_local.gui": None}):
            code, printed, _ = run_cli("--gui", "--prepare-llm")
        self.assertEqual((code, "Falling back to the terminal dialog" in printed, "Preparation complete." in printed),
                         (0, True, True))

    def test_speech_model_preparation_checks_every_stage(self):
        code, printed, _ = run_cli("--prepare", "--align")
        checked = [line.split(": ")[1] for line in printed.splitlines() if line.startswith("Model download")]
        self.assertEqual((code, checked), (0, ["prepare-diarize", "prepare-asr", "prepare-align"]))

    def test_llm_preparation_downloads_the_model(self):
        code, printed, _ = run_cli("--prepare-llm")
        self.assertEqual((code, printed), (0, "Ministral downloaded\nPreparation complete.\n"))

    def test_ctrl_c_cancels_with_exit_code_130(self):
        with patch("builtins.input", cancelled_input):
            stderr = io.StringIO()
            with (patch.object(sys, "argv", ["interview-local"]), redirect_stdout(io.StringIO()),
                  redirect_stderr(stderr)):
                code = main()
        self.assertEqual((code, "Processing cancelled" in stderr.getvalue()), (130, True))


class SavedStageTests(CommandLineTestCase):
    def test_review_saves_the_reviewed_draft(self):
        readable = write_json(self.root / "03_readable.json", readable_transcript(TWO_PARAGRAPHS))
        code, printed, _ = run_cli("--review", readable, "--output", self.output, paragraphs=TWO_PARAGRAPHS)
        self.assertEqual((code, self.saved("04_review.json")["stage"], "Review saved to" in printed),
                         (0, "review", True))

    def test_enhancement_saves_a_readable_draft(self):
        source = write_json(self.root / "02_speakers.json", speaker_transcript())
        code, _, _ = run_cli("--enhance", source, "--output", self.output, paragraphs=TWO_PARAGRAPHS)
        self.assertEqual((code, self.saved("03_readable.json")["stage"]), (0, "readable"))

    def test_enhancement_prints_the_readable_draft(self):
        source = write_json(self.root / "02_speakers.json", speaker_transcript())
        _, printed, _ = run_cli("--enhance", source, "--output", self.output, paragraphs=TWO_PARAGRAPHS)
        self.assertIn(readable_text(self.saved("03_readable.json")), printed)

    def test_enhancement_requires_a_speaker_transcript(self):
        for case, transcript in (("raw transcript", {"segments": []}),
                                 ("no words", {"speakers": {"A": "Anna"}, "segments": []}),
                                 ("no speakers", {"words": [], "segments": []})):
            with self.subTest(case):
                source = write_json(self.root / "02_speakers.json", transcript)
                code, _, error = run_cli("--enhance", source, "--output", self.output)
                self.assertEqual((code, "Enhancement requires" in error), (1, True))

    def test_renaming_saves_the_entered_names(self):
        source = write_json(self.root / "02_speakers.json", speaker_transcript())
        code, _, _ = run_cli("--rename", source, "--output", self.output, answers=("Anna", ""))
        self.assertEqual((code, self.saved("02_speakers.json")["speakers"]), (0, {"A": "Anna", "B": "Speaker B"}))

    def test_renaming_prints_the_renamed_transcript(self):
        source = write_json(self.root / "02_speakers.json", speaker_transcript())
        _, printed, _ = run_cli("--rename", source, "--output", self.output, answers=("Anna", "Bernd"))
        self.assertEqual(("] Anna\n" in printed, "] Bernd\n" in printed), (True, True))

    def test_renaming_requires_a_speaker_transcript(self):
        for case, transcript in (("words only", {"words": []}), ("no segments", {"speakers": {}, "words": []}),
                                 ("no speakers", {"segments": [], "words": []})):
            with self.subTest(case):
                source = write_json(self.root / "02_speakers.json", transcript)
                code, _, error = run_cli("--rename", source, "--output", self.output)
                self.assertEqual((code, "Renaming requires" in error), (1, True))

    def test_review_prints_the_reviewed_draft(self):
        readable = write_json(self.root / "03_readable.json", readable_transcript(TWO_PARAGRAPHS))
        _, printed, _ = run_cli("--review", readable, "--output", self.output, paragraphs=TWO_PARAGRAPHS)
        self.assertIn(readable_text(self.saved("04_review.json")), printed)


class TranscriptionTests(CommandLineTestCase):
    def test_missing_audio_file_is_rejected(self):
        code, _, error = run_cli(self.root / "missing.wav")
        self.assertEqual((code, "Audio file not found" in error), (1, True))

    def test_raw_only_prints_the_raw_transcript_and_stops(self):
        code, printed, _ = run_cli(self.audio, "--raw-only", "--output", self.output)
        saved = sorted(p.name for p in self.output.iterdir())
        raw_text = (self.output / "01_raw.txt").read_text(encoding="utf-8")
        self.assertEqual((code, raw_text in printed, saved), (0, True, ["01_raw.json", "01_raw.txt"]))

    def test_transcription_prints_the_readable_draft(self):
        _, printed, _ = run_cli(self.audio, "--output", self.output)
        self.assertIn(readable_text(self.saved("03_readable.json")), printed)

    def test_alignment_option_aligns_the_words(self):
        run_cli(self.audio, "--align", "--no-llm", "--output", self.output)
        metadata = self.saved("02_speakers.json")["metadata"]
        self.assertEqual((metadata["alignment"], metadata["word_timing_counts"]), (True, {"aligned": 2}))

    def test_transcription_without_alignment_option_records_no_alignment(self):
        for case, options in (("default", ()), ("--no-align", ("--no-align",))):
            with self.subTest(case):
                output = self.root / case
                code, _, _ = run_cli(self.audio, *options, "--no-llm", "--output", output)
                metadata = json.loads((output / "02_speakers.json").read_text(encoding="utf-8"))["metadata"]
                self.assertEqual((code, metadata["alignment"]), (0, False))

    def test_offline_transcription_uses_only_cached_models(self):
        for case, options in (("speakers", ("--no-llm",)), ("raw only", ("--raw-only",))):
            with self.subTest(case):
                code, _, error = run_cli(self.audio, "--offline", *options, "--output", self.root / case,
                                         models=uncached_speech_models())
                self.assertEqual((code, "not in the local cache" in error), (1, True))

    def test_offline_transcription_never_asks_for_an_hf_token(self):
        code, _, _ = run_cli_without_credentials(self.audio, "--offline", "--no-llm", "--output", self.output,
                                                 models=speech_models())
        self.assertEqual(code, 0)

    def test_unreachable_local_llm_stops_the_run_before_the_speech_models(self):
        code, _, _ = run_cli(self.audio, "--output", self.output, llm=UnreachableClient)
        self.assertEqual((code, self.output.exists()), (1, False))

    def test_without_llm_an_unreachable_local_llm_is_not_needed(self):
        code, _, _ = run_cli(self.audio, "--no-llm", "--output", self.output, llm=UnreachableClient)
        self.assertEqual(code, 0)

    def test_asked_unquoted_path_with_spaces_is_found(self):
        spaced = self.root / "my interview.wav"
        spaced.touch()
        code, _, _ = run_cli("--no-llm", "--output", self.output, answers=(str(spaced), "Anna", "Bernd"))
        self.assertEqual(code, 0)

    def test_asked_windows_path_keeps_its_backslashes(self):
        windows_path = self.root / "Interviews\\interview.wav"
        windows_path.touch()
        with patch("interview_local.cli.os", WINDOWS_OS):
            code, _, _ = run_cli("--no-llm", "--output", self.output, answers=(str(windows_path), "Anna", "Bernd"))
        self.assertEqual(code, 0)

    def test_asked_quoted_windows_path_keeps_its_drive_letter(self):
        relative_drive_path = "X:\\Interviews\\interview.wav"
        (self.root / relative_drive_path).touch()
        with patch("interview_local.cli.os", WINDOWS_OS), chdir(self.root):
            code, _, _ = run_cli("--no-llm", "--output", self.output,
                                 answers=(f'"{relative_drive_path}"', "Anna", "Bernd"))
        self.assertEqual(code, 0)

    def test_asked_windows_path_may_be_quoted(self):
        spaced = self.root / "my interview.wav"
        spaced.touch()
        with patch("interview_local.cli.os", WINDOWS_OS):
            code, _, _ = run_cli("--no-llm", "--output", self.output, answers=(f'"{spaced}"', "Anna", "Bernd"))
        self.assertEqual(code, 0)

    def test_without_llm_the_named_speaker_transcript_is_printed(self):
        code, printed, _ = run_cli(self.audio, "--no-llm", "--name-a", "Anna", "--output", self.output)
        drafted = (self.output / "03_readable.json").exists()
        self.assertEqual((code, "] Anna\nGuten Tag" in printed, drafted), (0, True, False))

    def test_given_audio_is_transcribed_without_asking_for_names(self):
        run_cli(self.audio, "--no-llm", "--output", self.output)
        self.assertEqual(self.saved("02_speakers.json")["speakers"], {"A": "Speaker A", "B": "Speaker B"})

    def test_transcription_drafts_a_readable_transcript(self):
        code, printed, _ = run_cli(self.audio, "--output", self.output)
        self.assertEqual((code, self.saved("03_readable.json")["stage"], "run --review on 03_readable.json" in printed),
                         (0, "readable", True))

    def test_asked_audio_path_may_be_quoted_and_names_are_asked(self):
        spaced = self.root / "my interview.wav"
        spaced.touch()
        answers = (f"'{spaced}'", "Anna", "Bernd")
        code, _, _ = run_cli("--no-llm", "--output", self.output, answers=answers)
        self.assertEqual((code, self.saved("02_speakers.json")["speakers"]), (0, {"A": "Anna", "B": "Bernd"}))


if __name__ == "__main__":
    unittest.main()
