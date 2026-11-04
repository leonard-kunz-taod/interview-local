# German fixtures intentionally exercise non-ASCII speech and speaker names.
import contextlib
import importlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from support.fake_ollama import FakeOllama, served_by
from support.fake_tk import FakeScreen
from support.fake_worker import SPOKEN_OUTPUTS, FakeWorker
from support.platform_encoding import windows_default_encoding

import interview_local
from interview_local.core import Turn, Word, merge
from interview_local.llm import enhance

PROCESS = "Process and save stages …"
COMPARE = "Compare readable draft …"
PREPARE = "Download / prepare Ministral"
CHOOSE = "Choose audio file …"
OFFLINE = "Offline speech models (download caches first)"
ALIGNMENT = "Additional German word alignment (optional)"
USE_LLM = "Create a readable draft with local Ministral"
LONG_MESSAGE = "x" * 600 + "end"
TIMESTAMP = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{6}"


class WindowTestCase(unittest.TestCase):
    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.screen = FakeScreen()
        stack.enter_context(patch.dict(sys.modules, self.screen.modules()))
        stack.callback(vars(interview_local).pop, "gui", None)
        sys.modules.pop("interview_local.gui", None)
        self.gui = importlib.import_module("interview_local.gui")
        self.service = stack.enter_context(served_by(self.local_service()))
        self.worker = FakeWorker(SPOKEN_OUTPUTS)
        stack.enter_context(patch("interview_local.pipeline.worker", self.worker))
        self.directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.audio = self.directory / "interview.wav"
        self.audio.touch()
        self.gui.launch()

    def local_service(self):
        return FakeOllama()

    def process_audio(self):
        """Run the full process for the test recording; return its output folder."""
        self.screen.entries()[0].set(str(self.audio))
        self.screen.chosen_directories.append(str(self.directory))
        self.screen.press(PROCESS)
        self.screen.settle()
        return next(self.directory.glob("interview_*"))

    def saved_speakers(self, output):
        return json.loads((output / "02_speakers.json").read_text("utf-8"))["speakers"]


class FileChoiceTests(WindowTestCase):
    def test_chosen_audio_file_fills_the_path(self):
        self.screen.chosen_files.append("/recordings/interview.wav")
        self.screen.press(CHOOSE)
        self.assertEqual(self.screen.entries()[0].get(), "/recordings/interview.wav")

    def test_cancelled_file_choice_keeps_the_path(self):
        self.screen.entries()[0].set("/recordings/earlier.wav")
        self.screen.chosen_files.append("")
        self.screen.press(CHOOSE)
        self.assertEqual(self.screen.entries()[0].get(), "/recordings/earlier.wav")


class ProcessingTests(WindowTestCase):
    def test_missing_audio_file_is_reported(self):
        self.screen.entries()[0].set(str(self.directory / "missing.wav"))
        self.screen.press(PROCESS)
        self.assertEqual(
            self.screen.shown,
            [("error", "File not found", "Choose an existing audio file.")],
        )

    def test_cancelled_folder_choice_starts_nothing(self):
        self.screen.entries()[0].set(str(self.audio))
        self.screen.chosen_directories.append("")
        self.screen.press(PROCESS)
        self.assertEqual((self.worker.stages, self.screen.busy()), ([], False))

    def test_processing_reports_where_the_stages_were_saved(self):
        output = self.process_audio()
        self.assertEqual(self.screen.status(), f"Done. Saved stages: {output}")

    def test_run_folder_is_named_with_its_start_time(self):
        output = self.process_audio()
        self.assertRegex(output.name, f"^interview_{TIMESTAMP}$")

    def test_processing_shows_every_saved_stage(self):
        output = self.process_audio()
        shown = {label: self.screen.view(label) for label in ("Raw", "Speakers")}
        self.assertEqual(
            shown,
            {
                "Raw": (output / "01_raw.txt").read_text(encoding="utf-8"),
                "Speakers": (output / "02_speakers.txt").read_text(encoding="utf-8"),
            },
        )

    def test_german_stages_are_shown_where_the_platform_default_is_not_utf8(self):
        with windows_default_encoding():
            self.process_audio()
        self.assertIn("\nGrüße\n", self.screen.view("Raw"))

    def test_processing_creates_a_readable_draft(self):
        self.process_audio()
        self.assertTrue(
            self.screen.view("Readable").startswith("Readable transcript — LLM draft")
        )

    def test_entered_names_are_saved_with_the_speakers(self):
        entries = self.screen.entries()
        entries[2].set(" Änne ")
        entries[3].set("")
        output = self.process_audio()
        self.assertEqual(self.saved_speakers(output), {"A": "Änne", "B": "Speaker B"})

    def test_blank_first_name_keeps_its_default(self):
        entries = self.screen.entries()
        entries[2].set(" ")
        entries[3].set("Bernd")
        output = self.process_audio()
        self.assertEqual(self.saved_speakers(output), {"A": "Speaker A", "B": "Bernd"})

    def test_untouched_names_are_saved_as_the_default_speakers(self):
        output = self.process_audio()
        self.assertEqual(
            self.saved_speakers(output), {"A": "Speaker A", "B": "Speaker B"}
        )

    def test_entered_token_reaches_the_speech_workers(self):
        self.screen.entries()[1].set(" hf_entered ")
        self.process_audio()
        self.assertEqual(self.worker.tokens, {"hf_entered"})

    def test_empty_token_runs_without_a_token(self):
        self.process_audio()
        self.assertEqual(self.worker.tokens, {None})

    def test_token_is_cleared_after_the_run(self):
        self.screen.entries()[1].set("hf_entered")
        self.process_audio()
        self.assertEqual(self.screen.entries()[1].get(), "")

    def test_offline_option_runs_every_worker_offline(self):
        self.screen.option(OFFLINE).set(True)
        self.process_audio()
        self.assertEqual(self.worker.offline, {True})

    def test_alignment_option_aligns_the_words(self):
        self.screen.option(ALIGNMENT).set(True)
        self.process_audio()
        self.assertIn("align", self.worker.stages)

    def test_readable_draft_is_skipped_without_the_llm(self):
        self.screen.option(USE_LLM).set(False)
        output = self.process_audio()
        self.assertEqual(
            (self.screen.view("Readable"), (output / "03_readable.json").exists()),
            ("", False),
        )

    def test_failed_processing_is_reported(self):
        with patch(
            "interview_local.pipeline.worker", side_effect=RuntimeError("Broken")
        ):
            self.process_audio()
        self.assertEqual(
            (self.screen.status(), self.screen.shown),
            (
                "Processing failed. Completed stages remain saved.",
                [("error", "Processing failed", "Broken")],
            ),
        )

    def test_second_run_starts_once_the_first_has_finished(self):
        self.process_audio()
        self.process_audio()
        self.assertEqual(len(list(self.directory.glob("interview_*"))), 2)

    def test_new_run_clears_the_previous_views(self):
        self.process_audio()
        self.screen.option(USE_LLM).set(False)
        self.process_audio()
        self.assertEqual(self.screen.view("Readable"), "")


class ComparisonTests(WindowTestCase):
    def saved_draft(self, word="Hallo"):
        speakers = merge([Word(0, 0.5, word, 0.9, "whisper")], [Turn(0, 1, "s")], [])
        draft = enhance(speakers, lambda _: None)
        path = self.directory / "03_readable.json"
        path.write_text(json.dumps(draft, ensure_ascii=False), encoding="utf-8")
        return path

    def compare(self, draft):
        self.screen.chosen_files.append(str(draft))
        self.screen.press(COMPARE)
        self.screen.settle()

    def test_comparison_shows_the_review(self):
        self.compare(self.saved_draft())
        self.assertTrue(self.screen.view("Review").startswith("Reviewed transcript"))

    def test_comparison_is_saved_next_to_the_draft(self):
        self.compare(self.saved_draft())
        review = next(self.directory.glob("review_*"))
        self.assertEqual(
            sorted(p.name for p in review.iterdir()),
            ["04_review.json", "04_review.md", "04_review.txt"],
        )

    def test_comparison_folder_is_named_with_its_start_time(self):
        self.compare(self.saved_draft())
        review = next(self.directory.glob("review_*"))
        self.assertRegex(review.name, f"^review_{TIMESTAMP}$")

    def test_german_draft_is_compared_where_the_platform_default_is_not_utf8(self):
        draft = self.saved_draft(word="Grüße")
        with windows_default_encoding():
            self.compare(draft)
        self.assertIn("\nGrüße\n", self.screen.view("Review"))

    def test_cancelled_comparison_starts_nothing(self):
        self.screen.chosen_files.append("")
        self.screen.press(COMPARE)
        self.assertEqual(
            (list(self.directory.glob("review_*")), self.screen.busy()), ([], False)
        )


class PreparationTests(WindowTestCase):
    def test_preparation_reports_the_prepared_model(self):
        self.screen.press(PREPARE)
        self.screen.settle()
        self.assertEqual(self.screen.status(), "Local model prepared.")

    def test_preparation_leaves_the_stage_views_empty(self):
        self.screen.press(PREPARE)
        self.screen.settle()
        self.assertEqual(self.screen.view("Raw"), "")


class RunningJobTests(WindowTestCase):
    """A model download that reports a long message, then stalls until released."""

    def local_service(self):
        self.release = threading.Event()
        updates = ({"status": LONG_MESSAGE}, {"status": "success"})
        return FakeOllama(pull_updates=updates, pull_gate=self.release)

    def setUp(self):
        super().setUp()
        self.addCleanup(self.finish)
        self.screen.press(PREPARE)

    def finish(self):
        self.release.set()
        self.screen.settle()

    def test_controls_are_disabled_while_a_job_runs(self):
        states = {
            b.options["state"]
            for b in map(self.screen.button, (PROCESS, COMPARE, PREPARE))
        }
        self.assertEqual(states, {"disabled"})

    def test_controls_are_enabled_again_after_the_job(self):
        self.finish()
        self.assertFalse(self.screen.busy())

    def test_progress_shows_the_last_500_characters(self):
        last_characters = LONG_MESSAGE[-500:]
        self.screen.tick_until(lambda: self.screen.status() == last_characters)
        self.assertEqual(self.screen.status(), last_characters)

    def test_running_job_never_keeps_the_app_from_exiting(self):
        self.screen.tick_until(lambda: self.screen.status() == LONG_MESSAGE[-500:])
        blocking = [
            t
            for t in threading.enumerate()
            if t is not threading.main_thread() and not t.daemon
        ]
        self.assertEqual(blocking, [])

    def test_second_job_is_ignored_while_one_runs(self):
        self.screen.chosen_files.append(str(self.directory / "03_readable.json"))
        self.screen.press(COMPARE)
        self.finish()
        self.assertEqual(self.screen.shown, [])

    def test_closing_while_a_job_runs_keeps_the_window_open(self):
        self.screen.window.handlers["WM_DELETE_WINDOW"]()
        self.assertEqual(
            (self.screen.shown[0][:2], self.screen.window.destroyed),
            (("info", "Processing in progress"), False),
        )


class CloseTests(WindowTestCase):
    def test_closing_an_idle_window_destroys_it(self):
        self.screen.window.handlers["WM_DELETE_WINDOW"]()
        self.assertTrue(self.screen.window.destroyed)


if __name__ == "__main__":
    unittest.main()
