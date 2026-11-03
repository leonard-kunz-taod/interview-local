import json
from pathlib import Path
import tempfile
import unittest

from interview_local.corrections import (CorrectedParagraph, edit_paragraph_text, load_corrected_paragraphs,
                                         load_speaker_names, save_corrections)
from run_fixtures import TWO_PARAGRAPHS, readable_transcript, reassign


def summary(paragraphs):
    return [(p.first_word, p.last_word, p.speaker, p.text, p.speaker_edited) for p in paragraphs]


class LoadAndSaveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.directory.name)
        self.readable = readable_transcript(TWO_PARAGRAPHS)

    def tearDown(self):
        self.directory.cleanup()

    def test_without_corrections_the_readable_paragraphs_are_used(self):
        paragraphs = load_corrected_paragraphs(self.run_dir, self.readable)
        self.assertEqual(summary(paragraphs), [(0, 3, "A", "Guten Tag, wie geht's?", False),
                                               (4, 7, "B", "Danke, gut. Und selbst?", False)])

    def test_without_corrections_the_readable_speaker_names_are_used(self):
        self.assertEqual(load_speaker_names(self.run_dir, self.readable), {"A": "Anna", "B": "Bernd"})

    def test_saved_paragraphs_load_again(self):
        paragraphs = edit_paragraph_text(load_corrected_paragraphs(self.run_dir, self.readable), 4, "Danke, sehr gut.")
        save_corrections(self.run_dir, self.readable, {"A": "Anna", "B": "Bernd"}, paragraphs)
        self.assertEqual(load_corrected_paragraphs(self.run_dir, self.readable), paragraphs)

    def test_saved_speaker_names_load_again(self):
        save_corrections(self.run_dir, self.readable, {"A": "Bernd", "B": "Anna"},
                         load_corrected_paragraphs(self.run_dir, self.readable))
        self.assertEqual(load_speaker_names(self.run_dir, self.readable), {"A": "Bernd", "B": "Anna"})

    def test_saving_an_empty_speaker_name_is_rejected(self):
        with self.assertRaises(ValueError):
            save_corrections(self.run_dir, self.readable, {"A": "Anna", "B": " "},
                             load_corrected_paragraphs(self.run_dir, self.readable))

    def test_corrections_file_of_another_schema_is_rejected(self):
        (self.run_dir / "corrections.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_corrected_paragraphs(self.run_dir, self.readable)

    def test_corrections_of_a_changed_readable_transcript_are_rejected(self):
        save_corrections(self.run_dir, self.readable, {"A": "Anna", "B": "Bernd"},
                         load_corrected_paragraphs(self.run_dir, self.readable))
        changed = readable_transcript([(0, 3, "A", "Guten Tag. Wie geht's?"), TWO_PARAGRAPHS[1]])
        with self.assertRaises(ValueError):
            load_corrected_paragraphs(self.run_dir, changed)

    def test_corrections_with_broken_coverage_are_rejected(self):
        save_corrections(self.run_dir, self.readable, {"A": "Anna", "B": "Bernd"},
                         load_corrected_paragraphs(self.run_dir, self.readable))
        path = self.run_dir / "corrections.json"
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["paragraphs"][1]["first_word"] = 5
        path.write_text(json.dumps(stored), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_corrected_paragraphs(self.run_dir, self.readable)

    def test_saving_a_gap_is_rejected(self):
        paragraphs = load_corrected_paragraphs(self.run_dir, self.readable)[:1]
        with self.assertRaises(ValueError):
            save_corrections(self.run_dir, self.readable, {"A": "Anna", "B": "Bernd"}, paragraphs)
        self.assertFalse((self.run_dir / "corrections.json").exists())

    def test_saving_leaves_the_readable_transcript_unchanged(self):
        before = json.dumps(self.readable, sort_keys=True)
        save_corrections(self.run_dir, self.readable, {"A": "Anna", "B": "Bernd"},
                         load_corrected_paragraphs(self.run_dir, self.readable))
        self.assertEqual(json.dumps(self.readable, sort_keys=True), before)

    def test_saving_writes_only_corrections_json(self):
        save_corrections(self.run_dir, self.readable, {"A": "Anna", "B": "Bernd"},
                         load_corrected_paragraphs(self.run_dir, self.readable))
        self.assertEqual([p.name for p in self.run_dir.iterdir()], ["corrections.json"])


class EditTextTests(unittest.TestCase):
    def setUp(self):
        self.paragraphs = [CorrectedParagraph(0, 3, "A", "Guten Tag", False, False),
                           CorrectedParagraph(4, 7, "B", "Danke", False, False)]

    def test_edit_replaces_only_that_text_and_marks_it_edited(self):
        edited = edit_paragraph_text(self.paragraphs, 4, " Danke schön ")
        self.assertEqual(edited, [self.paragraphs[0], CorrectedParagraph(4, 7, "B", "Danke schön", True, False)])

    def test_empty_text_is_rejected(self):
        for text in ("", "   ", " – ?"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                edit_paragraph_text(self.paragraphs, 0, text)

    def test_unknown_paragraph_is_rejected(self):
        with self.assertRaises(ValueError):
            edit_paragraph_text(self.paragraphs, 2, "Hallo")


class ReassignTests(unittest.TestCase):
    def setUp(self):
        self.readable = readable_transcript(TWO_PARAGRAPHS)
        self.paragraphs = load_corrected_paragraphs(Path("/nonexistent-run"), self.readable)

    def test_middle_selection_splits_into_three_paragraphs(self):
        result = reassign(self.readable, self.paragraphs, 5, 6, "A")
        self.assertEqual(summary(result)[1:], [(4, 4, "B", "Danke,", False), (5, 6, "A", "gut. Und", True),
                                               (7, 7, "B", "selbst?", False)])

    def test_whole_paragraph_changes_speaker_without_split(self):
        result = reassign(self.readable, self.paragraphs, 0, 3, "B")
        self.assertEqual(summary(result), [(0, 3, "B", "Guten Tag, wie geht's?", True),
                                           (4, 7, "B", "Danke, gut. Und selbst?", False)])

    def test_selection_across_paragraphs_is_rejected(self):
        with self.assertRaises(ValueError):
            reassign(self.readable, self.paragraphs, 3, 4, "B")

    def test_unknown_speaker_is_rejected(self):
        with self.assertRaises(ValueError):
            reassign(self.readable, self.paragraphs, 0, 1, "C")

    def test_boundary_inside_a_display_word_widens_to_the_whole_word(self):
        paragraphs = edit_paragraph_text(self.paragraphs, 0, "Gutentag, wie geht's?")
        result = reassign(self.readable, paragraphs, 1, 2, "B")
        self.assertEqual(summary(result)[:2], [(0, 2, "B", "Gutentag, wie", True), (3, 3, "A", "geht's?", False)])

    def test_inserted_word_goes_with_its_preceding_source_word(self):
        paragraphs = edit_paragraph_text(self.paragraphs, 4, "Danke, sehr gut. Und selbst?")
        result = reassign(self.readable, paragraphs, 4, 4, "A")
        self.assertEqual(summary(result)[1:3], [(4, 4, "A", "Danke, sehr", True),
                                                (5, 7, "B", "gut. Und selbst?", False)])

    def test_range_without_remaining_text_joins_the_selection(self):
        paragraphs = edit_paragraph_text(self.paragraphs, 4, "gut. Und selbst?")
        result = reassign(self.readable, paragraphs, 5, 5, "A")
        self.assertEqual(summary(result)[1:], [(4, 5, "A", "gut.", True), (6, 7, "B", "Und selbst?", False)])


if __name__ == "__main__":
    unittest.main()
