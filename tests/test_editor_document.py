import copy
from pathlib import Path
import unittest

from interview_local.artifacts import corrected_text
from interview_local.core import Turn
from interview_local.corrections import (edit_paragraph_text, load_corrected_paragraphs, readable_source_words,
                                         reassign_words)
from interview_local.editor_document import LOW_ASR_PROBABILITY, build_editor_document
from run_fixtures import INTERVIEW_TURNS, TWO_PARAGRAPHS, interview_words, readable_transcript

NAMES = {"A": "Anna", "B": "Bernd"}
NO_CORRECTIONS = Path("/nonexistent-run")


def document_of(readable, paragraphs=None):
    paragraphs = paragraphs or load_corrected_paragraphs(NO_CORRECTIONS, readable)
    return build_editor_document(readable, paragraphs, NAMES)


def reassign(readable, paragraphs, first_word, last_word, speaker):
    return reassign_words(paragraphs, readable_source_words(readable), set(readable["speakers"]),
                          first_word, last_word, speaker)


class ParagraphTests(unittest.TestCase):
    def test_paragraph_shows_speaker_name_and_source_time_range(self):
        paragraph = document_of(readable_transcript(TWO_PARAGRAPHS))["paragraphs"][1]
        self.assertEqual((paragraph["speaker_name"], paragraph["start"], paragraph["end"]), ("Bernd", 5, 7))
        self.assertEqual(paragraph["text"], "Danke, gut. Und selbst?")

    def test_clean_paragraphs_have_no_speaker_uncertainty(self):
        document = document_of(readable_transcript(TWO_PARAGRAPHS))
        self.assertEqual([p["speaker_uncertainty"] for p in document["paragraphs"]], [[], []])

    def test_simultaneous_speech_is_a_speaker_uncertainty(self):
        regular = INTERVIEW_TURNS + [Turn(5.4, 6.1, "first")]
        document = document_of(readable_transcript(TWO_PARAGRAPHS, regular_turns=regular))
        self.assertEqual(document["paragraphs"][1]["speaker_uncertainty"], ["overlapping_speech"])

    def test_unassigned_paragraph_is_a_speaker_uncertainty(self):
        readable = readable_transcript([(0, 3, None, "Guten Tag, wie geht's?"), TWO_PARAGRAPHS[1]])
        paragraph = document_of(readable)["paragraphs"][0]
        self.assertEqual(paragraph["speaker_name"], "Unassigned")
        self.assertIn("unassigned", paragraph["speaker_uncertainty"])

    def test_speaker_changed_by_llm_is_a_speaker_uncertainty(self):
        readable = readable_transcript([TWO_PARAGRAPHS[0], (4, 7, "A", "Danke, gut. Und selbst?")])
        self.assertEqual(document_of(readable)["paragraphs"][1]["speaker_uncertainty"], ["speaker_changed_by_llm"])

    def test_manual_reassignment_clears_speaker_uncertainty(self):
        regular = INTERVIEW_TURNS + [Turn(5.4, 6.1, "first")]
        readable = readable_transcript([TWO_PARAGRAPHS[0], (4, 7, "A", "Danke, gut. Und selbst?")],
                                       regular_turns=regular)
        paragraphs = reassign(readable, load_corrected_paragraphs(NO_CORRECTIONS, readable), 4, 7, "B")
        paragraph = document_of(readable, paragraphs)["paragraphs"][1]
        self.assertEqual((paragraph["speaker"], paragraph["speaker_uncertainty"]), ("B", []))

    def test_document_reflects_swapped_speaker_names(self):
        readable = readable_transcript(TWO_PARAGRAPHS)
        document = build_editor_document(readable, load_corrected_paragraphs(NO_CORRECTIONS, readable),
                                         {"A": "Bernd", "B": "Anna"})
        self.assertEqual([p["speaker_name"] for p in document["paragraphs"]], ["Bernd", "Anna"])


def single_paragraph_draft(text):
    """One LLM paragraph over all eight words; small wording changes stay within the editor's limits."""
    return readable_transcript([(0, 7, "A", text)])


class WordTests(unittest.TestCase):
    def test_llm_changed_word_carries_its_original(self):
        readable = single_paragraph_draft("Guten Morgen, wie geht's? Danke, gut. Und selbst?")
        word = document_of(readable)["paragraphs"][0]["words"][1]
        self.assertEqual((word["text"], word["change"], word["original_text"]), ("Morgen,", "replaced", "Tag"))

    def test_user_edit_is_compared_with_speech_recognition(self):
        readable = single_paragraph_draft("Guten Morgen, wie geht's? Danke, gut. Und selbst?")
        paragraphs = edit_paragraph_text(load_corrected_paragraphs(NO_CORRECTIONS, readable), 0,
                                         "Guten Tag, wie geht's? Danke, gut. Und selbst?")
        words = document_of(readable, paragraphs)["paragraphs"][0]["words"]
        self.assertEqual([w["change"] for w in words], ["unchanged"] * 8)

    def test_word_seeks_to_its_source_start(self):
        words = document_of(readable_transcript(TWO_PARAGRAPHS))["paragraphs"][1]["words"]
        self.assertEqual([w["seek_time"] for w in words], [5, 5.5, 6, 6.5])

    def test_inserted_word_seeks_to_the_end_of_the_previous_source_word(self):
        readable = single_paragraph_draft("Ach, guten lieben Tag, wie geht's? Danke, gut. Und selbst?")
        words = document_of(readable)["paragraphs"][0]["words"]
        self.assertEqual([(w["text"], w["seek_time"]) for w in words[:4]],
                         [("Ach,", 0), ("guten", 0), ("lieben", .5), ("Tag,", .5)])

    def test_low_confidence_starts_below_the_threshold(self):
        just_below = LOW_ASR_PROBABILITY - 1e-6
        probabilities = [LOW_ASR_PROBABILITY, just_below, .9, .9, .9, .9, .9, .9]
        readable = readable_transcript(TWO_PARAGRAPHS, words=interview_words(probabilities))
        words = document_of(readable)["paragraphs"][0]["words"]
        self.assertEqual([w["low_asr_confidence"] for w in words], [False, True, False, False])

    def test_run_without_word_probabilities_cannot_be_opened(self):
        readable = copy.deepcopy(readable_transcript(TWO_PARAGRAPHS))
        for word in readable["source_transcript"]["words"]:
            del word["probability"]
        with self.assertRaises(ValueError):
            document_of(readable)


class CorrectedExportTests(unittest.TestCase):
    def test_export_lists_speaker_time_range_and_text_without_markers(self):
        regular = INTERVIEW_TURNS + [Turn(5.4, 6.1, "first")]
        readable = readable_transcript(TWO_PARAGRAPHS, regular_turns=regular)
        paragraphs = edit_paragraph_text(load_corrected_paragraphs(NO_CORRECTIONS, readable), 4, "Danke, sehr gut.")
        text = corrected_text(document_of(readable, paragraphs), markdown=True)
        self.assertEqual(text.splitlines()[:7], ["# Corrected transcript", "",
                                                 "### Anna · 00:00:00.000–00:00:02.000", "",
                                                 "Guten Tag, wie geht's?", "",
                                                 "### Bernd · 00:00:05.000–00:00:07.000"])
        self.assertIn("Danke, sehr gut.", text)
        self.assertNotIn("UNCERTAIN", text.upper())
        self.assertNotIn("#", corrected_text(document_of(readable, paragraphs), markdown=False))


if __name__ == "__main__":
    unittest.main()
