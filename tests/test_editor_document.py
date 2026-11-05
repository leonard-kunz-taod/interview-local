import copy
import unittest
from dataclasses import replace
from pathlib import Path

from run_fixtures import (
    INTERVIEW_TURNS,
    TWO_PARAGRAPHS,
    interview_words,
    readable_transcript,
    reassign,
)

from interview_local.artifacts import corrected_text
from interview_local.core import Turn
from interview_local.corrections import (
    edit_paragraph_text,
    load_corrected_paragraphs,
    set_paragraph_checked,
)
from interview_local.editor_document import (
    LONG_SILENCE_SECONDS,
    LOW_ASR_PROBABILITY,
    build_editor_document,
)

NAMES = {"A": "Anna", "B": "Bernd"}
NO_CORRECTIONS = Path("/nonexistent-run")


def document_of(readable, paragraphs=None):
    paragraphs = paragraphs or load_corrected_paragraphs(NO_CORRECTIONS, readable)
    return build_editor_document(readable, paragraphs, NAMES)


class DocumentTests(unittest.TestCase):
    def test_document_carries_the_speaker_names_and_the_word_marking_thresholds(self):
        document = document_of(readable_transcript(TWO_PARAGRAPHS))
        self.assertEqual(
            {key: value for key, value in document.items() if key != "paragraphs"},
            {
                "speakers": NAMES,
                "low_asr_probability": LOW_ASR_PROBABILITY,
                "long_silence_seconds": LONG_SILENCE_SECONDS,
            },
        )


class ParagraphTests(unittest.TestCase):
    def test_paragraph_shows_its_word_range_speaker_source_time_range_text_and_review_marks(
        self,
    ):
        paragraph = dict(
            document_of(readable_transcript(TWO_PARAGRAPHS))["paragraphs"][1]
        )
        del paragraph["words"]
        self.assertEqual(
            paragraph,
            {
                "first_word": 4,
                "last_word": 7,
                "speaker": "B",
                "speaker_name": "Bernd",
                "start": 5,
                "end": 7,
                "text": "Danke, gut. Und selbst?",
                "text_edited": False,
                "speaker_edited": False,
                "checked": False,
                "speaker_uncertainty": [],
            },
        )

    def test_checked_paragraph_carries_its_review_status(self):
        readable = readable_transcript(TWO_PARAGRAPHS)
        paragraphs = set_paragraph_checked(
            load_corrected_paragraphs(NO_CORRECTIONS, readable), 4, True
        )
        self.assertEqual(
            [p["checked"] for p in document_of(readable, paragraphs)["paragraphs"]],
            [False, True],
        )

    def test_clean_paragraphs_have_no_speaker_uncertainty(self):
        document = document_of(readable_transcript(TWO_PARAGRAPHS))
        self.assertEqual(
            [p["speaker_uncertainty"] for p in document["paragraphs"]], [[], []]
        )

    def test_simultaneous_speech_is_a_speaker_uncertainty(self):
        regular = INTERVIEW_TURNS + [Turn(5.4, 6.1, "first")]
        document = document_of(
            readable_transcript(TWO_PARAGRAPHS, regular_turns=regular)
        )
        self.assertEqual(
            document["paragraphs"][1]["speaker_uncertainty"], ["overlapping_speech"]
        )

    def test_unassigned_paragraph_is_a_speaker_uncertainty(self):
        readable = readable_transcript(
            [(0, 3, None, "Guten Tag, wie geht's?"), TWO_PARAGRAPHS[1]]
        )
        paragraph = document_of(readable)["paragraphs"][0]
        self.assertIn("unassigned", paragraph["speaker_uncertainty"])

    def test_unassigned_paragraph_is_named_unassigned(self):
        readable = readable_transcript(
            [(0, 3, None, "Guten Tag, wie geht's?"), TWO_PARAGRAPHS[1]]
        )
        self.assertEqual(
            document_of(readable)["paragraphs"][0]["speaker_name"], "Unassigned"
        )

    def test_speaker_changed_by_llm_is_a_speaker_uncertainty(self):
        readable = readable_transcript(
            [TWO_PARAGRAPHS[0], (4, 7, "A", "Danke, gut. Und selbst?")]
        )
        self.assertEqual(
            document_of(readable)["paragraphs"][1]["speaker_uncertainty"],
            ["speaker_changed_by_llm"],
        )

    def test_manual_reassignment_clears_speaker_uncertainty(self):
        regular = INTERVIEW_TURNS + [Turn(5.4, 6.1, "first")]
        readable = readable_transcript(
            [TWO_PARAGRAPHS[0], (4, 7, "A", "Danke, gut. Und selbst?")],
            regular_turns=regular,
        )
        paragraphs = reassign(
            readable, load_corrected_paragraphs(NO_CORRECTIONS, readable), 4, 7, "B"
        )
        paragraph = document_of(readable, paragraphs)["paragraphs"][1]
        self.assertEqual(
            (paragraph["speaker"], paragraph["speaker_uncertainty"]), ("B", [])
        )

    def test_document_reflects_swapped_speaker_names(self):
        readable = readable_transcript(TWO_PARAGRAPHS)
        document = build_editor_document(
            readable,
            load_corrected_paragraphs(NO_CORRECTIONS, readable),
            {"A": "Bernd", "B": "Anna"},
        )
        self.assertEqual(
            [p["speaker_name"] for p in document["paragraphs"]], ["Bernd", "Anna"]
        )


def single_paragraph_draft(text):
    """One LLM paragraph over all eight words; small wording changes stay within the editor's limits."""
    return readable_transcript([(0, 7, "A", text)])


class WordTests(unittest.TestCase):
    def test_llm_changed_word_carries_its_original(self):
        readable = single_paragraph_draft(
            "Guten Morgen, wie geht's? Danke, gut. Und selbst?"
        )
        word = document_of(readable)["paragraphs"][0]["words"][1]
        self.assertEqual(
            word,
            {
                "text": "Morgen,",
                "source_word_ids": [1],
                "seek_time": 0.5,
                "change": "replaced",
                "original_text": "Tag",
                "removed_before": "",
                "low_asr_confidence": False,
                "after_long_silence": False,
            },
        )

    def test_user_edit_is_compared_with_speech_recognition(self):
        readable = single_paragraph_draft(
            "Guten Morgen, wie geht's? Danke, gut. Und selbst?"
        )
        paragraphs = edit_paragraph_text(
            load_corrected_paragraphs(NO_CORRECTIONS, readable),
            0,
            "Guten Tag, wie geht's? Danke, gut. Und selbst?",
        )
        words = document_of(readable, paragraphs)["paragraphs"][0]["words"]
        self.assertEqual([w["change"] for w in words], ["unchanged"] * 8)

    def test_word_after_a_pause_seeks_to_its_own_source_start(self):
        source_words = interview_words()
        pause = 0.25
        source_words[5] = replace(source_words[5], start=source_words[5].start + pause)
        words = document_of(readable_transcript(TWO_PARAGRAPHS, words=source_words))[
            "paragraphs"
        ][1]["words"]
        self.assertEqual([w["seek_time"] for w in words], [5, 5.5 + pause, 6, 6.5])

    def test_inserted_word_seeks_to_the_end_of_the_previous_source_word(self):
        readable = single_paragraph_draft(
            "Ach, guten lieben Tag, wie geht's? Danke, gut. Und selbst?"
        )
        words = document_of(readable)["paragraphs"][0]["words"]
        self.assertEqual(
            [(w["text"], w["seek_time"]) for w in words[:4]],
            [("Ach,", 0), ("guten", 0), ("lieben", 0.5), ("Tag,", 0.5)],
        )

    def test_low_confidence_starts_below_the_threshold(self):
        just_below = LOW_ASR_PROBABILITY - 1e-6
        probabilities = [LOW_ASR_PROBABILITY, just_below, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        readable = readable_transcript(
            TWO_PARAGRAPHS, words=interview_words(probabilities)
        )
        words = document_of(readable)["paragraphs"][0]["words"]
        self.assertEqual(
            [w["low_asr_confidence"] for w in words], [False, True, False, False]
        )

    def test_run_without_word_probabilities_cannot_be_opened(self):
        readable = copy.deepcopy(readable_transcript(TWO_PARAGRAPHS))
        for word in readable["source_transcript"]["words"]:
            del word["probability"]
        with self.assertRaises(ValueError):
            document_of(readable)


def silence_before_second_paragraph(seconds):
    """Interview words whose second paragraph (words 4-7) starts ``seconds`` after word 3 ends."""
    source_words = interview_words()
    shift = seconds - (source_words[4].start - source_words[3].end)
    source_words[4:] = [
        replace(w, start=w.start + shift, end=w.end + shift) for w in source_words[4:]
    ]
    return readable_transcript(TWO_PARAGRAPHS, words=source_words)


class LongSilenceTests(unittest.TestCase):
    def test_word_after_exactly_the_threshold_of_silence_is_marked(self):
        readable = silence_before_second_paragraph(LONG_SILENCE_SECONDS)
        words = document_of(readable)["paragraphs"][1]["words"]
        self.assertEqual(
            [w["after_long_silence"] for w in words], [True, False, False, False]
        )

    def test_word_after_slightly_shorter_silence_is_not_marked(self):
        readable = silence_before_second_paragraph(LONG_SILENCE_SECONDS - 0.01)
        words = document_of(readable)["paragraphs"][1]["words"]
        self.assertEqual([w["after_long_silence"] for w in words], [False] * 4)

    def test_inserted_word_after_a_long_silence_is_not_marked(self):
        readable = silence_before_second_paragraph(LONG_SILENCE_SECONDS)
        paragraphs = edit_paragraph_text(
            load_corrected_paragraphs(NO_CORRECTIONS, readable),
            4,
            "Also Danke, gut. Und selbst?",
        )
        words = document_of(readable, paragraphs)["paragraphs"][1]["words"]
        self.assertEqual(
            [(w["text"], w["after_long_silence"]) for w in words[:2]],
            [("Also", False), ("Danke,", True)],
        )

    def test_first_word_is_measured_from_the_audio_start(self):
        source_words = [
            replace(
                w,
                start=w.start + LONG_SILENCE_SECONDS,
                end=w.end + LONG_SILENCE_SECONDS,
            )
            for w in interview_words()
        ]
        readable = readable_transcript(TWO_PARAGRAPHS, words=source_words)
        words = document_of(readable)["paragraphs"][0]["words"]
        self.assertEqual(
            [w["after_long_silence"] for w in words], [True, False, False, False]
        )


def corrected_uncertain_document():
    """A document with an uncertain-speaker paragraph whose text was edited by hand."""
    regular = INTERVIEW_TURNS + [Turn(5.4, 6.1, "first")]
    readable = readable_transcript(TWO_PARAGRAPHS, regular_turns=regular)
    paragraphs = edit_paragraph_text(
        load_corrected_paragraphs(NO_CORRECTIONS, readable), 4, "Danke, sehr gut."
    )
    return document_of(readable, paragraphs)


class CorrectedExportTests(unittest.TestCase):
    def test_markdown_export_lists_speaker_time_range_and_corrected_text_without_markers(
        self,
    ):
        expected = (
            "# Corrected transcript\n\n"
            "### Anna · 00:00:00.000–00:00:02.000\n\nGuten Tag, wie geht's?\n\n"
            "### Bernd · 00:00:05.000–00:00:07.000\n\nDanke, sehr gut.\n"
        )
        self.assertEqual(
            corrected_text(corrected_uncertain_document()["paragraphs"], markdown=True),
            expected,
        )

    def test_text_export_lists_speaker_time_range_and_corrected_text_without_markers(
        self,
    ):
        expected = (
            "Corrected transcript\n\n"
            "Anna · 00:00:00.000–00:00:02.000\n\nGuten Tag, wie geht's?\n\n"
            "Bernd · 00:00:05.000–00:00:07.000\n\nDanke, sehr gut.\n"
        )
        self.assertEqual(
            corrected_text(
                corrected_uncertain_document()["paragraphs"], markdown=False
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
