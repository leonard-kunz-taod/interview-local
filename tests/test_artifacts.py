# German fixtures intentionally exercise non-ASCII speech and speaker names.
import json
import os
import tempfile
import unittest
from pathlib import Path

from interview_local.artifacts import (
    readable_markdown,
    readable_text,
    save_raw,
    save_readable,
    save_speakers_with_aliases,
)
from interview_local.core import Turn, Word, merge, render

PARAGRAPH = {
    "speaker": "A",
    "start": 0,
    "end": 2.5,
    "text": "Grüße aus Köln.",
    "uncertain": True,
    "overlap": True,
}
READABLE = {"stage": "readable", "speakers": {"A": "Änne"}, "paragraphs": [PARAGRAPH]}
REVIEWED = {
    "stage": "review",
    "speakers": {"A": "Änne"},
    "paragraphs": [
        dict(PARAGRAPH, uncertain=False, overlap=False, review_reasons=["Name?"])
    ],
}
NOTICE = "Machine-generated draft. Check uncertain passages against the audio."


def named_speaker_transcript():
    result = merge([Word(0, 1, "Grüße")], [Turn(0, 2, "x")])
    result["speakers"]["A"] = "Änne"
    return result


class RawArtifactTests(unittest.TestCase):
    def test_raw_text_lists_each_segment_with_its_time_range(self):
        raw = {"segments": [{"start": 0, "end": 1.25, "text": " Guten Tag. "}]}
        with tempfile.TemporaryDirectory() as directory:
            save_raw(raw, Path(directory))
            text = (Path(directory) / "01_raw.txt").read_text(encoding="utf-8")
        self.assertEqual(
            text,
            "RAW TRANSCRIPT — original speech recognition output\n\n"
            "[00:00:00.000 – 00:00:01.250]\nGuten Tag.\n",
        )

    def test_raw_json_preserves_the_recognition_output(self):
        raw = {"segments": [{"start": 0, "end": 1.25, "text": "Grüße"}]}
        with tempfile.TemporaryDirectory() as directory:
            save_raw(raw, Path(directory))
            saved = (Path(directory) / "01_raw.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(saved), raw)


class ReadableArtifactTests(unittest.TestCase):
    def test_plain_text_marks_uncertain_and_overlapping_paragraphs(self):
        self.assertEqual(
            readable_text(READABLE),
            f"Readable transcript — LLM draft\n\n{NOTICE}\n\n"
            "Änne · 00:00:00.000–00:00:02.500 · UNCERTAIN · OVERLAPPING SPEECH\n\n"
            "Grüße aus Köln.\n",
        )

    def test_markdown_puts_title_and_paragraph_headings_in_headings(self):
        self.assertEqual(
            readable_markdown(READABLE),
            f"# Readable transcript — LLM draft\n\n{NOTICE}\n\n"
            "### Änne · 00:00:00.000–00:00:02.500 · UNCERTAIN · OVERLAPPING SPEECH\n\n"
            "Grüße aus Köln.\n",
        )

    def test_review_text_lists_the_reviewer_reasons(self):
        self.assertEqual(
            readable_text(REVIEWED),
            f"Reviewed transcript\n\n{NOTICE}\n\n"
            "Änne · 00:00:00.000–00:00:02.500\n\nGrüße aus Köln.\n\nReview: Name?\n",
        )

    def test_readable_stage_is_saved_as_03_files(self):
        with tempfile.TemporaryDirectory() as directory:
            save_readable(READABLE, Path(directory))
            saved = sorted(os.listdir(directory))
        self.assertEqual(
            saved, ["03_readable.json", "03_readable.md", "03_readable.txt"]
        )

    def test_review_stage_is_saved_as_04_files(self):
        with tempfile.TemporaryDirectory() as directory:
            save_readable(REVIEWED, Path(directory))
            saved = sorted(os.listdir(directory))
        self.assertEqual(saved, ["04_review.json", "04_review.md", "04_review.txt"])

    def test_saved_markdown_is_the_rendered_markdown(self):
        with tempfile.TemporaryDirectory() as directory:
            save_readable(READABLE, Path(directory))
            markdown = (Path(directory) / "03_readable.md").read_text(encoding="utf-8")
        self.assertEqual(markdown, readable_markdown(READABLE))


class SpeakerAliasTests(unittest.TestCase):
    def test_final_speaker_stage_is_saved_with_its_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            save_speakers_with_aliases(named_speaker_transcript(), Path(directory))
            saved = sorted(os.listdir(directory))
        self.assertEqual(
            saved,
            [
                "02_speakers.json",
                "02_speakers.txt",
                "transcript.json",
                "transcript.txt",
            ],
        )

    def test_alias_json_preserves_names_and_unicode_words(self):
        result = named_speaker_transcript()
        with tempfile.TemporaryDirectory() as directory:
            save_speakers_with_aliases(result, Path(directory))
            saved = (Path(directory) / "transcript.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(saved), result)

    def test_alias_text_is_the_rendered_speaker_transcript(self):
        result = named_speaker_transcript()
        with tempfile.TemporaryDirectory() as directory:
            save_speakers_with_aliases(result, Path(directory))
            text = (Path(directory) / "transcript.txt").read_text(encoding="utf-8")
        self.assertEqual(text, render(result))


if __name__ == "__main__":
    unittest.main()
