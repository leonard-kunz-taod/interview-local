# German fixtures intentionally exercise non-ASCII speech and speaker names.
import json
import os
import tempfile
import unittest
from pathlib import Path

from support.platform_encoding import windows_default_encoding

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
ALL_ARTIFACTS = (
    "01_raw.json",
    "01_raw.txt",
    "02_speakers.json",
    "02_speakers.txt",
    "transcript.json",
    "transcript.txt",
    "03_readable.json",
    "03_readable.md",
    "03_readable.txt",
    "04_review.json",
    "04_review.md",
    "04_review.txt",
)


def named_speaker_transcript():
    result = merge([Word(0, 1, "Grüße")], [Turn(0, 2, "x")], [])
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

    def test_json_is_indented_and_keeps_german_characters_readable(self):
        raw = {"segments": [{"start": 0, "end": 1, "text": "Grüße"}]}
        with tempfile.TemporaryDirectory() as directory:
            save_raw(raw, Path(directory))
            saved = (Path(directory) / "01_raw.json").read_text(encoding="utf-8")
        self.assertEqual(
            saved,
            '{\n  "segments": [\n    {\n      "start": 0,\n      "end": 1,\n'
            '      "text": "Grüße"\n    }\n  ]\n}',
        )

    def test_stages_are_saved_into_a_new_nested_directory(self):
        raw = {"segments": []}
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "interviews" / "first"
            save_raw(raw, nested)
            saved = sorted(os.listdir(nested))
        self.assertEqual(saved, ["01_raw.json", "01_raw.txt"])


class PlatformEncodingTests(unittest.TestCase):
    def test_every_artifact_is_utf8_where_the_platform_default_is_not(self):
        raw = {"segments": [{"start": 0, "end": 1, "text": "Grüße"}]}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            review = Path(directory) / "review"
            with windows_default_encoding():
                save_raw(raw, run)
                save_speakers_with_aliases(named_speaker_transcript(), run)
                save_readable(READABLE, run)
                save_readable(REVIEWED, review)
            decoded = {
                path.name: "Grüße" in path.read_bytes().decode("utf-8")
                for path in [*run.iterdir(), *review.iterdir()]
            }
        self.assertEqual(decoded, dict.fromkeys(ALL_ARTIFACTS, True))


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

    def test_saved_readable_json_holds_the_readable_document(self):
        with tempfile.TemporaryDirectory() as directory:
            save_readable(READABLE, Path(directory))
            saved = (Path(directory) / "03_readable.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(saved), READABLE)

    def test_review_reasons_are_separated_by_semicolons(self):
        paragraph = dict(REVIEWED["paragraphs"][0], review_reasons=["Name?", "Zahl?"])
        text = readable_text(dict(REVIEWED, paragraphs=[paragraph]))
        self.assertIn("\nReview: Name?; Zahl?\n", text)

    def test_paragraph_without_speaker_is_headed_unassigned(self):
        paragraph = dict(PARAGRAPH, speaker=None, uncertain=False, overlap=False)
        text = readable_text(dict(READABLE, paragraphs=[paragraph]))
        self.assertIn("\nUnassigned · 00:00:00.000–00:00:02.500\n", text)


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
