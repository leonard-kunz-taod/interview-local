# German fixtures intentionally exercise non-ASCII speech and speaker names.
import json
import tempfile
import unittest
from pathlib import Path

from interview_local.evaluate import distance, normalize, score

REFERENCE = (
    "Beg\tEnd\tSpeaker\tText\n"
    "10.0\t11.0\tA\tGuten Tag {laugh}\n"
    "11.0\t12.0\tB\tHallo Welt\n"
)
WORDS = [
    {"start": 0, "end": 0.5, "text": "Guten", "speaker": "A"},
    {"start": 0.5, "end": 1, "text": "Tag", "speaker": "A"},
    {"start": 1, "end": 1.5, "text": "Hallo", "speaker": "B"},
    {"start": 2.5, "end": 3, "text": "Danach", "speaker": "B"},
]
TURNS = [
    {"start": 0, "end": 1, "speaker": "A"},
    {"start": 1, "end": 2, "speaker": "B"},
]


def scored(hypothesis, start=0, end=2):
    with tempfile.TemporaryDirectory() as directory:
        reference_file = Path(directory) / "reference.tsv"
        hypothesis_file = Path(directory) / "hypothesis.json"
        reference_file.write_text(REFERENCE, encoding="utf-8")
        hypothesis_file.write_text(json.dumps(hypothesis), encoding="utf-8")
        return score(reference_file, hypothesis_file, 10, start, end)


def word_error_summary(report):
    keys = ("reference_words", "hypothesis_words", "word_errors", "wer")
    return {key: report[key] for key in keys}


EXPECTED_WER = {
    "reference_words": 4,
    "hypothesis_words": 3,
    "word_errors": 1,
    "wer": 0.25,
}


class NormalizationTests(unittest.TestCase):
    def test_noise_labels_and_punctuation_are_removed(self):
        self.assertEqual(normalize("Grüße, {laugh} Äpfel!"), ["grüße", "äpfel"])


class DistanceTests(unittest.TestCase):
    def test_edit_distance_counts_insertions_deletions_and_substitutions(self):
        cases = [
            (["a", "b", "c"], ["a", "d", "c", "e"]),
            ([], ["a"]),
            (["a"], []),
            (["a"], ["a"]),
        ]
        self.assertEqual([distance(r, h) for r, h in cases], [2, 1, 1, 0])


class ScoreTests(unittest.TestCase):
    def test_word_error_rate_covers_words_inside_the_window(self):
        report = scored({"words": WORDS})
        self.assertEqual(word_error_summary(report), EXPECTED_WER)

    def test_words_are_read_from_segments_without_a_word_list(self):
        report = scored({"segments": [{"words": WORDS}]})
        self.assertEqual(word_error_summary(report), EXPECTED_WER)

    def test_window_without_reference_words_is_rejected(self):
        with self.assertRaises(ValueError):
            scored({"words": WORDS}, start=50, end=60)

    def test_two_speaker_hypothesis_gets_a_concatenated_permutation_wer(self):
        report = scored({"words": WORDS, "diarization": TURNS})
        self.assertEqual(report["cpwer"], 0.25)


if __name__ == "__main__":
    unittest.main()
