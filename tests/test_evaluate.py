# German fixtures intentionally exercise non-ASCII speech and speaker names.
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interview_local.evaluate import distance, main, normalize, score

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

    def test_swapped_speaker_labels_do_not_count_as_errors(self):
        swapped = [dict(w, speaker={"A": "B", "B": "A"}[w["speaker"]]) for w in WORDS]
        report = scored({"words": swapped, "diarization": TURNS})
        self.assertEqual(report["cpwer"], 0.25)

    def test_unassigned_hypothesis_words_count_as_errors_in_cpwer(self):
        unassigned = {"start": 1.5, "end": 2, "text": "Welt", "speaker": None}
        report = scored({"words": [*WORDS, unassigned], "diarization": TURNS})
        self.assertEqual(report["cpwer"], 0.5)

    def test_one_speaker_hypothesis_gets_no_cpwer(self):
        words = [dict(w, speaker="A") for w in WORDS]
        report = scored({"words": words, "diarization": TURNS})
        self.assertNotIn("cpwer", report)

    def test_hypothesis_without_diarization_gets_no_speaker_metrics(self):
        report = scored({"words": WORDS})
        self.assertFalse({"der", "der_conditions", "cpwer"} & report.keys())

    def test_matching_diarization_has_no_diarization_error(self):
        report = scored({"words": WORDS, "diarization": TURNS})
        self.assertEqual(report["der"]["diarization error rate"], 0.0)

    def test_turns_outside_the_window_do_not_count_toward_der(self):
        outside = {"start": 5, "end": 6, "speaker": "A"}
        report = scored({"words": WORDS, "diarization": [*TURNS, outside]})
        self.assertEqual(report["der"]["diarization error rate"], 0.0)

    def test_report_states_the_evaluated_window_and_offset(self):
        report = scored({"words": WORDS})
        self.assertEqual(
            [report[key] for key in KEYS_OF_WINDOW],
            [10, 0, 2],
        )


KEYS_OF_WINDOW = ("reference_offset_seconds", "evaluated_start", "evaluated_end")


class CommandLineTests(unittest.TestCase):
    def evaluate(self, *extra):
        """Printed report and the files left in the working directory."""
        with tempfile.TemporaryDirectory() as directory:
            reference_file = Path(directory) / "reference.tsv"
            hypothesis_file = Path(directory) / "hypothesis.json"
            reference_file.write_text(REFERENCE, encoding="utf-8")
            hypothesis_file.write_text(json.dumps({"words": WORDS}), encoding="utf-8")
            arguments = [
                reference_file,
                hypothesis_file,
                "--offset",
                "10",
                "--end",
                "2",
            ]
            printed = io.StringIO()
            with (
                patch("sys.argv", ["evaluate", *map(str, arguments), *extra]),
                contextlib.redirect_stdout(printed),
                contextlib.chdir(directory),
            ):
                main()
            files = {p.name: p.read_text("utf-8") for p in Path(directory).iterdir()}
        return json.loads(printed.getvalue()), files

    def test_report_is_printed_as_json(self):
        printed, _ = self.evaluate()
        self.assertEqual(word_error_summary(printed), EXPECTED_WER)

    def test_report_is_saved_to_the_output_file(self):
        printed, files = self.evaluate("--output", "report.json")
        self.assertEqual(json.loads(files["report.json"]), printed)

    def test_report_is_not_saved_without_an_output_file(self):
        _, files = self.evaluate()
        self.assertEqual(sorted(files), ["hypothesis.json", "reference.tsv"])


if __name__ == "__main__":
    unittest.main()
