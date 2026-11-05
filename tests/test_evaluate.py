# German fixtures intentionally exercise non-ASCII speech and speaker names.
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support.fake_ollama import FakeOllama, ScriptedModel, served_by
from support.platform_encoding import windows_default_encoding

from interview_local import evaluate
from interview_local.evaluate import (
    CAVEAT,
    DER_COLLAR_SECONDS,
    DER_CONDITIONS,
    NORMALIZATION,
    distance,
    main,
    normalize,
    score,
)
from interview_local.llm import enhance

REFERENCE = (
    "Beg\tEnd\tSpeaker\tText\n"
    "10.0\t11.0\tA\tGuten Tag {laugh}\n"
    "11.0\t12.0\tB\tHallo Welt\n"
)
OVERLAPPING_REFERENCE = (
    "Beg\tEnd\tSpeaker\tText\n10.0\t12.0\tA\tGuten Tag\n11.0\t12.0\tB\tHallo\n"
)
GERMAN_REFERENCE = "Beg\tEnd\tSpeaker\tText\n10.0\t11.0\tA\tGrüße aus Köln\n"
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


REFERENCE_OFFSET = 10
DISTANCE_CASES = (
    ("substitution and insertion", ["a", "b", "c"], ["a", "d", "c", "e"], 2),
    ("insertion into nothing", [], ["a"], 1),
    ("deletion of everything", ["a"], [], 1),
    ("identical words", ["a"], ["a"], 0),
)


def scored(hypothesis, start=0, end=2, reference=REFERENCE):
    with tempfile.TemporaryDirectory() as directory:
        reference_file = Path(directory) / "reference.tsv"
        hypothesis_file = Path(directory) / "hypothesis.json"
        reference_file.write_text(reference, encoding="utf-8")
        hypothesis_file.write_text(
            json.dumps(hypothesis, ensure_ascii=False), encoding="utf-8"
        )
        return score(reference_file, hypothesis_file, REFERENCE_OFFSET, start, end)


def word_error_summary(report):
    keys = ("reference_words", "hypothesis_words", "word_errors", "wer")
    return {key: report[key] for key in keys}


def spoken(text, start, end, speaker="A"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


EXPECTED_WER = {
    "reference_words": 4,
    "hypothesis_words": 3,
    "word_errors": 1,
    "wer": 0.25,
}


class NormalizationTests(unittest.TestCase):
    def test_noise_labels_and_punctuation_are_removed(self):
        self.assertEqual(normalize("Grüße, {laugh} Äpfel!"), ["grüße", "äpfel"])

    def test_markup_tags_are_removed(self):
        self.assertEqual(normalize("<b>Hallo</b> Welt"), ["hallo", "welt"])


class DistanceTests(unittest.TestCase):
    def test_edit_distance_counts_insertions_deletions_and_substitutions(self):
        for case, reference, hypothesis, edits in DISTANCE_CASES:
            with self.subTest(case):
                self.assertEqual(distance(reference, hypothesis), edits)


class ScoreTests(unittest.TestCase):
    def test_word_error_rate_covers_words_inside_the_window(self):
        report = scored({"words": WORDS})
        self.assertEqual(word_error_summary(report), EXPECTED_WER)

    def test_words_are_read_from_segments_without_a_word_list(self):
        report = scored({"segments": [{"words": WORDS}]})
        self.assertEqual(word_error_summary(report), EXPECTED_WER)

    def test_raw_segment_without_words_contributes_none(self):
        report = scored({"segments": [{"words": WORDS}, {"text": "Danke"}]})
        self.assertEqual(word_error_summary(report), EXPECTED_WER)

    def test_window_without_reference_words_is_rejected(self):
        with self.assertRaises(ValueError):
            scored({"words": WORDS}, start=50, end=60)

    def test_utterance_ending_at_the_window_start_is_left_out(self):
        report = scored({"words": WORDS}, start=1, end=2)
        self.assertEqual(report["reference_words"], len(["hallo", "welt"]))

    def test_utterance_beginning_at_the_window_end_is_left_out(self):
        report = scored({"words": WORDS}, start=0, end=1)
        self.assertEqual(report["reference_words"], len(["guten", "tag"]))

    def test_word_centred_on_the_window_start_is_counted(self):
        report = scored({"words": [spoken("Hallo", 0.9, 1.1)]}, start=1, end=2)
        self.assertEqual(report["hypothesis_words"], 1)

    def test_word_centred_on_the_window_end_is_left_out(self):
        report = scored({"words": [spoken("Welt", 1.9, 2.1)]}, start=1, end=2)
        self.assertEqual(report["hypothesis_words"], 0)

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

    def test_unassigned_word_counts_once_in_cpwer(self):
        unassigned = spoken("Danke", 1.5, 2, speaker=None)
        report = scored({"words": [*WORDS, unassigned], "diarization": TURNS})
        missing_welt, unassigned_danke = 1, 1
        self.assertEqual(report["cpwer"], (missing_welt + unassigned_danke) / 4)

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

    def test_speaker_change_is_measured_outside_the_collar_only(self):
        late_change = 1.2
        turns = [
            {"start": 0, "end": late_change, "speaker": "A"},
            {"start": late_change, "end": 2, "speaker": "B"},
        ]
        report = scored({"words": WORDS, "diarization": turns})
        half_collar = DER_COLLAR_SECONDS / 2
        confused = late_change - (1 + half_collar)
        boundaries_at_0_1_and_2 = 4 * half_collar
        evaluated = 2 - boundaries_at_0_1_and_2
        self.assertAlmostEqual(
            report["der"]["diarization error rate"], confused / evaluated
        )

    def test_overlapping_reference_speech_counts_toward_der(self):
        turns = [{"start": 0, "end": 2, "speaker": "A"}]
        report = scored(
            {"words": WORDS, "diarization": turns}, reference=OVERLAPPING_REFERENCE
        )
        self.assertGreater(report["der"]["missed detection"], 0)

    def test_speaker_metrics_state_their_der_conditions(self):
        report = scored({"words": WORDS, "diarization": TURNS})
        self.assertEqual(report["der_conditions"], DER_CONDITIONS)

    def test_report_states_its_normalization_and_caveat(self):
        report = scored({"words": WORDS})
        self.assertEqual(
            (report["normalization"], report["caveat"]), (NORMALIZATION, CAVEAT)
        )

    def test_report_states_the_evaluated_window_and_offset(self):
        report = scored({"words": WORDS}, start=0, end=2)
        self.assertEqual(
            [report[key] for key in KEYS_OF_WINDOW],
            [REFERENCE_OFFSET, 0, 2],
        )

    def test_german_files_are_read_as_utf8_where_the_platform_default_is_not(self):
        words = [spoken("Grüße", 0, 0.3), spoken("aus", 0.3, 0.6)]
        with windows_default_encoding():
            report = scored({"words": words}, end=1, reference=GERMAN_REFERENCE)
        self.assertEqual(report["word_errors"], len(["köln"]))


KEYS_OF_WINDOW = ("reference_offset_seconds", "evaluated_start", "evaluated_end")
SPEAKERS = {"A": "Speaker A", "B": "Speaker B"}
SPEAKER_METRICS = ("word_errors", "wer", "cpwer")


def speaker_transcript(words):
    """A stage-2 transcript of the given words, numbered in order."""
    return {
        "stage": "speakers",
        "speakers": SPEAKERS,
        "words": [dict(w, word_id=i) for i, w in enumerate(words)],
        "diarization": TURNS,
    }


def readable_transcript(source, *paragraphs):
    """A stage-3 transcript of (speaker, text, source word ids) paragraphs."""
    return {
        "stage": "readable",
        "source_transcript": source,
        "paragraphs": [
            {"speaker": speaker, "text": text, "source_word_ids": list(ids)}
            for speaker, text, ids in paragraphs
        ],
    }


def speaker_metrics(report):
    return {key: report[key] for key in SPEAKER_METRICS}


class ReadableScoreTests(unittest.TestCase):
    def test_readable_transcript_gets_wer_and_cpwer_but_no_der(self):
        source = speaker_transcript(WORDS)
        readable = readable_transcript(
            source, ("A", "Guten Tag.", [0, 1]), ("B", "Hallo. Danach", [2, 3])
        )
        report = scored(readable)
        self.assertEqual(
            (word_error_summary(report), report["cpwer"], "der" in report),
            (EXPECTED_WER, 0.25, False),
        )

    def test_readable_of_rejected_edits_scores_like_its_source(self):
        source = speaker_transcript(
            [
                spoken("guten", 0, 0.5),
                spoken("Tag.", 0.5, 1),
                spoken("Hallo,", 1, 1.4, speaker="B"),
                spoken("Welt", 1.4, 1.8, speaker=None),
                spoken("Danach", 2.5, 3, speaker="B"),
            ]
        )
        with served_by(FakeOllama(ScriptedModel(invalid_edits=2))):
            readable = enhance(source, lambda _message: None)
        self.assertEqual(
            speaker_metrics(scored(readable)), speaker_metrics(scored(source))
        )

    def test_paragraph_given_to_the_right_speaker_lowers_cpwer_and_keeps_wer(self):
        source = speaker_transcript(
            [
                spoken("Guten", 0, 0.5),
                spoken("Tag", 0.5, 1),
                spoken("Hallo", 1, 1.5),
                spoken("Welt", 1.5, 2, speaker="B"),
            ]
        )
        corrected = readable_transcript(
            source, ("A", "Guten Tag", [0, 1]), ("B", "Hallo Welt", [2, 3])
        )
        before, after = scored(source), scored(corrected)
        self.assertEqual(
            ((before["wer"], before["cpwer"]), (after["wer"], after["cpwer"])),
            ((0.0, 0.5), (0.0, 0.0)),
        )

    def test_inserted_word_takes_the_time_of_the_preceding_source_word(self):
        source = speaker_transcript([spoken("Tag", 0.5, 1), spoken("Danach", 2.5, 3)])
        readable = readable_transcript(source, ("A", "Tag, ja, danach", [0, 1]))
        report = scored(readable, start=0, end=2)
        self.assertEqual(report["hypothesis_words"], len(["tag", "ja"]))

    def test_inserted_first_word_takes_the_time_of_the_paragraphs_first_word(self):
        source = speaker_transcript([spoken("Guten", 0, 0.5), spoken("Hallo", 1, 1.5)])
        readable = readable_transcript(
            source, ("A", "Guten", [0]), ("B", "Ja, hallo", [1])
        )
        report = scored(readable, start=1, end=2)
        self.assertEqual(report["hypothesis_words"], len(["ja", "hallo"]))

    def test_paragraph_text_without_words_contributes_none(self):
        source = speaker_transcript([spoken("Guten", 0, 0.5), spoken("…", 0.5, 1)])
        readable = readable_transcript(source, ("A", "Guten", [0]), ("A", "…", [1]))
        report = scored(readable)
        self.assertEqual(report["hypothesis_words"], len(["guten"]))


class CommandLineTests(unittest.TestCase):
    def evaluate(
        self, *extra, window=("--offset", "10", "--end", "2"), hypothesis=None
    ):
        """Printed report and the files left in the working directory."""
        with tempfile.TemporaryDirectory() as directory:
            reference_file = Path(directory) / "reference.tsv"
            hypothesis_file = Path(directory) / "hypothesis.json"
            reference_file.write_text(REFERENCE, encoding="utf-8")
            hypothesis_file.write_text(
                json.dumps(hypothesis or {"words": WORDS}), encoding="utf-8"
            )
            arguments = [reference_file, hypothesis_file, *window, *extra]
            printed = io.StringIO()
            with (
                patch("sys.argv", ["evaluate", *map(str, arguments)]),
                contextlib.redirect_stdout(printed),
                contextlib.redirect_stderr(io.StringIO()),
                contextlib.chdir(directory),
            ):
                main()
            files = {p.name: p.read_text("utf-8") for p in Path(directory).iterdir()}
        return printed.getvalue(), files

    def rejected_exit_code(self, window):
        with self.assertRaises(SystemExit) as exit_:
            self.evaluate(window=window)
        return exit_.exception.code

    def test_report_is_printed_as_json(self):
        printed, _ = self.evaluate()
        self.assertEqual(word_error_summary(json.loads(printed)), EXPECTED_WER)

    def test_report_is_printed_with_two_space_indentation(self):
        printed, _ = self.evaluate()
        self.assertEqual(printed, json.dumps(json.loads(printed), indent=2) + "\n")

    def test_readable_transcript_is_scored_from_the_command_line(self):
        readable = readable_transcript(
            speaker_transcript(WORDS),
            ("A", "Guten Tag.", [0, 1]),
            ("B", "Hallo. Danach", [2, 3]),
        )
        printed, _ = self.evaluate(hypothesis=readable)
        report = json.loads(printed)
        self.assertEqual(
            (word_error_summary(report), report["cpwer"]), (EXPECTED_WER, 0.25)
        )

    def test_report_is_saved_to_the_output_file(self):
        printed, files = self.evaluate("--output", "report.json")
        self.assertEqual(json.loads(files["report.json"]), json.loads(printed))

    def test_report_is_not_saved_without_an_output_file(self):
        _, files = self.evaluate()
        self.assertEqual(sorted(files), ["hypothesis.json", "reference.tsv"])

    def test_window_start_is_read_in_seconds(self):
        printed, _ = self.evaluate("--start", "0.5")
        self.assertEqual(json.loads(printed)["evaluated_start"], 0.5)

    def test_missing_offset_is_a_usage_error(self):
        self.assertEqual(self.rejected_exit_code(window=("--end", "2")), 2)

    def test_missing_window_end_is_a_usage_error(self):
        self.assertEqual(self.rejected_exit_code(window=("--offset", "10")), 2)

    def test_help_describes_the_evaluation(self):
        printed = io.StringIO()
        with (
            patch("sys.argv", ["evaluate", "--help"]),
            contextlib.redirect_stdout(printed),
            self.assertRaises(SystemExit),
        ):
            main()
        self.assertIn(evaluate.__doc__.strip(), printed.getvalue())


if __name__ == "__main__":
    unittest.main()
