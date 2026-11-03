# German fixtures intentionally exercise non-ASCII speech and speaker names.
import json
from pathlib import Path
import tempfile
import unittest
from interview_local.core import Word, Turn, merge, export, timestamp

class MergeTests(unittest.TestCase):
    def test_speaker_change_overlap_and_gap(self):
        turns = [Turn(0, 1, "cluster9"), Turn(1, 3, "cluster2")]
        result = merge([Word(.2,.6,"Guten",.9), Word(.6,.9,"Tag.",.9),
                        Word(.95,1.3,"Hallo!",.9), Word(5,5.2,"Echo",.9)], turns)
        self.assertEqual([w["speaker"] for w in result["words"]], ["A","A","B",None])
        self.assertEqual(result["segments"][0]["text"], "Guten Tag.")
        self.assertTrue(result["words"][-1]["uncertain"])

    def test_tie_is_flagged_and_zero_length_preserved(self):
        result = merge([Word(.5,1.5,"Ja",.9), Word(1,1,"42",None,"estimated")],
                       [Turn(0,1,"x"), Turn(1,2,"y")])
        self.assertTrue(result["words"][0]["uncertain"])
        self.assertEqual(result["words"][1]["speaker"], "B")
        self.assertTrue(result["words"][1]["uncertain"])

    def test_empty_and_export_renames(self):
        self.assertEqual(merge([],[])["segments"], [])
        result = merge([Word(0,1,"Grüße",.9)], [Turn(0,2,"x")])
        result["speakers"]["A"] = "Änne"
        with tempfile.TemporaryDirectory() as d:
            export(result, Path(d))
            self.assertIn("Änne", (Path(d)/"transcript.txt").read_text(encoding="utf-8"))
            decoded = json.loads((Path(d)/"transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(decoded["words"][0]["text"], "Grüße")
            self.assertEqual(decoded["speakers"]["A"], "Änne")
        self.assertEqual(timestamp(3599.9996), "01:00:00.000")

class WordProbabilityTests(unittest.TestCase):
    def test_merge_copies_word_probability_unchanged(self):
        result = merge([Word(0,.5,"Guten",.42), Word(.5,1,"Tag",None,"estimated")], [Turn(0,2,"x")])
        self.assertEqual([w["probability"] for w in result["words"]], [.42, None])

    def test_whisper_word_requires_a_probability(self):
        for timing in ("whisper", "whisper_fallback"):
            with self.subTest(timing=timing), self.assertRaises(ValueError):
                Word(0, 1, "Tag", None, timing)

    def test_aligned_or_estimated_word_has_no_whisper_probability(self):
        for timing in ("aligned", "estimated"):
            with self.subTest(timing=timing), self.assertRaises(ValueError):
                Word(0, 1, "Tag", .9, timing)


class SpeakerLetterTests(unittest.TestCase):
    def test_speaker_heard_only_in_regular_diarization_gets_the_next_letter(self):
        result = merge([Word(0, 1, "Ja", .9)], [Turn(0, 2, "x")], [Turn(0, 2, "x"), Turn(1, 2, "y")])
        self.assertEqual(result["speakers"], {"A": "Speaker A", "B": "Speaker B"})

    def test_regular_diarization_is_kept_with_letters(self):
        result = merge([], [Turn(0, 2, "x")], [Turn(1, 2, "y"), Turn(0, 2, "x")])
        self.assertEqual(result["regular_diarization"], [{"start": 1, "end": 2, "speaker": "B"},
                                                         {"start": 0, "end": 2, "speaker": "A"}])


class UncertaintyTests(unittest.TestCase):
    def reasons(self, words, turns, regular_turns=None):
        return [w["uncertainty_reasons"] for w in merge(words, turns, regular_turns)["words"]]

    def test_word_during_simultaneous_speech_is_overlapping(self):
        result = merge([Word(1, 1.5, "Ja", .9)], [Turn(0, 2, "x")], [Turn(0, 2, "x"), Turn(1, 2, "y")])
        word = result["words"][0]
        self.assertEqual((word["overlap"], word["overlap_speakers"], word["uncertainty_reasons"]),
                         (True, ["A", "B"], ["overlapping_speech"]))

    def test_zero_length_word_inside_simultaneous_speech_is_overlapping(self):
        result = merge([Word(1.5, 1.5, "Ja", None, "estimated")], [Turn(0, 2, "x")],
                       [Turn(0, 2, "x"), Turn(1, 2, "y")])
        self.assertEqual(result["words"][0]["overlap_speakers"], ["A", "B"])

    def test_word_after_simultaneous_speech_is_not_overlapping(self):
        result = merge([Word(2.5, 3, "Ja", .9)], [Turn(0, 3, "x")], [Turn(0, 3, "x"), Turn(1, 2, "y")])
        self.assertEqual(result["words"][0]["overlap"], False)

    def test_fallback_timing_is_uncertain(self):
        self.assertEqual(self.reasons([Word(0, 1, "Ja", .9, "whisper_fallback")], [Turn(0, 2, "x")]),
                         [["estimated_or_fallback_timing"]])

    def test_word_without_speaker_has_no_match_and_invalid_duration(self):
        self.assertEqual(self.reasons([Word(5, 4, "Ja", .9)], [Turn(0, 2, "x")]),
                         [["no_matching_speaker", "invalid_word_duration"]])

    def test_clear_majority_overlap_is_not_ambiguous(self):
        self.assertEqual(self.reasons([Word(.5, 1.1, "Ja", .9)], [Turn(0, 1, "x"), Turn(1, 2, "y")]), [[]])

    def test_overlap_detection_is_available_with_empty_regular_diarization(self):
        self.assertTrue(merge([], [], [])["overlap_detection_available"])


class SegmentTests(unittest.TestCase):
    def segment_texts(self, words, turns):
        return [s["text"] for s in merge(words, turns)["segments"]]

    def test_whitespace_word_is_kept_but_not_written_into_segments(self):
        result = merge([Word(0, .5, "Ja", .9), Word(.5, 1, " ", .9)], [Turn(0, 2, "x")])
        self.assertEqual(([w["text"] for w in result["words"]], result["segments"][0]["word_ids"]), (["Ja", ""], [0]))

    def test_same_speaker_words_join_one_segment(self):
        self.assertEqual(self.segment_texts([Word(0, .5, "Guten", .9), Word(1.9, 2.2, "Tag", .9)], [Turn(0, 3, "x")]),
                         ["Guten Tag"])

    def test_pause_of_one_and_a_half_seconds_starts_a_new_segment(self):
        self.assertEqual(self.segment_texts([Word(0, .5, "Guten", .9), Word(2, 2.5, "Tag", .9)], [Turn(0, 3, "x")]),
                         ["Guten", "Tag"])

    def test_word_ending_35_seconds_after_the_segment_start_starts_a_new_one(self):
        words = [Word(i, i + 1, f"w{i}", .9) for i in range(36)]
        self.assertEqual([len(s.split()) for s in self.segment_texts(words, [Turn(0, 40, "x")])], [34, 2])

    def test_segment_spans_its_words_and_carries_their_flags(self):
        result = merge([Word(0, 1, "Ja", .9), Word(1, 1.5, "nein", None, "estimated")], [Turn(0, 2, "x")])
        segment = result["segments"][0]
        self.assertEqual((segment["start"], segment["end"], segment["uncertain"], segment["overlap"]),
                         (0, 1.5, True, False))


if __name__ == "__main__":
    unittest.main()
