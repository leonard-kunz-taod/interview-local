# German fixtures intentionally exercise non-ASCII speech and speaker names.
import json
from pathlib import Path
import tempfile
import unittest
from interview_local.core import AMBIGUOUS_OVERLAP_RATIO, Word, Turn, merge, export, timestamp

SPEAKER_CHANGE_WORDS = [Word(.2, .6, "Guten", .9, "whisper"), Word(.6, .9, "Tag.", .9, "whisper"),
                         Word(.95, 1.3, "Hallo!", .9, "whisper"), Word(5, 5.2, "Echo", .9, "whisper")]
SPEAKER_CHANGE_TURNS = [Turn(0, 1, "cluster9"), Turn(1, 3, "cluster2")]


class MergeTests(unittest.TestCase):
    def test_words_take_the_speaker_of_their_most_overlapping_turn(self):
        result = merge(SPEAKER_CHANGE_WORDS[:3], SPEAKER_CHANGE_TURNS, [])
        self.assertEqual([w["speaker"] for w in result["words"]], ["A", "A", "B"])

    def test_word_after_every_turn_is_unassigned_and_uncertain(self):
        word = merge(SPEAKER_CHANGE_WORDS, SPEAKER_CHANGE_TURNS, [])["words"][-1]
        self.assertEqual((word["speaker"], word["uncertain"]), (None, True))

    def test_speaker_change_starts_a_new_segment(self):
        result = merge(SPEAKER_CHANGE_WORDS, SPEAKER_CHANGE_TURNS, [])
        self.assertEqual([s["text"] for s in result["segments"]], ["Guten Tag.", "Hallo!", "Echo"])

    def test_zero_length_word_on_a_turn_boundary_takes_the_starting_turn(self):
        result = merge([Word(1, 1, "42", None, "estimated")], [Turn(0, 1, "x"), Turn(1, 2, "y")], [])
        self.assertEqual(result["words"][0]["speaker"], "B")

    def test_zero_length_word_at_the_end_of_the_last_turn_is_unassigned(self):
        result = merge([Word(2, 2, "42", None, "estimated")], [Turn(0, 2, "x")], [])
        self.assertEqual(result["words"][0]["speaker"], None)

    def test_no_words_give_no_segments(self):
        self.assertEqual(merge([], [], [])["segments"], [])

    def test_speaker_transcript_lists_speakers_segments_words_and_diarization(self):
        result = merge([Word(0, 1, "Ja", .9, "whisper")], [Turn(0, 2, "x")], [])
        word = {"word_id": 0, "start": 0, "end": 1, "text": "Ja", "speaker": "A", "timing": "whisper", "probability": .9,
                    "uncertain": False, "uncertainty_reasons": [], "overlap": False, "overlap_speakers": []}
        segment = {"start": 0, "end": 1, "text": "Ja", "speaker": "A", "uncertain": False, "overlap": False, "word_ids": [0]}
        self.assertEqual(result, {"schema_version": 2, "language": "de", "speakers": {"A": "Speaker A"},
                                      "segments": [segment], "words": [word],
                                      "diarization": [{"start": 0, "end": 2, "speaker": "A"}], "regular_diarization": [],
                                      "overlaps": [], "overlap_detection_available": True})


class ExportTests(unittest.TestCase):
    def test_export_keeps_renamed_speakers_and_umlauts_in_both_files(self):
        result = merge([Word(0, 1, "Grüße", .9, "whisper")], [Turn(0, 2, "x")], [])
        result["speakers"]["A"] = "Änne"
        with tempfile.TemporaryDirectory() as d:
            export(result, Path(d))
            text = (Path(d) / "transcript.txt").read_text(encoding="utf-8")
            decoded = json.loads((Path(d) / "transcript.json").read_text(encoding="utf-8"))
        self.assertEqual(("Änne" in text, decoded["words"][0]["text"], decoded["speakers"]["A"]),
                         (True, "Grüße", "Änne"))

    def test_timestamp_rounds_up_to_the_next_hour(self):
        self.assertEqual(timestamp(3599.9996), "01:00:00.000")

class WordProbabilityTests(unittest.TestCase):
    def test_merge_copies_word_probability_unchanged(self):
        result = merge([Word(0,.5,"Guten",.42, "whisper"), Word(.5,1,"Tag",None,"estimated")], [Turn(0,2,"x")], [])
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
        result = merge([Word(0, 1, "Ja", .9, "whisper")], [Turn(0, 2, "x")], [Turn(0, 2, "x"), Turn(1, 2, "y")])
        self.assertEqual(result["speakers"], {"A": "Speaker A", "B": "Speaker B"})

    def test_regular_diarization_is_kept_with_letters(self):
        result = merge([], [Turn(0, 2, "x")], [Turn(1, 2, "y"), Turn(0, 2, "x")])
        self.assertEqual(result["regular_diarization"], [{"start": 1, "end": 2, "speaker": "B"},
                                                         {"start": 0, "end": 2, "speaker": "A"}])


class UncertaintyTests(unittest.TestCase):
    def reasons(self, words, turns):
        return [w["uncertainty_reasons"] for w in merge(words, turns, [])["words"]]

    def test_word_during_simultaneous_speech_is_overlapping(self):
        result = merge([Word(1, 1.5, "Ja", .9, "whisper")], [Turn(0, 2, "x")], [Turn(0, 2, "x"), Turn(1, 2, "y")])
        word = result["words"][0]
        self.assertEqual((word["overlap"], word["overlap_speakers"], word["uncertainty_reasons"]),
                         (True, ["A", "B"], ["overlapping_speech"]))

    def test_zero_length_word_inside_simultaneous_speech_is_overlapping(self):
        result = merge([Word(1.5, 1.5, "Ja", None, "estimated")], [Turn(0, 2, "x")],
                       [Turn(0, 2, "x"), Turn(1, 2, "y")])
        self.assertEqual(result["words"][0]["overlap_speakers"], ["A", "B"])

    def test_word_after_simultaneous_speech_is_not_overlapping(self):
        result = merge([Word(2.5, 3, "Ja", .9, "whisper")], [Turn(0, 3, "x")], [Turn(0, 3, "x"), Turn(1, 2, "y")])
        self.assertEqual(result["words"][0]["overlap"], False)

    def test_fallback_timing_is_uncertain(self):
        self.assertEqual(self.reasons([Word(0, 1, "Ja", .9, "whisper_fallback")], [Turn(0, 2, "x")]),
                         [["estimated_or_fallback_timing"]])

    def test_word_without_speaker_has_no_match_and_invalid_duration(self):
        self.assertEqual(self.reasons([Word(5, 4, "Ja", .9, "whisper")], [Turn(0, 2, "x")]),
                         [["no_matching_speaker", "invalid_word_duration"]])

    def test_tie_between_two_turns_is_an_ambiguous_speaker_boundary(self):
        self.assertEqual(self.reasons([Word(.5, 1.5, "Ja", .9, "whisper")], [Turn(0, 1, "x"), Turn(1, 2, "y")]),
                         [["ambiguous_speaker_boundary"]])

    def test_runner_up_overlapping_exactly_the_ambiguity_ratio_is_ambiguous(self):
        best_overlap = 1.25
        runner_up_overlap = best_overlap * AMBIGUOUS_OVERLAP_RATIO
        word = Word(2 - best_overlap, 2 + runner_up_overlap, "Ja", .9, "whisper")
        self.assertEqual(self.reasons([word], [Turn(0, 2, "x"), Turn(2, 4, "y")]), [["ambiguous_speaker_boundary"]])

    def test_zero_length_word_has_an_invalid_duration(self):
        self.assertEqual(self.reasons([Word(1, 1, "Ja", .9, "whisper")], [Turn(0, 2, "x")]),
                         [["invalid_word_duration"]])

    def test_clear_majority_overlap_is_not_ambiguous(self):
        self.assertEqual(self.reasons([Word(.5, 1.1, "Ja", .9, "whisper")], [Turn(0, 1, "x"), Turn(1, 2, "y")]), [[]])

    def test_overlap_detection_is_available_with_empty_regular_diarization(self):
        self.assertTrue(merge([], [], [])["overlap_detection_available"])


class SegmentTests(unittest.TestCase):
    def segment_texts(self, words, turns):
        return [s["text"] for s in merge(words, turns, [])["segments"]]

    def test_whitespace_word_is_kept_but_not_written_into_segments(self):
        result = merge([Word(0, .5, "Ja", .9, "whisper"), Word(.5, 1, " ", .9, "whisper")], [Turn(0, 2, "x")], [])
        self.assertEqual(([w["text"] for w in result["words"]], result["segments"][0]["word_ids"]), (["Ja", ""], [0]))

    def test_same_speaker_words_join_one_segment(self):
        self.assertEqual(self.segment_texts([Word(0, .5, "Guten", .9, "whisper"), Word(1.9, 2.2, "Tag", .9, "whisper")], [Turn(0, 3, "x")]),
                         ["Guten Tag"])

    def test_pause_of_one_and_a_half_seconds_starts_a_new_segment(self):
        self.assertEqual(self.segment_texts([Word(0, .5, "Guten", .9, "whisper"), Word(2, 2.5, "Tag", .9, "whisper")], [Turn(0, 3, "x")]),
                         ["Guten", "Tag"])

    def test_word_ending_35_seconds_after_the_segment_start_starts_a_new_one(self):
        words = [Word(i, i + 1, f"w{i}", .9, "whisper") for i in range(36)]
        self.assertEqual([len(s.split()) for s in self.segment_texts(words, [Turn(0, 40, "x")])], [34, 2])

    def test_segment_spans_its_words_and_carries_their_flags(self):
        result = merge([Word(0, 1, "Ja", .9, "whisper"), Word(1, 1.5, "nein", None, "estimated")], [Turn(0, 2, "x")], [])
        segment = result["segments"][0]
        self.assertEqual((segment["start"], segment["end"], segment["uncertain"], segment["overlap"]),
                         (0, 1.5, True, False))

    def test_empty_word_inside_a_segment_does_not_end_it(self):
        words = [Word(0, .5, "Ja", .9, "whisper"), Word(.5, 1, " ", .9, "whisper"), Word(1, 1.5, "nein", .9, "whisper")]
        self.assertEqual(self.segment_texts(words, [Turn(0, 2, "x")]), ["Ja nein"])

    def test_segment_lists_the_ids_of_its_words(self):
        result = merge([Word(0, .5, "Ja", .9, "whisper"), Word(.5, 1, "gut", .9, "whisper")], [Turn(0, 2, "x")], [])
        self.assertEqual(result["segments"][0]["word_ids"], [0, 1])

    def test_segment_is_uncertain_when_its_first_word_is(self):
        result = merge([Word(0, 1, "Ja", None, "estimated"), Word(1, 1.5, "nein", .9, "whisper")], [Turn(0, 2, "x")], [])
        self.assertEqual(result["segments"][0]["uncertain"], True)

    def test_segment_overlaps_when_any_of_its_words_overlaps(self):
        words = [Word(0, .5, "Ja", .9, "whisper"), Word(1.5, 2, "gut", .9, "whisper")]
        for case, second_voice in (("first word overlaps", Turn(0, 1, "y")), ("last word overlaps", Turn(1.5, 3, "y"))):
            with self.subTest(case):
                result = merge(words, [Turn(0, 3, "x")], [Turn(0, 3, "x"), second_voice])
                self.assertEqual([s["overlap"] for s in result["segments"]], [True])


if __name__ == "__main__":
    unittest.main()
