# German fixtures intentionally exercise non-ASCII speech and speaker names.
import unittest

from interview_local.core import Turn, Word, merge, overlap_regions, render, timestamp

GREETING_TURNS = [Turn(0, 1, "cluster9"), Turn(1, 3, "cluster2")]
GREETING_WORDS = [
    Word(0.2, 0.6, "Guten"),
    Word(0.6, 0.9, "Tag."),
    Word(0.95, 1.3, "Hallo!"),
    Word(5, 5.2, "Echo"),
]
NEIGHBOURING_TURNS = [Turn(0, 1, "x"), Turn(1, 2, "y")]


def overlapping_interview():
    return merge(
        [Word(0.2, 0.8, "Hello"), Word(1.1, 1.3, "yes")],
        [Turn(0, 1, "first"), Turn(1, 3, "second")],
        [Turn(0, 1.5, "first"), Turn(1, 3, "second")],
    )


class SpeakerAssignmentTests(unittest.TestCase):
    def test_each_word_takes_the_speaker_of_its_most_overlapping_turn(self):
        result = merge(GREETING_WORDS, GREETING_TURNS)
        speakers = [w["speaker"] for w in result["words"]]
        self.assertEqual(speakers, ["A", "A", "B", None])

    def test_speakers_are_lettered_in_order_of_their_first_turn(self):
        result = merge(GREETING_WORDS, GREETING_TURNS)
        self.assertEqual(result["speakers"], {"A": "Speaker A", "B": "Speaker B"})

    def test_word_outside_every_turn_is_uncertain(self):
        result = merge([Word(5, 5.2, "Echo")], GREETING_TURNS)
        reasons = result["words"][0]["uncertainty_reasons"]
        self.assertEqual(reasons, ["no_matching_speaker"])

    def test_near_tie_between_two_turns_is_flagged_ambiguous(self):
        result = merge([Word(0.5, 1.5, "Ja")], NEIGHBOURING_TURNS)
        reasons = result["words"][0]["uncertainty_reasons"]
        self.assertEqual(reasons, ["ambiguous_speaker_boundary"])

    def test_zero_length_word_goes_to_the_turn_containing_it(self):
        result = merge([Word(1, 1, "42")], NEIGHBOURING_TURNS)
        self.assertEqual(result["words"][0]["speaker"], "B")

    def test_zero_length_estimated_word_is_uncertain(self):
        result = merge([Word(1, 1, "42", "estimated")], NEIGHBOURING_TURNS)
        reasons = result["words"][0]["uncertainty_reasons"]
        self.assertEqual(
            reasons, ["invalid_word_duration", "estimated_or_fallback_timing"]
        )

    def test_empty_input_produces_no_segments(self):
        self.assertEqual(merge([], [])["segments"], [])


class SegmentTests(unittest.TestCase):
    def test_consecutive_words_of_one_speaker_form_one_segment(self):
        result = merge(GREETING_WORDS, GREETING_TURNS)
        texts = [s["text"] for s in result["segments"]]
        self.assertEqual(texts, ["Guten Tag.", "Hallo!", "Echo"])

    def test_pause_of_one_and_a_half_seconds_starts_a_new_segment(self):
        result = merge([Word(0, 1, "eins"), Word(2.5, 3, "zwei")], [Turn(0, 4, "x")])
        texts = [s["text"] for s in result["segments"]]
        self.assertEqual(texts, ["eins", "zwei"])

    def test_segment_is_closed_once_it_would_reach_thirty_five_seconds(self):
        words = [Word(i, i + 0.5, f"w{i}") for i in range(37)]
        result = merge(words, [Turn(0, 40, "x")])
        sizes = [len(s["word_ids"]) for s in result["segments"]]
        self.assertEqual(sizes, [35, 2])


class OverlapTests(unittest.TestCase):
    def test_simultaneous_regular_turns_form_an_overlap_region(self):
        overlaps = overlapping_interview()["overlaps"]
        self.assertEqual(overlaps, [{"start": 1, "end": 1.5, "speakers": ["A", "B"]}])

    def test_word_before_the_overlap_is_not_marked(self):
        self.assertFalse(overlapping_interview()["words"][0]["overlap"])

    def test_word_in_the_overlap_keeps_its_exclusive_speaker_and_is_flagged(self):
        word = overlapping_interview()["words"][1]
        keys = ("speaker", "overlap", "overlap_speakers", "uncertainty_reasons")
        self.assertEqual(
            {key: word[key] for key in keys},
            {
                "speaker": "B",
                "overlap": True,
                "overlap_speakers": ["A", "B"],
                "uncertainty_reasons": ["overlapping_speech"],
            },
        )

    def test_duplicate_tracks_of_one_speaker_are_not_overlap(self):
        turns = [Turn(0, 2, "x"), Turn(1, 2, "x")]
        self.assertEqual(overlap_regions(turns, {"x": "A"}), [])

    def test_touching_turns_are_not_overlap(self):
        turns = [Turn(0, 2, "x"), Turn(2, 3, "y")]
        self.assertEqual(overlap_regions(turns, {"x": "A", "y": "B"}), [])

    def test_missing_regular_turns_mark_overlap_detection_unavailable(self):
        self.assertFalse(merge([], [])["overlap_detection_available"])


class TimestampTests(unittest.TestCase):
    def test_rounds_to_milliseconds_across_the_hour(self):
        self.assertEqual(timestamp(3599.9996), "01:00:00.000")


class RenderTests(unittest.TestCase):
    def test_flagged_segment_shows_name_time_range_and_flags(self):
        segment = {
            "start": 0,
            "end": 1.5,
            "text": "Grüße",
            "speaker": "A",
            "uncertain": True,
            "overlap": True,
        }
        text = render({"speakers": {"A": "Änne"}, "segments": [segment]})
        self.assertEqual(
            text,
            "[00:00:00.000 – 00:00:01.500] Änne "
            "[Review assignment; Overlapping speech]\nGrüße\n",
        )

    def test_segment_without_speaker_is_unassigned(self):
        segment = {"start": 0, "end": 1, "text": "Ja", "speaker": None}
        text = render({"speakers": {"A": "Änne"}, "segments": [segment]})
        self.assertEqual(text, "[00:00:00.000 – 00:00:01.000] Unassigned\nJa\n")

    def test_empty_transcript_says_no_speech_detected(self):
        self.assertEqual(
            render({"speakers": {}, "segments": []}), "No speech detected."
        )


if __name__ == "__main__":
    unittest.main()
