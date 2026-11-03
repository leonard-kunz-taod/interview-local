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


if __name__ == "__main__":
    unittest.main()
