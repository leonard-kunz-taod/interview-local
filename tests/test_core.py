# German fixtures intentionally exercise non-ASCII speech and speaker names.
import json
from pathlib import Path
import tempfile
import unittest
from interview_local.core import Word, Turn, merge, export, timestamp

class MergeTests(unittest.TestCase):
    def test_speaker_change_overlap_and_gap(self):
        turns = [Turn(0, 1, "cluster9"), Turn(1, 3, "cluster2")]
        result = merge([Word(.2,.6,"Guten"), Word(.6,.9,"Tag."),
                        Word(.95,1.3,"Hallo!"), Word(5,5.2,"Echo")], turns)
        self.assertEqual([w["speaker"] for w in result["words"]], ["A","A","B",None])
        self.assertEqual(result["segments"][0]["text"], "Guten Tag.")
        self.assertTrue(result["words"][-1]["uncertain"])

    def test_tie_is_flagged_and_zero_length_preserved(self):
        result = merge([Word(.5,1.5,"Ja"), Word(1,1,"42", "estimated")],
                       [Turn(0,1,"x"), Turn(1,2,"y")])
        self.assertTrue(result["words"][0]["uncertain"])
        self.assertEqual(result["words"][1]["speaker"], "B")
        self.assertTrue(result["words"][1]["uncertain"])

    def test_empty_and_export_renames(self):
        self.assertEqual(merge([],[])["segments"], [])
        result = merge([Word(0,1,"Grüße")], [Turn(0,2,"x")])
        result["speakers"]["A"] = "Änne"
        with tempfile.TemporaryDirectory() as d:
            export(result, Path(d))
            self.assertIn("Änne", (Path(d)/"transcript.txt").read_text(encoding="utf-8"))
            decoded = json.loads((Path(d)/"transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(decoded["words"][0]["text"], "Grüße")
            self.assertEqual(decoded["speakers"]["A"], "Änne")
        self.assertEqual(timestamp(3599.9996), "01:00:00.000")

if __name__ == "__main__":
    unittest.main()
