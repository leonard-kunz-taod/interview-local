import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from interview_local.core import Word, Turn, merge, overlap_regions
from interview_local.llm import enhance, review, OllamaClient, validate_edit
from interview_local.artifacts import save_readable, readable_text
from interview_local.pipeline import process

class FakeClient:
    def __init__(self, invalid=False, wording=None):
        self.invalid = invalid
        self.wording = wording
        self.unloaded = False
    def ready(self):
        return {"model":"fake-local-test"}
    def unload(self):
        self.unloaded = True
    def chat(self, instruction, data, schema):
        if "editable_words" in data:
            words = data["editable_words"]
            return {"paragraphs":[dict(first_word=words[0]["id"]+(1 if self.invalid else 0),
                last_word=words[-1]["id"], speaker="A", text=self.wording or " ".join(w["t"] for w in words),
                uncertain=False, reason="Grouped into a readable paragraph.")]}
        return {"reviews":[dict(paragraph_id=p["paragraph_id"],uncertain=False,reasons=[])
                            for p in data["paragraphs"]]}

class OverlapTests(unittest.TestCase):
    def test_regular_overlap_uses_shared_speaker_mapping(self):
        source = merge([Word(.2,.8,"Hello"),Word(1.1,1.3,"yes")],
            [Turn(0,1,"first"),Turn(1,3,"second")],
            [Turn(0,1.5,"first"),Turn(1,3,"second")])
        self.assertEqual(source["overlaps"],[dict(start=1,end=1.5,speakers=["A","B"])])
        self.assertFalse(source["words"][0]["overlap"])
        self.assertEqual(source["words"][1]["overlap_speakers"],["A","B"])
        self.assertTrue(source["words"][1]["uncertain"])
        self.assertEqual(source["words"][1]["speaker"],"B")
    def test_duplicate_tracks_and_touching_boundaries_are_not_overlap(self):
        self.assertEqual(overlap_regions([Turn(0,2,"x"),Turn(1,2,"x"),Turn(2,3,"y")],{"x":"A","y":"B"}),[])
        self.assertFalse(merge([],[])["overlap_detection_available"])

class LLMTests(unittest.TestCase):
    def source(self):
        return merge([Word(0,.3,"Hello"),Word(.3,.7,"there.")],[Turn(0,1,"speaker")])
    def test_edit_preserves_source_and_human_readable_exports(self):
        source = self.source()
        before = copy.deepcopy(source)
        client = FakeClient()
        result = enhance(source,client=client,status=lambda _:None)
        self.assertEqual(source,before)
        self.assertEqual(result["paragraphs"][0]["source_word_ids"],[0,1])
        self.assertTrue(client.unloaded)
        with tempfile.TemporaryDirectory() as directory:
            save_readable(result,Path(directory))
            text=(Path(directory)/"03_readable.md").read_text()
            self.assertIn("### Speaker A",text)
            self.assertIn("Hello there.",text)
            self.assertNotIn('"paragraphs":',text)
    def test_omitted_words_and_large_rewrites_are_rejected(self):
        source=self.source()
        result=enhance(source,client=FakeClient(invalid=True),status=lambda _:None)
        self.assertEqual(result["metadata"]["rejected_chunks"],1)
        self.assertTrue(result["paragraphs"][0]["uncertain"])
        self.assertEqual(result["paragraphs"][0]["text"],"Hello there.")
        result=enhance(source,client=FakeClient(wording="Invented statement"),status=lambda _:None)
        self.assertEqual(result["metadata"]["rejected_chunks"],1)
    def test_review_never_erases_source_uncertainty(self):
        source=self.source()
        source["words"][0]["uncertain"]=True
        edited=enhance(source,client=FakeClient(),status=lambda _:None)
        checked=review(edited,client=FakeClient(),status=lambda _:None)
        self.assertTrue(checked["paragraphs"][0]["uncertain"])
        self.assertFalse(checked["paragraphs"][0]["review_uncertain"])
        self.assertEqual(edited["metadata"]["verification"],"not_reviewed")
        self.assertIn("not_audio_verified",checked["metadata"]["verification"])
    def test_review_detects_modified_provenance_and_duplicate_words(self):
        edited=enhance(self.source(),client=FakeClient(),status=lambda _:None)
        corrupted=copy.deepcopy(edited)
        corrupted["source_transcript"]["words"][0]["text"]="changed"
        with self.assertRaises(ValueError): review(corrupted,client=FakeClient())
        edited["paragraphs"][0]["source_word_ids"]=[0,0]
        with self.assertRaises(ValueError): review(edited,client=FakeClient())
    def test_wording_changes_stay_uncertain_even_when_reviewer_agrees(self):
        text = "one two three four five six seven eight nine ten"
        source = merge([Word(i,i+.5,t) for i,t in enumerate(text.split())],[Turn(0,11,"s")])
        edited = enhance(source,client=FakeClient(wording="one two three four five six seven eight nine"),status=lambda _:None)
        self.assertTrue(edited["paragraphs"][0]["content_changed"])
        checked = review(edited,client=FakeClient(),status=lambda _:None)
        self.assertTrue(checked["paragraphs"][0]["uncertain"])

    def test_local_endpoint_required(self):
        for url in ("https://example.org","http://example.org","http://localhost@evil.org", "http://127.0.0.1/redirect"):
            with self.assertRaises(ValueError): OllamaClient(url)

class PreservationTests(unittest.TestCase):
    def test_raw_survives_later_diarization_failure(self):
        raw={"segments":[dict(start=0,end=1,text="Original",words=[])]}
        def fake(stage,source,target,*args):
            if stage=="decode": target.touch()
            elif stage=="asr": target.write_text(json.dumps(raw))
            elif stage=="check-access": raise RuntimeError("Access denied")
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            audio=root/"audio.wav";audio.touch()
            output=root/"output"
            with patch("interview_local.pipeline.worker",fake):
                with self.assertRaises(RuntimeError): process(audio,output_dir=output,status=lambda _:None)
            self.assertEqual(json.loads((output/"01_raw.json").read_text())["segments"],raw["segments"])
            self.assertIn("Original",(output/"01_raw.txt").read_text())
            self.assertFalse((output/"02_speakers.json").exists())
