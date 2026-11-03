import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from interview_local.pipeline import process, worker

class PipelineTests(unittest.TestCase):
    def test_orchestration_sequential_and_cleanup(self):
        calls = []
        temp = []
        def worker(stage, source, target, *args):
            calls.append(stage)
            temp.append(target.parent)
            if stage == "check-access":
                return
            if stage == "decode":
                target.touch()
            else:
                content = {"asr": {"segments": []},
                    "align": [{"start": 0, "end": 1, "text": "Hallo", "timing": "aligned", "probability": None}],
                    "diarize": [{"start": 0, "end": 1, "speaker": "s0"}]}[stage]
                target.write_text(json.dumps(content))
        with tempfile.TemporaryDirectory() as d:
            source = Path(d)/"test.m4a"
            source.touch()
            with patch("interview_local.pipeline.worker", worker):
                result = process(source, status=lambda _: None, alignment=True)
        self.assertEqual(calls, ["decode", "asr", "check-access", "align", "diarize"])
        self.assertEqual(result["segments"][0]["speaker"], "A")
        self.assertTrue(all(not p.exists() for p in temp))
        self.assertEqual(result["metadata"]["detected_speakers"], 1)

    def test_error_cleanup_and_invalid_input(self):
        with self.assertRaises(ValueError):
            process(Path("/missing.wav"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"broken.wav"
            p.touch()
            with patch("interview_local.pipeline.worker", side_effect=RuntimeError("Broken")):
                with self.assertRaises(RuntimeError):
                    process(p, status=lambda _: None)

class DecodeFailureTests(unittest.TestCase):
    def test_unreadable_audio_gets_its_own_message(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d)/"notes.wav"
            fake.write_text("This is not audio.")
            with self.assertRaises(RuntimeError) as raised:
                worker("decode", fake, Path(d)/"decoded.wav", status=lambda _: None)
        message = str(raised.exception)
        names_audio_cause_without_login_advice = ("could not be read as audio" in message, "401/403" in message)
        self.assertEqual(names_audio_cause_without_login_advice, (True, False))

if __name__ == "__main__":
    unittest.main()
