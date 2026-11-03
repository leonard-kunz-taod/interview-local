import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from interview_local.backends import GermanWhisperXAligner

class AlignmentTests(unittest.TestCase):
    def test_word_loss_falls_back_and_missing_time_is_marked(self):
        import numpy as np
        import soundfile as sf
        aligner = GermanWhisperXAligner.__new__(GermanWhisperXAligner)
        aligner.model = None
        aligner.metadata = {}
        segment = dict(start=1., end=2., text="Test 42", words=[
            dict(start=1.,end=1.3,text="Test",timing="whisper"),
            dict(start=1.3,end=2.,text="42",timing="whisper")])
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory)/"test.wav"
            sf.write(audio,np.zeros(48000),16000)
            fake = SimpleNamespace(align=lambda *a,**kw: {"word_segments": [{"word":"Test","start":0.,"end":.3}]})
            with patch.dict(sys.modules,{"whisperx.alignment":fake}):
                result = aligner.align(audio,{"segments":[segment]})
            self.assertEqual([w["text"] for w in result],["Test","42"])
            self.assertTrue(all(w["timing"]=="whisper_fallback" for w in result))
            fake = SimpleNamespace(align=lambda *a,**kw: {"word_segments": [
                {"word":"Test","start":0.,"end":.3}, {"word":"42"}]})
            with patch.dict(sys.modules,{"whisperx.alignment":fake}):
                result = aligner.align(audio,{"segments":[segment]})
            self.assertEqual(result[0]["start"],1.)
            self.assertEqual(result[1]["timing"],"estimated")
