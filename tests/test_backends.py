import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from interview_local.backends import FasterWhisperTranscriber, GermanWhisperXAligner, fallback_words

class TranscriptionTests(unittest.TestCase):
    def test_word_probability_is_stored(self):
        word = SimpleNamespace(start=0., end=.4, word=" Guten", probability=.8125)
        segment = SimpleNamespace(start=0., end=.4, text=" Guten", words=[word])
        transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
        transcriber.model = SimpleNamespace(transcribe=lambda *a, **kw: (iter([segment]), SimpleNamespace(language="de")))
        result = transcriber.transcribe(Path("unused.wav"))
        self.assertEqual(result["segments"][0]["words"][0]["probability"], .8125)

    def test_segment_without_words_has_no_whisper_probability(self):
        words = fallback_words(dict(start=0., end=1., text="Guten Tag", words=[]))
        self.assertEqual([(w["timing"], w["probability"]) for w in words], [("estimated", None)])

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

    def test_segment_without_audio_samples_falls_back_to_whisper_words(self):
        import numpy as np
        import soundfile as sf
        aligner = GermanWhisperXAligner.__new__(GermanWhisperXAligner)
        aligner.model = None
        aligner.metadata = {}
        segment = {"start": 1., "end": 1., "text": "Ja",
                   "words": [{"start": 1., "end": 1., "text": "Ja", "timing": "whisper"}]}
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory)/"test.wav"
            sf.write(audio, np.zeros(48000), 16000)
            with patch.dict(sys.modules, {"whisperx.alignment": SimpleNamespace(align=None)}):
                result = aligner.align(audio, {"segments": [segment]})
        self.assertEqual([(w["text"], w["timing"]) for w in result], [("Ja", "whisper_fallback")])
