import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from interview_local.backends import GermanWhisperXAligner

SEGMENT = {
    "start": 1.0,
    "end": 2.0,
    "text": "Test 42",
    "words": [
        {"start": 1.0, "end": 1.3, "text": "Test", "timing": "whisper"},
        {"start": 1.3, "end": 2.0, "text": "42", "timing": "whisper"},
    ],
}


def aligned(word_segments):
    """Align SEGMENT in a silent three-second recording with a stubbed model."""
    aligner = GermanWhisperXAligner.__new__(GermanWhisperXAligner)
    aligner.model = None
    aligner.metadata = {}
    alignment = SimpleNamespace(align=lambda *a, **kw: {"word_segments": word_segments})
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "test.wav"
        sf.write(audio, np.zeros(48000), 16000)
        with patch.dict(sys.modules, {"whisperx.alignment": alignment}):
            return aligner.align(audio, {"segments": [SEGMENT]})


class AlignmentTests(unittest.TestCase):
    def test_alignment_losing_a_word_falls_back_to_whisper_words(self):
        result = aligned([{"word": "Test", "start": 0.0, "end": 0.3}])
        self.assertEqual(
            [(w["text"], w["timing"]) for w in result],
            [("Test", "whisper_fallback"), ("42", "whisper_fallback")],
        )

    def test_aligned_word_without_timing_spans_the_segment_as_estimated(self):
        result = aligned([{"word": "Test", "start": 0.0, "end": 0.3}, {"word": "42"}])
        self.assertEqual(
            [(w["start"], w["end"], w["timing"]) for w in result],
            [(1.0, 1.3, "aligned"), (1.0, 2.0, "estimated")],
        )


if __name__ == "__main__":
    unittest.main()
