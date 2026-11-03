import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from interview_local.backends import FasterWhisperTranscriber, GermanWhisperXAligner, fallback_words

SAMPLE_RATE = 16000
RECORDING_SECONDS = 3
ALIGNMENT_MODEL = object()
ALIGNMENT_METADATA = {"language": "de"}
TORCH_DEVICE_TYPES = {"cpu", "cuda", "mps"}


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel: reads the audio file, reports the language it was told to use and,
    like faster-whisper, yields word timings only when they are requested."""

    def __init__(self, segments):
        self.segments = segments

    def transcribe(self, audio, language=None, word_timestamps=False, **options):
        if not Path(audio).is_file():
            raise FileNotFoundError(audio)
        segments = [s if word_timestamps else SimpleNamespace(**{**vars(s), "words": None}) for s in self.segments]
        return iter(segments), SimpleNamespace(language=language)


def transcriber_with(segments):
    transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
    transcriber.model = FakeWhisperModel(segments)
    return transcriber


class TranscriptionTests(unittest.TestCase):
    def test_transcript_lists_german_segments_with_their_timed_words(self):
        word = SimpleNamespace(start=0., end=.4, word=" Guten", probability=np.float32(.8125))
        segment = SimpleNamespace(start=0., end=.4, text=" Guten", words=[word])
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "original.wav"
            audio.touch()
            result = transcriber_with([segment]).transcribe(audio)
        expected_word = {"start": 0., "end": .4, "text": " Guten", "timing": "whisper", "probability": .8125}
        self.assertEqual(result, {"segments": [{"start": 0., "end": .4, "text": " Guten", "words": [expected_word]}],
                                  "language": "de"})

    def test_segment_without_words_has_no_whisper_probability(self):
        words = fallback_words(dict(start=0., end=1., text="Guten Tag", words=[]))
        self.assertEqual([(w["timing"], w["probability"]) for w in words], [("estimated", None)])


def fake_alignment(word_segments=None):
    """Stands in for whisperx.alignment. Like align(), it needs the loaded model, float32 samples and a torch device
    type; it returns the given word segments, or else times the segment's words evenly within the segment."""
    def align(transcript, model, align_model_metadata, audio, device, return_char_alignments=False):
        if model is not ALIGNMENT_MODEL or align_model_metadata is not ALIGNMENT_METADATA:
            raise RuntimeError("The alignment model is not loaded.")
        if audio.dtype != np.float32:
            raise RuntimeError("expected scalar type Float but found Double")
        if device not in TORCH_DEVICE_TYPES:
            raise RuntimeError(f"Expected one of {sorted(TORCH_DEVICE_TYPES)} device type, got {device}")
        (segment,) = transcript
        if word_segments is not None:
            return {"word_segments": word_segments}
        words = segment["text"].split()
        step = (segment["end"] - segment["start"]) / len(words)
        return {"word_segments": [{"word": w, "start": segment["start"] + i * step, "end": segment["start"] + (i + 1) * step}
                                  for i, w in enumerate(words)]}
    return SimpleNamespace(align=align)


def aligned(segment, alignment):
    """Align one ASR segment of a silent recording with the given whisperx.alignment stand-in."""
    aligner = GermanWhisperXAligner.__new__(GermanWhisperXAligner)
    aligner.model, aligner.metadata = ALIGNMENT_MODEL, ALIGNMENT_METADATA
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "original.wav"
        sf.write(audio, np.zeros(SAMPLE_RATE * RECORDING_SECONDS), SAMPLE_RATE)
        with patch.dict(sys.modules, {"whisperx.alignment": alignment}):
            return aligner.align(audio, {"segments": [segment]})


def whisper_segment(start, end, *texts):
    """An ASR segment whose Whisper words share its time span evenly."""
    step = (end - start) / len(texts)
    words = [{"start": start + i * step, "end": start + (i + 1) * step, "text": text, "timing": "whisper", "probability": .9}
             for i, text in enumerate(texts)]
    return {"start": start, "end": end, "text": " ".join(texts), "words": words}


class AlignmentTests(unittest.TestCase):
    def test_aligned_words_are_placed_in_recording_time(self):
        result = aligned(whisper_segment(1., 2., "Guten", "Tag"), fake_alignment())
        self.assertEqual(result, [{"start": 1., "end": 1.5, "text": "Guten", "timing": "aligned", "probability": None},
                                  {"start": 1.5, "end": 2., "text": "Tag", "timing": "aligned", "probability": None}])

    def test_segment_at_the_recording_start_is_aligned_from_its_first_sample(self):
        result = aligned(whisper_segment(0., 1., "Ja"), fake_alignment())
        self.assertEqual([(w["start"], w["end"]) for w in result], [(0., 1.)])

    def test_word_the_aligner_could_not_time_fully_is_estimated_up_to_its_segment_end(self):
        word_segments = [{"word": "Test", "start": 0., "end": .25}, {"word": "42"}, {"word": "mal", "start": .5}]
        result = aligned(whisper_segment(1., 2., "Test", "42", "mal"), fake_alignment(word_segments))
        self.assertEqual(result, [{"start": 1., "end": 1.25, "text": "Test", "timing": "aligned", "probability": None},
                                  {"start": 1., "end": 2., "text": "42", "timing": "estimated", "probability": None},
                                  {"start": 1.5, "end": 2., "text": "mal", "timing": "estimated", "probability": None}])

    def test_aligner_losing_a_word_falls_back_to_the_whisper_words(self):
        segment = whisper_segment(1., 2., "Test", "42")
        result = aligned(segment, fake_alignment([{"word": "Test", "start": 0., "end": .25}]))
        self.assertEqual(result, [dict(w, timing="whisper_fallback") for w in segment["words"]])

    def test_segment_without_audio_samples_falls_back_to_whisper_words(self):
        segment = {"start": 1., "end": 1., "text": "Ja",
                   "words": [{"start": 1., "end": 1., "text": "Ja", "timing": "whisper"}]}
        result = aligned(segment, SimpleNamespace(align=None))
        self.assertEqual([(w["text"], w["timing"]) for w in result], [("Ja", "whisper_fallback")])


if __name__ == "__main__":
    unittest.main()
