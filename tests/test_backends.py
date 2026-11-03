import contextlib
import importlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
from test_models import without_local_model

from interview_local.backends import (
    CommunityDiarizer,
    FasterWhisperTranscriber,
    GermanWhisperXAligner,
)
from interview_local.models import DIARIZATION_ID

TOKEN = "hf_secretToken123"
LOADER_LIBRARIES = ("nltk", "onnxruntime", "torch")
SEGMENT = {
    "start": 1.0,
    "end": 2.0,
    "text": "Test 42",
    "words": [
        {"start": 1.0, "end": 1.3, "text": "Test", "timing": "whisper"},
        {"start": 1.3, "end": 2.0, "text": "42", "timing": "whisper"},
    ],
}


def setUpModule():
    # Torch cannot be imported twice; patched module tables would drop it again.
    for name in LOADER_LIBRARIES:
        importlib.import_module(name)


def aligned(word_segments, segments=(SEGMENT,)):
    """Align segments in a silent three-second recording with a stubbed model."""
    aligner = GermanWhisperXAligner.__new__(GermanWhisperXAligner)
    aligner.model = None
    aligner.metadata = {}
    alignment = SimpleNamespace(align=lambda *a, **kw: {"word_segments": word_segments})
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "test.wav"
        sf.write(audio, np.zeros(48000), 16000)
        with patch.dict(sys.modules, {"whisperx.alignment": alignment}):
            return aligner.align(audio, {"segments": list(segments)})


class FakePipeline:
    """Records how the diarization pipeline was loaded and configured."""

    def __init__(self, checkpoint, token):
        self.checkpoint = checkpoint
        self.token = token
        self.device = None

    @classmethod
    def from_pretrained(cls, checkpoint, token):
        return cls(checkpoint, token)

    def to(self, device):
        self.device = str(device)


def loaded_diarizer(environment):
    pyannote_audio = SimpleNamespace(Pipeline=FakePipeline)
    with (
        patch.dict(sys.modules, {"pyannote.audio": pyannote_audio}),
        patch.dict(os.environ, environment),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        return CommunityDiarizer().pipeline


def loaded_aligner(environment, find=lambda resource: resource):
    alignment = SimpleNamespace(load_align_model=lambda **options: ("model", options))
    with (
        patch.dict(sys.modules, {"whisperx.alignment": alignment}),
        patch.dict(os.environ, environment),
        patch("nltk.data.find", side_effect=find),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        return GermanWhisperXAligner()


def missing_resource(resource):
    raise LookupError(resource)


def recognized(segments):
    """Transcribe with a stubbed Whisper model that yields the given segments."""
    transcriber = FasterWhisperTranscriber.__new__(FasterWhisperTranscriber)
    info = SimpleNamespace(language="de")
    transcriber.model = SimpleNamespace(transcribe=lambda *a, **kw: (segments, info))
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        result = transcriber.transcribe(Path("interview.wav"))
    return result, printed.getvalue()


def whisper_segment(end, words):
    return SimpleNamespace(start=0.0, end=end, text=" Hallo", words=words)


class TranscriptionTests(unittest.TestCase):
    def test_segments_keep_their_whisper_word_timestamps(self):
        word = SimpleNamespace(start=0.0, end=0.5, word=" Hallo")
        result, _ = recognized([whisper_segment(0.5, [word])])
        self.assertEqual(
            result,
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 0.5,
                        "text": " Hallo",
                        "words": [
                            {
                                "start": 0.0,
                                "end": 0.5,
                                "text": " Hallo",
                                "timing": "whisper",
                            }
                        ],
                    }
                ],
                "language": "de",
            },
        )

    def test_segment_without_word_timestamps_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "without word timestamps"):
            recognized([whisper_segment(0.5, None)])

    def test_progress_is_reported_per_segment(self):
        _, printed = recognized([whisper_segment(12.4, []), whisper_segment(30, [])])
        self.assertEqual(
            printed,
            "Transcribed through 12 seconds\nTranscribed through 30 seconds\n",
        )


class DiarizerLoadingTests(unittest.TestCase):
    def test_local_model_is_loaded_without_a_token(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.yaml").touch()
            pipeline = loaded_diarizer(
                {"INTERVIEW_DIARIZATION_MODEL": directory, "HF_TOKEN": TOKEN}
            )
            expected = str(Path(directory).resolve())
        self.assertEqual((pipeline.checkpoint, pipeline.token), (expected, False))

    def test_hub_model_is_loaded_with_the_environment_token(self):
        with without_local_model():
            pipeline = loaded_diarizer({"HF_TOKEN": TOKEN})
        self.assertEqual((pipeline.checkpoint, pipeline.token), (DIARIZATION_ID, TOKEN))

    def test_hub_model_without_a_token_uses_the_cached_login(self):
        with without_local_model():
            pipeline = loaded_diarizer({"HF_TOKEN": ""})
        self.assertIsNone(pipeline.token)

    def test_pipeline_runs_on_the_cpu_one_batch_at_a_time(self):
        with without_local_model():
            pipeline = loaded_diarizer({})
        self.assertEqual(
            (
                pipeline.device,
                pipeline.segmentation_batch_size,
                pipeline.embedding_batch_size,
            ),
            ("cpu", 1, 1),
        )


class AlignerLoadingTests(unittest.TestCase):
    def test_german_alignment_model_is_loaded_on_the_cpu(self):
        aligner = loaded_aligner({"HF_HUB_OFFLINE": "0"})
        self.assertEqual(
            (aligner.model, aligner.metadata),
            (
                "model",
                {
                    "language_code": "de",
                    "device": "cpu",
                    "model_name": "facebook/wav2vec2-base-10k-voxpopuli-ft-de",
                    "model_cache_only": False,
                },
            ),
        )

    def test_offline_loading_uses_only_the_model_cache(self):
        aligner = loaded_aligner({"HF_HUB_OFFLINE": "1"})
        self.assertTrue(aligner.metadata["model_cache_only"])

    def test_offline_loading_without_language_data_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "NLTK language data is missing"):
            loaded_aligner({"HF_HUB_OFFLINE": "1"}, find=missing_resource)

    def test_online_loading_does_not_require_cached_language_data(self):
        aligner = loaded_aligner({"HF_HUB_OFFLINE": "0"}, find=missing_resource)
        self.assertEqual(aligner.model, "model")


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

    def test_segment_beyond_the_recording_falls_back_to_whisper_words(self):
        late = dict(SEGMENT, start=3.0, end=4.0)
        result = aligned([], segments=[late])
        self.assertEqual(
            [w["timing"] for w in result], ["whisper_fallback", "whisper_fallback"]
        )

    def test_alignment_reports_progress_per_aligned_segment(self):
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            aligned([{"word": "Test", "start": 0.0, "end": 0.3}, {"word": "42"}])
        self.assertEqual(printed.getvalue(), "Aligned through 2 seconds\n")


if __name__ == "__main__":
    unittest.main()
