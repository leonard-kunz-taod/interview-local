import contextlib
import importlib
import io
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from pyannote.core import Annotation, Segment
from support.model_folders import without_local_model

from interview_local.backends import (
    CommunityDiarizer,
    FasterWhisperTranscriber,
    GermanWhisperXAligner,
)
from interview_local.core import DiarizationResult, Turn
from interview_local.models import DIARIZATION_ID, WHISPER_MODEL

TOKEN = "hf_secretToken123"
LOADER_LIBRARIES = ("nltk", "onnxruntime", "torch")
SAMPLE_RATE = 16000
GERMAN_SENTENCE_DATA = "tokenizers/punkt_tab/german/"
TORCHCODEC_WARNING = "torchcodec is not installed correctly so decoding will fail."
TORCHCODEC_MODULE = "pyannote.audio.core.io"
SEGMENT = {
    "start": 1.0,
    "end": 2.0,
    "text": "Test 42",
    "words": [
        {
            "start": 1.0,
            "end": 1.3,
            "text": "Test",
            "timing": "whisper",
            "probability": 0.9,
        },
        {
            "start": 1.3,
            "end": 2.0,
            "text": "42",
            "timing": "whisper",
            "probability": 0.9,
        },
    ],
}


def setUpModule():
    # Torch cannot be imported twice; patched module tables would drop it again.
    for name in LOADER_LIBRARIES:
        importlib.import_module(name)


@contextlib.contextmanager
def torch_threads_restored():
    threads = torch.get_num_threads()
    try:
        yield
    finally:
        torch.set_num_threads(threads)


class RecordingAlign:
    """whisperx.alignment.align: records each call, answers with fixed words."""

    def __init__(self, word_segments):
        self.word_segments = word_segments
        self.calls = []

    def __call__(self, transcript, model, metadata, audio, device, **options):
        self.calls.append((transcript, model, metadata, audio, device, options))
        return {"word_segments": self.word_segments}


def aligned(word_segments, segments=(SEGMENT,), recording=None):
    """Align segments in a three-second recording with a stubbed model."""
    return aligned_with(RecordingAlign(word_segments), segments, recording)[0]


def aligned_with(align, segments=(SEGMENT,), recording=None):
    aligner = GermanWhisperXAligner.__new__(GermanWhisperXAligner)
    aligner.model = "alignment model"
    aligner.metadata = {"language": "de"}
    alignment = SimpleNamespace(align=align)
    samples = np.zeros(3 * SAMPLE_RATE) if recording is None else recording
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "test.wav"
        sf.write(audio, samples, SAMPLE_RATE, subtype="FLOAT")
        with patch.dict(sys.modules, {"whisperx.alignment": alignment}):
            return aligner.align(audio, {"segments": list(segments)}), align


class FakePipeline:
    """Records how the diarization pipeline was loaded and configured.

    Like pyannote, it needs a float32 (channel, time) waveform; it splits the
    recording evenly among the requested speakers, and its regular
    diarization adds a turn in which the last speaker talks over the first.
    """

    def __init__(self, checkpoint, token):
        self.checkpoint = checkpoint
        self.token = token
        self.device = None

    @classmethod
    def from_pretrained(cls, checkpoint, token):
        return cls(checkpoint, token)

    def to(self, device):
        self.device = str(device)

    def __call__(self, recording, num_speakers=None):
        waveform = recording["waveform"]
        if waveform.dtype != torch.float32 or waveform.dim() != 2:
            raise ValueError("Expected a float32 (channel, time) waveform.")
        duration = waveform.shape[1] / recording["sample_rate"]
        speakers = num_speakers or 1
        share = duration / speakers
        exclusive = Annotation()
        for i in range(speakers):
            exclusive[Segment(i * share, (i + 1) * share)] = f"SPEAKER_{i:02d}"
        regular = exclusive.copy()
        regular[Segment(0, share / 2), "overlap"] = f"SPEAKER_{speakers - 1:02d}"
        return SimpleNamespace(
            exclusive_speaker_diarization=exclusive, speaker_diarization=regular
        )


class TorchCodecImport(SimpleNamespace):
    """pyannote.audio whose import warns about TorchCodec like the real one."""

    def __init__(self, category=UserWarning, message=TORCHCODEC_WARNING, module=None):
        super().__init__()
        self.warning = (category, message, module or TORCHCODEC_MODULE)

    @property
    def Pipeline(self):
        category, message, module = self.warning
        warnings.warn_explicit(message, category, "io.py", 1, module=module)
        return FakePipeline


class FakeOnnxRuntime(SimpleNamespace):
    def __init__(self):
        super().__init__(telemetry_enabled=True)

    def disable_telemetry_events(self):
        self.telemetry_enabled = False


class FakeWhisperModel:
    """faster_whisper.WhisperModel: records how it was loaded and decodes with."""

    def __init__(self, *args, **options):
        self.loaded_with = (args, options)
        self.decoded_with = None

    def transcribe(self, *args, **options):
        self.decoded_with = (args, options)
        info = SimpleNamespace(language="de")
        return [], info


def loaded_diarizer(environment, pyannote_audio=None):
    pyannote_audio = pyannote_audio or SimpleNamespace(Pipeline=FakePipeline)
    with (
        patch.dict(sys.modules, {"pyannote.audio": pyannote_audio}),
        patch.dict(os.environ, environment),
        contextlib.redirect_stdout(io.StringIO()),
        torch_threads_restored(),
    ):
        return CommunityDiarizer()


def shown_import_warnings(pyannote_audio):
    with warnings.catch_warnings(record=True) as caught, without_local_model():
        warnings.simplefilter("always")
        loaded_diarizer({}, pyannote_audio)
    return [str(w.message) for w in caught]


def known_sentence_data(resource):
    if resource != GERMAN_SENTENCE_DATA:
        raise LookupError(resource)
    return resource


def loaded_aligner(environment, find=known_sentence_data):
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


def loaded_transcriber(environment=None, cores=12):
    runtime = FakeOnnxRuntime()
    faster_whisper = SimpleNamespace(WhisperModel=FakeWhisperModel)
    with (
        patch.dict(
            sys.modules, {"faster_whisper": faster_whisper, "onnxruntime": runtime}
        ),
        patch.dict(os.environ, environment or {"HF_HUB_OFFLINE": "0"}),
        patch("os.cpu_count", return_value=cores),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        return FasterWhisperTranscriber(), runtime


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


def mono_recording(directory, seconds):
    audio = Path(directory) / "decoded.wav"
    sf.write(audio, np.zeros(int(seconds * SAMPLE_RATE)), SAMPLE_RATE)
    return audio


class TranscriberLoadingTests(unittest.TestCase):
    def test_whisper_is_loaded_for_int8_cpu_inference_with_one_worker(self):
        transcriber, _ = loaded_transcriber()
        self.assertEqual(
            transcriber.model.loaded_with,
            (
                (WHISPER_MODEL,),
                {
                    "device": "cpu",
                    "compute_type": "int8",
                    "cpu_threads": 6,
                    "num_workers": 1,
                    "use_auth_token": False,
                    "local_files_only": False,
                },
            ),
        )

    def test_whisper_uses_at_most_six_threads(self):
        for case, cores, threads in (
            ("many cores", 12, 6),
            ("few cores", 4, 4),
            ("unknown core count", None, 2),
        ):
            with self.subTest(case):
                transcriber, _ = loaded_transcriber(cores=cores)
                options = transcriber.model.loaded_with[1]
                self.assertEqual(options["cpu_threads"], threads)

    def test_offline_whisper_is_loaded_from_the_local_cache_only(self):
        transcriber, _ = loaded_transcriber({"HF_HUB_OFFLINE": "1"})
        self.assertTrue(transcriber.model.loaded_with[1]["local_files_only"])

    def test_loading_whisper_disables_runtime_telemetry(self):
        _, runtime = loaded_transcriber()
        self.assertFalse(runtime.telemetry_enabled)


class TranscriptionTests(unittest.TestCase):
    def test_german_speech_is_decoded_deterministically_with_word_timestamps(self):
        transcriber, _ = loaded_transcriber()
        with contextlib.redirect_stdout(io.StringIO()):
            transcriber.transcribe(Path("interview.wav"))
        self.assertEqual(
            transcriber.model.decoded_with,
            (
                ("interview.wav",),
                {
                    "language": "de",
                    "task": "transcribe",
                    "beam_size": 5,
                    "word_timestamps": True,
                    "vad_filter": False,
                    "condition_on_previous_text": True,
                    "temperature": 0.0,
                },
            ),
        )

    def test_segments_keep_their_whisper_word_timestamps(self):
        word = SimpleNamespace(start=0.0, end=0.5, word=" Hallo", probability=0.75)
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
                                "probability": 0.75,
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
            ).pipeline
            expected = str(Path(directory).resolve())
        self.assertEqual((pipeline.checkpoint, pipeline.token), (expected, False))

    def test_hub_model_is_loaded_with_the_environment_token(self):
        with without_local_model():
            pipeline = loaded_diarizer({"HF_TOKEN": TOKEN}).pipeline
        self.assertEqual((pipeline.checkpoint, pipeline.token), (DIARIZATION_ID, TOKEN))

    def test_hub_model_without_a_token_uses_the_cached_login(self):
        with without_local_model():
            pipeline = loaded_diarizer({"HF_TOKEN": ""}).pipeline
        self.assertIsNone(pipeline.token)

    def test_pipeline_runs_on_the_cpu_one_batch_at_a_time(self):
        with without_local_model():
            pipeline = loaded_diarizer({}).pipeline
        self.assertEqual(
            (
                pipeline.device,
                pipeline.segmentation_batch_size,
                pipeline.embedding_batch_size,
            ),
            ("cpu", 1, 1),
        )

    def test_diarization_uses_at_most_four_threads(self):
        with without_local_model(), torch_threads_restored():
            pyannote_audio = SimpleNamespace(Pipeline=FakePipeline)
            with (
                patch.dict(sys.modules, {"pyannote.audio": pyannote_audio}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                CommunityDiarizer()
            threads = torch.get_num_threads()
        self.assertEqual(threads, 4)

    def test_known_torchcodec_import_warning_is_silenced(self):
        self.assertEqual(shown_import_warnings(TorchCodecImport()), [])

    def test_other_warnings_from_pyannote_audio_io_are_still_shown(self):
        other = TorchCodecImport(message="Sample rate mismatch.")
        self.assertEqual(shown_import_warnings(other), ["Sample rate mismatch."])

    def test_same_text_as_another_warning_category_is_still_shown(self):
        deprecation = TorchCodecImport(category=DeprecationWarning)
        self.assertEqual(shown_import_warnings(deprecation), [TORCHCODEC_WARNING])

    def test_same_text_from_another_module_is_still_shown(self):
        elsewhere = TorchCodecImport(module="somewhere.else")
        self.assertEqual(shown_import_warnings(elsewhere), [TORCHCODEC_WARNING])


class DiarizationTests(unittest.TestCase):
    def test_two_speakers_are_found_in_the_whole_recording(self):
        with without_local_model(), tempfile.TemporaryDirectory() as directory:
            diarizer = loaded_diarizer({})
            result = diarizer.diarize(mono_recording(directory, seconds=2))
        exclusive = [Turn(0.0, 1.0, "SPEAKER_00"), Turn(1.0, 2.0, "SPEAKER_01")]
        overlap = Turn(0.0, 0.5, "SPEAKER_01")
        self.assertEqual(result, DiarizationResult(exclusive, [overlap, *exclusive]))


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

    def test_alignment_uses_at_most_four_threads(self):
        with torch_threads_restored():
            loaded_aligner({"HF_HUB_OFFLINE": "0"})
            threads = torch.get_num_threads()
        self.assertEqual(threads, 4)


class AlignmentTests(unittest.TestCase):
    def test_each_segment_is_aligned_on_the_cpu_against_its_own_audio(self):
        recording = np.linspace(-0.5, 0.5, 3 * SAMPLE_RATE)
        segment = dict(SEGMENT, start=0.0, end=1.0)
        words = [{"word": "Test", "start": 0.0, "end": 0.3}, {"word": "42"}]
        _, align = aligned_with(RecordingAlign(words), [segment], recording)
        transcript, model, metadata, audio, device, options = align.calls[0]
        expected_audio = recording[:SAMPLE_RATE].astype(np.float32)
        self.assertEqual(
            (transcript, model, metadata, device, options, audio.dtype),
            (
                [{"start": 0.0, "end": 1.0, "text": "Test 42"}],
                "alignment model",
                {"language": "de"},
                "cpu",
                {"return_char_alignments": False},
                np.float32,
            ),
        )
        np.testing.assert_array_equal(audio, expected_audio)

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

    def test_aligned_and_estimated_words_carry_no_whisper_probability(self):
        result = aligned([{"word": "Test", "start": 0.0, "end": 0.3}, {"word": "42"}])
        self.assertEqual(
            [(w["timing"], w["probability"]) for w in result],
            [("aligned", None), ("estimated", None)],
        )

    def test_aligned_word_with_only_a_start_time_is_estimated(self):
        result = aligned(
            [{"word": "Test", "start": 0.0, "end": 0.3}, {"word": "42", "start": 0.4}]
        )
        self.assertEqual(
            (result[1]["start"], result[1]["end"], result[1]["timing"]),
            (1.4, 2.0, "estimated"),
        )

    def test_aligned_words_are_shifted_by_their_segment_start(self):
        words = [{"word": "Test", "start": 0.1, "end": 0.3}, {"word": "42"}]
        result = aligned(words)
        self.assertEqual(
            [(w["text"], w["start"], w["end"]) for w in result],
            [("Test", 1.1, 1.3), ("42", 1.0, 2.0)],
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
