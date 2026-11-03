"""Model adapters. Import only inside the short-lived worker."""

import os
from pathlib import Path

from .core import DiarizationResult, Turn, segment_words
from .models import (
    WHISPER_MODEL,
    diarization_checkpoint,
    hub_offline,
    local_diarization_model,
)

_ALIGNMENT = "facebook/wav2vec2-base-10k-voxpopuli-ft-de"


class FasterWhisperTranscriber:
    def __init__(self):
        _disable_runtime_telemetry()
        from faster_whisper import WhisperModel

        print(
            "Loading Whisper large-v3 / downloading if needed (several GB) …",
            flush=True,
        )
        self.model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=min(6, os.cpu_count() or 2),
            num_workers=1,
            use_auth_token=False,
            local_files_only=hub_offline(),
        )

    def transcribe(self, audio: Path) -> dict:
        segments, info = self.model.transcribe(
            str(audio),
            language="de",
            task="transcribe",
            beam_size=5,
            word_timestamps=True,
            vad_filter=False,
            condition_on_previous_text=True,
            temperature=0.0,
        )
        result = []
        for s in segments:
            print(f"Transcribed through {s.end:.0f} seconds", flush=True)
            words = [
                {"start": w.start, "end": w.end, "text": w.word, "timing": "whisper"}
                for w in s.words or []
            ]
            result.append(
                {"start": s.start, "end": s.end, "text": s.text, "words": words}
            )
        return {"segments": result, "language": info.language}


class CommunityDiarizer:
    def __init__(self):
        _disable_runtime_telemetry()
        import warnings

        import torch

        # We pass a waveform; pyannote's optional TorchCodec decoder
        # is not used. Suppress only its known import warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\s*torchcodec is not installed correctly.*",
                category=UserWarning,
                module=r"pyannote\.audio\.core\.io",
            )
            from pyannote.audio import Pipeline
        torch.set_num_threads(4)
        print("Loading community-1 / downloading if needed …", flush=True)
        token = (
            False if local_diarization_model() else os.environ.get("HF_TOKEN") or None
        )
        self.pipeline = Pipeline.from_pretrained(diarization_checkpoint(), token=token)
        self.pipeline.to(torch.device("cpu"))
        self.pipeline.segmentation_batch_size = 1
        self.pipeline.embedding_batch_size = 1

    def diarize(self, audio: Path) -> DiarizationResult:
        import numpy as np
        import soundfile as sf
        import torch

        # The complete, untrimmed recording after PCM decoding.
        data, rate = sf.read(str(audio), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(np.ascontiguousarray(data.T))
        output = self.pipeline(
            {"waveform": waveform, "sample_rate": rate}, num_speakers=2
        )
        return DiarizationResult(
            _turns(output.exclusive_speaker_diarization),
            _turns(output.speaker_diarization),
        )


class GermanWhisperXAligner:
    def __init__(self):
        import torch
        from whisperx.alignment import load_align_model

        torch.set_num_threads(4)
        if hub_offline():
            import nltk

            try:
                nltk.data.find("tokenizers/punkt_tab/german/")
            except LookupError:
                raise RuntimeError(
                    "NLTK language data is missing offline. "
                    "Run --prepare --align with internet access first."
                ) from None
        print("Loading German alignment model / downloading if needed …", flush=True)
        self.model, self.metadata = load_align_model(
            language_code="de",
            device="cpu",
            model_name=_ALIGNMENT,
            model_cache_only=hub_offline(),
        )

    def align(self, audio: Path, transcript: dict) -> list[dict]:
        import soundfile as sf
        from whisperx.alignment import align

        words = []
        with sf.SoundFile(str(audio)) as recording:
            rate = recording.samplerate
            for segment in transcript["segments"]:
                # Load one ASR segment instead of the entire interview into alignment memory.
                start = max(0, int(segment["start"] * rate))
                end = min(len(recording), int(segment["end"] * rate))
                recording.seek(start)
                chunk = recording.read(end - start, dtype="float32")
                if not len(chunk):
                    words.extend(segment_words(segment, "whisper_fallback"))
                    continue
                duration = len(chunk) / rate
                aligned = align(
                    [{"start": 0.0, "end": duration, "text": segment["text"]}],
                    self.model,
                    self.metadata,
                    chunk,
                    "cpu",
                    return_char_alignments=False,
                )
                aligned_words = aligned.get("word_segments", [])
                # Preserve all words, including numbers and unalignable characters.
                original = "".join(segment["text"].split())
                recovered = "".join(w["word"] for w in aligned_words)
                if recovered != original:
                    words.extend(segment_words(segment, "whisper_fallback"))
                    continue
                words.extend(_timed_words(aligned_words, start / rate, duration))
                print(f"Aligned through {segment['end']:.0f} seconds", flush=True)
        return words


def _disable_runtime_telemetry():
    import onnxruntime

    onnxruntime.disable_telemetry_events()


def _turns(annotation) -> list[Turn]:
    return [
        Turn(float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _timed_words(
    aligned_words: list[dict], offset: float, duration: float
) -> list[dict]:
    """Shift segment-relative alignments; untimed words span the whole segment."""
    return [
        {
            "start": w.get("start", 0.0) + offset,
            "end": w.get("end", duration) + offset,
            "text": w["word"],
            "timing": "aligned" if "start" in w and "end" in w else "estimated",
        }
        for w in aligned_words
    ]
