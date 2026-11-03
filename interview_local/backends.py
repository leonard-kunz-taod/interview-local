"""Model adapters. Import only inside the short-lived worker."""
from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
from .core import Turn, DiarizationResult

if TYPE_CHECKING:
    import numpy as np
    from soundfile import SoundFile

WHISPER = "large-v3"
DIARIZATION = "pyannote/speaker-diarization-community-1"
ALIGNMENT = "facebook/wav2vec2-base-10k-voxpopuli-ft-de"

def disable_runtime_telemetry():
    import onnxruntime
    onnxruntime.disable_telemetry_events()

class FasterWhisperTranscriber:
    def __init__(self):
        disable_runtime_telemetry()
        import os
        from faster_whisper import WhisperModel
        print("Loading Whisper large-v3 / downloading if needed (several GB) …", flush=True)
        self.model = WhisperModel(WHISPER, device="cpu", compute_type="int8",
                                  cpu_threads=min(6, os.cpu_count() or 2), num_workers=1, use_auth_token=False,
                                  local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")

    def transcribe(self, audio: Path) -> dict:
        segments, info = self.model.transcribe(str(audio), language="de", task="transcribe",
                beam_size=5, word_timestamps=True, vad_filter=False,
                condition_on_previous_text=True, temperature=0.0)
        result = []
        for s in segments:
            print(f"Transcribed through {s.end:.0f} seconds", flush=True)
            result.append(dict(start=s.start, end=s.end, text=s.text,
                words=[dict(start=w.start, end=w.end, text=w.word, timing="whisper",
                           probability=float(w.probability)) for w in s.words or []]))
        return dict(segments=result, language=info.language)

class CommunityDiarizer:
    def __init__(self):
        disable_runtime_telemetry()
        import os
        import torch
        import warnings
        # We pass a waveform; pyannote's optional TorchCodec decoder
        # is not used. Suppress only its known import warning.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly.*",
                                    category=UserWarning, module=r"pyannote\.audio\.core\.io")
            from pyannote.audio import Pipeline
        torch.set_num_threads(4)
        print("Loading community-1 / downloading if needed …", flush=True)
        from .models import diarization_checkpoint, local_diarization_model
        self.pipeline = Pipeline.from_pretrained(diarization_checkpoint(),
            token=False if local_diarization_model() else os.environ.get("HF_TOKEN") or None)
        self.pipeline.to(torch.device("cpu"))
        self.pipeline.segmentation_batch_size = 1
        self.pipeline.embedding_batch_size = 1

    def diarize(self, audio: Path) -> DiarizationResult:
        import numpy as np
        import torch
        import soundfile as sf
        # The complete, untrimmed recording after PCM decoding.
        data, rate = sf.read(str(audio), dtype="float32", always_2d=True)
        output = self.pipeline({"waveform": torch.from_numpy(np.ascontiguousarray(data.T)),
                                "sample_rate": rate}, num_speakers=2)
        def turns(annotation):
            return [Turn(float(turn.start), float(turn.end), str(speaker))
                    for turn, _, speaker in annotation.itertracks(yield_label=True)]
        return DiarizationResult(turns(output.exclusive_speaker_diarization),
                                 turns(output.speaker_diarization))

class GermanWhisperXAligner:
    def __init__(self):
        import torch
        from whisperx.alignment import load_align_model
        torch.set_num_threads(4)
        import os
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            import nltk
            try:
                nltk.data.find("tokenizers/punkt_tab/german/")
            except LookupError:
                raise RuntimeError("NLTK language data is missing offline. Run --prepare --align with internet access first.") from None
        print("Loading German alignment model / downloading if needed …", flush=True)
        self.model, self.metadata = load_align_model(language_code="de", device="cpu", model_name=ALIGNMENT,
            model_cache_only=os.environ.get("HF_HUB_OFFLINE") == "1")

    def align(self, audio: Path, transcript: dict) -> list[dict]:
        import soundfile as sf
        words = []
        with sf.SoundFile(str(audio)) as recording:
            for segment in transcript["segments"]:
                words.extend(self._align_segment(recording, segment))
        return words

    def _align_segment(self, recording: SoundFile, segment: dict) -> list[dict]:
        from whisperx.alignment import align
        chunk, offset = read_segment(recording, segment)
        if not len(chunk):
            return fallback_words(segment)
        duration = len(chunk) / recording.samplerate
        aligned = align([dict(start=0., end=duration, text=segment["text"])],
                        self.model, self.metadata, chunk, "cpu", return_char_alignments=False)
        aligned_words = aligned["word_segments"]
        # Preserve all words, including numbers and unalignable characters.
        if not keeps_every_character(aligned_words, segment["text"]):
            return fallback_words(segment)
        print(f"Aligned through {segment['end']:.0f} seconds", flush=True)
        return [timed_word(w, offset, duration) for w in aligned_words]

def read_segment(recording: SoundFile, segment: dict) -> tuple[np.ndarray, float]:
    """Load one ASR segment instead of the entire interview into alignment memory.

    Returns:
        The segment's samples and its start in seconds.
    """
    rate = recording.samplerate
    start = max(0, int(segment["start"] * rate))
    end = min(len(recording), int(segment["end"] * rate))
    recording.seek(start)
    return recording.read(end - start, dtype="float32"), start / rate

def keeps_every_character(aligned_words: list[dict], text: str) -> bool:
    return "".join(w["word"] for w in aligned_words) == "".join(text.split())

def timed_word(aligned_word: dict, offset: float, duration: float) -> dict:
    """An aligned word in recording time; a word the aligner could not time spans the whole segment."""
    timed = "start" in aligned_word and "end" in aligned_word
    return dict(start=aligned_word.get("start", 0.) + offset, end=aligned_word.get("end", duration) + offset,
                text=aligned_word["word"], timing="aligned" if timed else "estimated", probability=None)

def fallback_words(segment: dict) -> list[dict]:
    if segment["words"]:
        return [dict(w, timing="whisper_fallback") for w in segment["words"]]
    return [dict(start=segment["start"], end=segment["end"], text=segment["text"], timing="estimated",
                 probability=None)]
