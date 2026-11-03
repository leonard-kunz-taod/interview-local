"""Each stage runs in a separate process so its model memory is released on exit."""

import json
import os
import re
import sys
import time
from pathlib import Path

# Set before library imports; disable telemetry and use local inference.
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SAMPLE_RATE = 16000
REDACTED = "[Token redacted]"
HF_TOKEN_PATTERN = r"hf_[A-Za-z0-9]+"


def main() -> None:
    """Run the stage named on the command line; report time, memory or the error."""
    started = time.monotonic()
    try:
        run(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
        elapsed = time.monotonic() - started
        print(f"Stage {sys.argv[1]} completed in {elapsed:.1f} s", flush=True)
        _report_peak_memory()
    except Exception as exc:
        message = _redacted(str(exc))
        print(f"ERROR ({type(exc).__name__}): {message}", file=sys.stderr, flush=True)
        sys.exit(1)


def run(stage: str, source: Path, target: Path) -> None:
    """Run one processing stage; stages with a result write it to target as JSON."""
    from dataclasses import asdict

    from .backends import (
        CommunityDiarizer,
        FasterWhisperTranscriber,
        GermanWhisperXAligner,
    )
    from .models import DIARIZATION_ID, hub_offline, local_diarization_model

    if stage == "check-access":
        if local_diarization_model():
            return
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            DIARIZATION_ID,
            "config.yaml",
            token=os.environ.get("HF_TOKEN") or None,
            local_files_only=hub_offline(),
        )
        return
    if stage == "decode":
        decode(source, target)
        return
    if stage == "asr":
        result = FasterWhisperTranscriber().transcribe(source)
    elif stage == "diarize":
        result = asdict(CommunityDiarizer().diarize(source))
    elif stage == "align":
        transcript = json.loads(
            source.with_name("asr.json").read_text(encoding="utf-8")
        )
        result = GermanWhisperXAligner().align(source, transcript)
    elif stage == "prepare-asr":
        FasterWhisperTranscriber()
        return
    elif stage == "prepare-diarize":
        CommunityDiarizer()
        return
    elif stage == "prepare-align":
        GermanWhisperXAligner()
        import nltk

        if not nltk.download("punkt_tab", quiet=True):
            raise RuntimeError("Could not download NLTK language data.")
        return
    else:
        raise ValueError("Unknown processing stage")
    target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def decode(source: Path, target: Path) -> None:
    """Decode the first audio track to 16 kHz mono 16-bit PCM WAV.

    Raises:
        ValueError: The file contains no audio samples.
    """
    import av
    import soundfile as sf

    samples = 0
    with (
        av.open(str(source)) as container,
        sf.SoundFile(
            str(target), "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16"
        ) as output,
    ):
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for array in _resampled(container, resampler):
            output.write(array)
            samples += len(array)
    if not samples:
        raise ValueError("The file contains no audio track.")


def _resampled(container, resampler):
    """Sample arrays of every decoded frame, then the resampler's remainder."""
    for frame in container.decode(audio=0):
        for chunk in resampler.resample(frame):
            yield chunk.to_ndarray().reshape(-1)
    for chunk in resampler.resample(None):
        yield chunk.to_ndarray().reshape(-1)


def _report_peak_memory() -> None:
    try:
        import platform
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 2**30 if platform.system() == "Darwin" else 2**20
        print(f"Peak process RAM (RSS): {rss / divisor:.2f} GiB", flush=True)
    except (ImportError, OSError):
        pass


def _redacted(message: str) -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        message = message.replace(token, REDACTED)
    return re.sub(HF_TOKEN_PATTERN, REDACTED, message)


if __name__ == "__main__":
    main()
