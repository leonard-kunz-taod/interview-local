"""Each stage runs in a separate process so its model memory is released on exit."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .redaction import redacted

if TYPE_CHECKING:
    from av.container import InputContainer

# The libraries read these when imported, so they must be set first.
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SAMPLE_RATE = 16000


def main() -> None:
    """Run the stage named on the command line; report time, memory or the error.

    Expected failures are reported as one redacted line and exit code 1; any
    other exception propagates with its traceback.
    """
    from av.error import FFmpegError

    started = time.monotonic()
    try:
        run(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
        elapsed = time.monotonic() - started
        print(f"Stage {sys.argv[1]} completed in {elapsed:.1f} s", flush=True)
        _report_peak_memory()
    except (ValueError, RuntimeError, OSError, FFmpegError) as exc:
        message = redacted(str(exc), os.environ.get("HF_TOKEN"))
        print(f"ERROR ({type(exc).__name__}): {message}", file=sys.stderr, flush=True)
        sys.exit(1)


def run(stage: str, source: Path, target: Path) -> None:
    """Run one processing stage; stages with a result write it to target as JSON."""
    if stage not in STAGES:
        raise ValueError("Unknown processing stage")
    STAGES[stage](source, target)


def check_access(source: Path, target: Path) -> None:
    """Verify Hub access to community-1 unless a complete local copy exists."""
    from .models import DIARIZATION_ID, hub_offline, local_diarization_model

    if local_diarization_model():
        return
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        DIARIZATION_ID,
        "config.yaml",
        token=os.environ.get("HF_TOKEN") or None,
        local_files_only=hub_offline(),
    )


def decode(source: Path, target: Path) -> None:
    """Decode the first audio track to 16 kHz mono 16-bit PCM WAV.

    Raises:
        ValueError: The file has no audio track, or its track has no samples.
    """
    import av

    with av.open(str(source)) as container:
        _require_audio_track(container)
        samples = _write_mono_pcm(container, target)
    if not samples:
        raise ValueError("The audio track contains no samples.")


def _require_audio_track(container: InputContainer) -> None:
    if not container.streams.audio:
        raise ValueError("The file contains no audio track.")


def _write_mono_pcm(container: InputContainer, target: Path) -> int:
    """Write the first audio track to target; return the number of samples."""
    import av
    import soundfile as sf

    samples = 0
    with (
        sf.SoundFile(
            str(target),
            "w",
            samplerate=SAMPLE_RATE,
            channels=1,
            subtype="PCM_16",  # pragma: no mutate -- PCM_16 is soundfile's default WAV subtype, and subtype names are case-insensitive
        ) as output
    ):
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for array in _resampled(container, resampler):
            output.write(array)
            # pragma: no mutate start -- only whether the count is zero is read, and PyAV never yields an empty chunk
            samples += len(array)
            # pragma: no mutate end
    return samples


def _resampled(container, resampler):
    """Sample arrays of every decoded frame, then the resampler's remainder."""
    for frame in container.decode(audio=0):
        for chunk in resampler.resample(frame):
            # pragma: no mutate start -- NumPy infers the size for any negative value
            yield chunk.to_ndarray().reshape(-1)
            # pragma: no mutate end
    for chunk in resampler.resample(None):
        # pragma: no mutate start -- NumPy infers the size for any negative value
        yield chunk.to_ndarray().reshape(-1)
        # pragma: no mutate end


def transcribe(source: Path, target: Path) -> None:
    from .backends import FasterWhisperTranscriber

    _save_result(target, FasterWhisperTranscriber().transcribe(source))


def diarize(source: Path, target: Path) -> None:
    from dataclasses import asdict

    from .backends import CommunityDiarizer

    _save_result(target, asdict(CommunityDiarizer().diarize(source)))


def align(source: Path, target: Path) -> None:
    """Align the asr.json transcript saved next to the decoded audio."""
    from .backends import GermanWhisperXAligner

    transcript = json.loads(source.with_name("asr.json").read_text(encoding="utf-8"))
    _save_result(target, GermanWhisperXAligner().align(source, transcript))


def prepare_transcriber(source: Path, target: Path) -> None:
    """Download or verify the speech recognition model; nothing is saved."""
    from .backends import FasterWhisperTranscriber

    FasterWhisperTranscriber()


def prepare_diarizer(source: Path, target: Path) -> None:
    """Download or verify the diarization model; nothing is saved."""
    from .backends import CommunityDiarizer

    CommunityDiarizer()


def prepare_aligner(source: Path, target: Path) -> None:
    """Download or verify the alignment model and its language data."""
    from .backends import GermanWhisperXAligner

    GermanWhisperXAligner()
    import nltk

    if not nltk.download("punkt_tab", quiet=True):
        raise RuntimeError("Could not download NLTK language data.")


def _save_result(target: Path, result) -> None:
    # pragma: no mutate start -- results are only read back through json.loads, which decodes escapes alike; codec names are case-insensitive
    target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    # pragma: no mutate end


def _report_peak_memory() -> None:
    try:
        import platform
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 2**30 if platform.system() == "Darwin" else 2**20
        print(f"Peak process RAM (RSS): {rss / divisor:.2f} GiB", flush=True)
    except (ImportError, OSError):
        pass


STAGES = {
    "check-access": check_access,
    "decode": decode,
    "asr": transcribe,
    "diarize": diarize,
    "align": align,
    "prepare-asr": prepare_transcriber,
    "prepare-diarize": prepare_diarizer,
    "prepare-align": prepare_aligner,
}


if __name__ == "__main__":
    main()
