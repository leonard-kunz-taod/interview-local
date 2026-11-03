"""Each stage runs in a separate process so its model memory is released on exit."""
import json
import os
from pathlib import Path
import sys

# Set before library imports; disable telemetry and use local inference.
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def decode(source, target):
    import av
    import soundfile as sf
    samples = 0
    with av.open(str(source)) as container, sf.SoundFile(str(target), "w", samplerate=16000,
            channels=1, subtype="PCM_16") as output:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for chunk in resampler.resample(frame):
                array = chunk.to_ndarray().reshape(-1)
                output.write(array)
                samples += len(array)
        for chunk in resampler.resample(None):
            array = chunk.to_ndarray().reshape(-1)
            output.write(array)
            samples += len(array)
    if not samples:
        raise ValueError("The file contains no audio track.")

def run(stage, source, target):
    from dataclasses import asdict
    from .backends import FasterWhisperTranscriber, CommunityDiarizer, GermanWhisperXAligner, fallback_words
    if stage == "check-access":
        from .models import local_diarization_model
        if local_diarization_model():
            return
        from huggingface_hub import hf_hub_download
        hf_hub_download("pyannote/speaker-diarization-community-1", "config.yaml",
            token=os.environ.get("HF_TOKEN") or None, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
        return
    if stage == "decode":
        decode(source, target)
        return
    if stage == "asr":
        result = FasterWhisperTranscriber().transcribe(source)
    elif stage == "diarize":
        result = asdict(CommunityDiarizer().diarize(source))
    elif stage == "align":
        transcript = json.loads(source.with_name("asr.json").read_text(encoding="utf-8"))
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

if __name__ == "__main__":
    import time
    started = time.monotonic()
    try:
        run(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
        elapsed = time.monotonic() - started
        print(f"Stage {sys.argv[1]} completed in {elapsed:.1f} s", flush=True)
        try:
            import resource
            import platform
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            divisor = 2**30 if platform.system() == "Darwin" else 2**20
            print(f"Peak process RAM (RSS): {rss/divisor:.2f} GiB", flush=True)
        except (ImportError, OSError):
            pass
    except Exception as exc:
        message = str(exc)
        token = os.environ.get("HF_TOKEN")
        if token:
            message = message.replace(token, "[Token redacted]")
        import re
        message = re.sub(r"hf_[A-Za-z0-9]+", "[Token redacted]", message)
        print(f"ERROR ({type(exc).__name__}): {message}", file=sys.stderr, flush=True)
        sys.exit(1)
