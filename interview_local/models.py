"""Resolve an optional complete local diarization model without Hub access."""
import os
from pathlib import Path

DIARIZATION_ID = "pyannote/speaker-diarization-community-1"


def local_diarization_model():
    configured = os.environ.get("INTERVIEW_DIARIZATION_MODEL")
    candidate = (Path(configured).expanduser() if configured else
                 Path(__file__).resolve().parent.parent / "models" / "community-1")
    if candidate.is_dir() and (candidate / "config.yaml").is_file():
        return candidate.resolve()
    if configured:
        raise ValueError("INTERVIEW_DIARIZATION_MODEL must point to a complete community-1 folder with config.yaml.")
    return None


def diarization_checkpoint():
    local = local_diarization_model()
    return str(local) if local else DIARIZATION_ID
