"""Resolve where speech models come from: a local folder or the Hugging Face Hub."""

import os
from pathlib import Path

DIARIZATION_ID = "pyannote/speaker-diarization-community-1"
WHISPER_MODEL = "large-v3"
BUNDLED_DIARIZATION_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "community-1"
)


def local_diarization_model() -> Path | None:
    """The complete local community-1 folder, or None when the Hub is used.

    Raises:
        ValueError: INTERVIEW_DIARIZATION_MODEL names an incomplete folder.
    """
    configured = os.environ.get("INTERVIEW_DIARIZATION_MODEL")
    candidate = (
        Path(configured).expanduser() if configured else BUNDLED_DIARIZATION_MODEL
    )
    if (candidate / "config.yaml").is_file():
        return candidate.resolve()
    if configured:
        raise ValueError(
            "INTERVIEW_DIARIZATION_MODEL must point to a complete community-1 "
            "folder with config.yaml."
        )
    return None


def diarization_checkpoint() -> str:
    """The local community-1 folder if present, otherwise its Hub ID."""
    local = local_diarization_model()
    return str(local) if local else DIARIZATION_ID


def hub_offline() -> bool:
    """Whether Hub downloads are disabled (HF_HUB_OFFLINE=1)."""
    return os.environ.get("HF_HUB_OFFLINE") == "1"
