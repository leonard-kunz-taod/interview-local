"""Where the community-1 diarization model is found: nowhere, or bundled."""

import contextlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


@contextlib.contextmanager
def without_local_model():
    """Neither a configured nor a bundled community-1 folder exists."""
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.dict(os.environ),
        patch("interview_local.models.BUNDLED_DIARIZATION_MODEL", Path(directory)),
    ):
        os.environ.pop("INTERVIEW_DIARIZATION_MODEL", None)
        yield


@contextlib.contextmanager
def bundled_local_model():
    """An unconfigured run finds a complete bundled community-1 folder."""
    with (
        tempfile.TemporaryDirectory() as directory,
        patch.dict(os.environ),
        patch("interview_local.models.BUNDLED_DIARIZATION_MODEL", Path(directory)),
    ):
        os.environ.pop("INTERVIEW_DIARIZATION_MODEL", None)
        (Path(directory) / "config.yaml").touch()
        yield Path(directory).resolve()
