import contextlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from interview_local.models import (
    DIARIZATION_ID,
    diarization_checkpoint,
    hub_offline,
    local_diarization_model,
)


def configured(directory):
    return patch.dict(os.environ, {"INTERVIEW_DIARIZATION_MODEL": str(directory)})


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


class LocalDiarizationModelTests(unittest.TestCase):
    def test_configured_complete_folder_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.yaml").touch()
            with configured(directory):
                model = local_diarization_model()
            self.assertEqual(model, Path(directory).resolve())

    def test_configured_folder_without_config_is_refused(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            configured(directory),
            self.assertRaisesRegex(ValueError, "complete community-1"),
        ):
            local_diarization_model()

    def test_configured_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.yaml"
            config.touch()
            with configured(config), self.assertRaises(ValueError):
                local_diarization_model()

    def test_bundled_folder_is_used_when_nothing_is_configured(self):
        with bundled_local_model() as bundled:
            self.assertEqual(local_diarization_model(), bundled)

    def test_hub_is_used_without_a_configured_or_bundled_folder(self):
        with without_local_model():
            self.assertIsNone(local_diarization_model())

    def test_hub_identifier_is_the_checkpoint_without_a_local_folder(self):
        with without_local_model():
            self.assertEqual(diarization_checkpoint(), DIARIZATION_ID)

    def test_configured_folder_is_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.yaml").touch()
            with configured(directory):
                checkpoint = diarization_checkpoint()
            self.assertEqual(checkpoint, str(Path(directory).resolve()))


class HubTests(unittest.TestCase):
    def test_offline_flag_disables_the_hub(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}):
            self.assertTrue(hub_offline())

    def test_hub_is_online_without_the_offline_flag(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}):
            self.assertFalse(hub_offline())


if __name__ == "__main__":
    unittest.main()
