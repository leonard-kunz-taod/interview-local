import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from interview_local.pipeline import worker

TOKEN = "hf_secretToken123"


def stereo_recording(path):
    sf.write(path, np.zeros((8000, 2)), 8000)


class WorkerProcessTests(unittest.TestCase):
    def test_decoded_audio_is_16khz_mono_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            target = Path(directory) / "decoded.wav"
            stereo_recording(source)
            worker("decode", source, target, status=lambda _: None)
            info = sf.info(target)
        self.assertEqual(
            (info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16")
        )

    def test_completed_stage_reports_its_duration(self):
        messages = []
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            stereo_recording(source)
            worker("decode", source, Path(directory) / "out.wav", messages.append)
        self.assertTrue(
            any(m.startswith("Stage decode completed in") for m in messages)
        )

    def test_unknown_stage_fails_with_the_worker_detail(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "Unknown processing stage"),
        ):
            worker("unknown", Path(directory), Path(directory), lambda _: None)

    def test_token_is_redacted_from_failure_details(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / f"{TOKEN}.wav"
            with self.assertRaises(RuntimeError) as failure:
                worker("decode", missing, missing, lambda _: None, TOKEN)
        self.assertNotIn(TOKEN, str(failure.exception))


if __name__ == "__main__":
    unittest.main()
