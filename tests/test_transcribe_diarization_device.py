from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from daily_sync_agent.ai.transcribe import _transcribe_once


class TranscribeDiarizationDeviceTests(unittest.TestCase):
    def test_diarization_receives_same_device_as_whisper(self) -> None:
        class FakeWhisperModel:
            def __init__(self, model_size: str, device: str, compute_type: str) -> None:
                self.feature_extractor = types.SimpleNamespace(sampling_rate=16000)

            def transcribe(self, audio, **kwargs):
                seg = types.SimpleNamespace(text="hello", start=0.0, end=0.8)
                return iter([seg]), {"language": "en"}

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            with patch("daily_sync_agent.ai.transcribe.load_audio_ffmpeg_mono_f32", return_value=[0.0]):
                with patch(
                    "daily_sync_agent.ai.diarize.diarize_speakers",
                    return_value={
                        "speakers": [{"start": 0.0, "end": 1.0, "speaker": "Speaker_1"}],
                        "num_speakers_detected": 1,
                    },
                ) as mock_diarize:
                    text, diarization = _transcribe_once(
                        Path("/tmp/fake.flac"),
                        model_size="base",
                        device="cuda",
                        compute_type="float16",
                        diarize=True,
                        hf_token="hf_token",
                    )

        self.assertIn("Speaker_1", text)
        self.assertIsNotNone(diarization)
        self.assertEqual(mock_diarize.call_count, 1)
        self.assertEqual(mock_diarize.call_args.kwargs.get("device"), "cuda")


if __name__ == "__main__":
    unittest.main()

