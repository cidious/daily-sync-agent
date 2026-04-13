from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from daily_sync_agent.ai.summarize import _running_ollama_model_names, _with_ollama_keepalive
from daily_sync_agent.ai.transcribe import _transcribe_once, _wait_for_whisper_vram_release, transcribe_file
from daily_sync_agent.settings import AppConfig


class LowVramModeTests(unittest.TestCase):
    def test_config_default_is_disabled(self) -> None:
        self.assertIs(AppConfig().unload_models_after_task, False)

    def test_transcription_default_is_enabled(self) -> None:
        self.assertIs(AppConfig().transcribe_speech, True)

    def test_running_ollama_model_name_parser(self) -> None:
        names = _running_ollama_model_names(
            {
                "models": [
                    {"name": "qwen2.5:7b"},
                    {"name": "llama3.2"},
                    {"size": 123},
                ]
            }
        )
        self.assertEqual(names, ["qwen2.5:7b", "llama3.2"])

    def test_keepalive_option_is_added_only_when_enabled(self) -> None:
        payload = {"model": "qwen2.5:7b", "stream": False}
        unchanged = _with_ollama_keepalive(payload, unload_model_after_task=False)
        self.assertEqual(unchanged, payload)
        self.assertNotIn("keep_alive", unchanged)

        changed = _with_ollama_keepalive(payload, unload_model_after_task=True)
        self.assertEqual(changed["keep_alive"], 0)
        self.assertNotIn("keep_alive", payload)

    def test_wait_for_whisper_vram_release_returns_immediately_when_usage_already_low(self) -> None:
        with patch("daily_sync_agent.ai.transcribe._current_process_nvidia_vram_mib") as mock_query:
            self.assertTrue(_wait_for_whisper_vram_release(128))
        mock_query.assert_not_called()

    @patch("daily_sync_agent.ai.transcribe.time.sleep")
    @patch(
        "daily_sync_agent.ai.transcribe._current_process_nvidia_vram_mib",
        side_effect=[1024, 96],
    )
    def test_wait_for_whisper_vram_release_polls_until_usage_drops(self, mock_query, mock_sleep) -> None:
        self.assertTrue(_wait_for_whisper_vram_release(2048, timeout_s=1.0, poll_interval_s=0.01))
        self.assertEqual(mock_query.call_count, 2)
        mock_sleep.assert_called_once_with(0.01)

    def test_transcribe_once_waits_for_gpu_cleanup_before_returning(self) -> None:
        class FakeWhisperModel:
            def __init__(self, model_size: str, device: str, compute_type: str) -> None:
                self.feature_extractor = types.SimpleNamespace(sampling_rate=16000)

            def transcribe(self, audio, **kwargs):
                return iter([types.SimpleNamespace(text="hello"), types.SimpleNamespace(text="world")]), {
                    "language": "en"
                }

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            with patch("daily_sync_agent.ai.transcribe.load_audio_ffmpeg_mono_f32", return_value=[0.0]):
                with patch(
                    "daily_sync_agent.ai.transcribe._current_process_nvidia_vram_mib",
                    side_effect=[1536, 96],
                ) as mock_query:
                    with patch("daily_sync_agent.ai.transcribe.gc.collect") as mock_gc:
                        text = _transcribe_once(
                            Path("/tmp/fake.flac"),
                            model_size="base",
                            device="cuda",
                            compute_type="float16",
                            unload_model_after_task=True,
                        )

        self.assertEqual(text, "hello\nworld")
        self.assertEqual(mock_query.call_count, 2)
        mock_gc.assert_called_once()

    def test_transcribe_once_skips_gpu_polling_for_cpu_transcription(self) -> None:
        class FakeWhisperModel:
            def __init__(self, model_size: str, device: str, compute_type: str) -> None:
                self.feature_extractor = types.SimpleNamespace(sampling_rate=16000)

            def transcribe(self, audio, **kwargs):
                return iter([types.SimpleNamespace(text="cpu only")]), {"language": "en"}

        fake_module = types.SimpleNamespace(WhisperModel=FakeWhisperModel)
        with patch.dict(sys.modules, {"faster_whisper": fake_module}):
            with patch("daily_sync_agent.ai.transcribe.load_audio_ffmpeg_mono_f32", return_value=[0.0]):
                with patch("daily_sync_agent.ai.transcribe._current_process_nvidia_vram_mib") as mock_query:
                    text = _transcribe_once(
                        Path("/tmp/fake.flac"),
                        model_size="base",
                        device="cpu",
                        compute_type="int8",
                        unload_model_after_task=True,
                    )

        self.assertEqual(text, "cpu only")
        mock_query.assert_not_called()

    @patch("daily_sync_agent.ai.transcribe._transcribe_once", return_value="ok")
    def test_transcribe_file_logs_success_timing(self, _mock_once) -> None:
        with patch("daily_sync_agent.ai.transcribe.logger.debug") as mock_debug:
            out = transcribe_file(
                Path("/tmp/fake.flac"),
                model_size="small",
                device="auto",
                compute_type="default",
                unload_model_after_task=False,
            )
        self.assertEqual(out, "ok")
        self.assertTrue(
            any(
                "Whisper transcribe_file success" in str(call.args[0])
                for call in mock_debug.call_args_list
                if call.args
            )
        )


if __name__ == "__main__":
    unittest.main()
