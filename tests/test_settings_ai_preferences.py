from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from daily_sync_agent.settings import AppConfig


class SettingsAIPreferencesTests(unittest.TestCase):
    def test_normalized_turns_off_dependent_ai_flags_when_transcription_disabled(self) -> None:
        cfg = AppConfig(
            transcribe_speech=False,
            summarize_transcript=True,
            unload_models_after_task=True,
            diarize_speakers=True,
            identify_speakers=True,
            huggingface_token="hf_abc",
        ).normalized()
        self.assertFalse(cfg.transcribe_speech)
        self.assertFalse(cfg.summarize_transcript)
        self.assertFalse(cfg.unload_models_after_task)
        self.assertFalse(cfg.diarize_speakers)
        self.assertFalse(cfg.identify_speakers)

    def test_normalized_requires_hf_token_for_diarization(self) -> None:
        cfg = AppConfig(
            transcribe_speech=True,
            diarize_speakers=True,
            identify_speakers=True,
            huggingface_token="",
        ).normalized()
        self.assertFalse(cfg.diarize_speakers)
        self.assertFalse(cfg.identify_speakers)

    def test_normalized_falls_back_unknown_summary_mode(self) -> None:
        cfg = AppConfig(summary_mode="custom_mode").normalized()
        self.assertEqual(cfg.summary_mode, "general")

    def test_load_applies_ai_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "transcribe_speech": False,
                        "summarize_transcript": True,
                        "diarize_speakers": True,
                        "identify_speakers": True,
                        "unload_models_after_task": True,
                        "summary_mode": "bad_value",
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(AppConfig, "config_path", return_value=config_path):
                cfg = AppConfig.load()

        self.assertFalse(cfg.transcribe_speech)
        self.assertFalse(cfg.summarize_transcript)
        self.assertFalse(cfg.diarize_speakers)
        self.assertFalse(cfg.identify_speakers)
        self.assertFalse(cfg.unload_models_after_task)
        self.assertEqual(cfg.summary_mode, "general")

    def test_normalized_clamps_speaker_id_quality_knobs(self) -> None:
        cfg = AppConfig(
            speaker_id_similarity_threshold=9.0,
            speaker_id_min_ref_segment_s=-3.0,
            speaker_id_max_ref_segments=99,
        ).normalized()
        self.assertEqual(cfg.speaker_id_similarity_threshold, 1.0)
        self.assertEqual(cfg.speaker_id_min_ref_segment_s, 0.5)
        self.assertEqual(cfg.speaker_id_max_ref_segments, 8)


if __name__ == "__main__":
    unittest.main()

