from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from daily_sync_agent.ai.pipeline import run_transcribe_and_summarize
from daily_sync_agent.settings import AppConfig


class OptionalSummaryPipelineTests(unittest.TestCase):
    def test_config_default_keeps_summarization_enabled(self) -> None:
        self.assertIs(AppConfig().summarize_transcript, True)

    @patch("daily_sync_agent.ai.pipeline.summarize_text")
    @patch("daily_sync_agent.ai.pipeline.transcribe_file", return_value="hello world")
    def test_pipeline_skips_summary_when_disabled(self, mock_transcribe, mock_summarize) -> None:
        cfg = AppConfig(summarize_transcript=False)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            transcript_path, summary_path = run_transcribe_and_summarize(Path("/tmp/fake.flac"), out_dir, cfg)

            self.assertEqual(transcript_path, out_dir / "transcript.txt")
            self.assertIsNone(summary_path)
            self.assertTrue(transcript_path.is_file())
            self.assertEqual(transcript_path.read_text(encoding="utf-8"), "hello world\n")
            mock_transcribe.assert_called_once()
            mock_summarize.assert_not_called()
            self.assertFalse((out_dir / "summary.txt").exists())

    @patch("daily_sync_agent.ai.pipeline.summarize_text", return_value="short summary")
    @patch("daily_sync_agent.ai.pipeline.transcribe_file", return_value="hello world")
    def test_pipeline_passes_low_vram_flag_through_transcribe_and_summary(
        self,
        mock_transcribe,
        mock_summarize,
    ) -> None:
        cfg = AppConfig(unload_models_after_task=True)
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            transcript_path, summary_path = run_transcribe_and_summarize(Path("/tmp/fake.flac"), out_dir, cfg)
            transcript_text = transcript_path.read_text(encoding="utf-8")
            summary_text = summary_path.read_text(encoding="utf-8")

        self.assertEqual(transcript_text, "hello world\n")
        self.assertEqual(summary_text, "short summary\n")
        self.assertTrue(mock_transcribe.call_args.kwargs["unload_model_after_task"])
        self.assertTrue(mock_summarize.call_args.kwargs["unload_model_after_task"])


if __name__ == "__main__":
    unittest.main()

