from __future__ import annotations

import unittest

from daily_sync_agent.ai.transcribe import _transcribe_kwargs


class TranscribeOptionTests(unittest.TestCase):
    def test_transcribe_kwargs_enable_anti_hallucination_defaults(self) -> None:
        kwargs = _transcribe_kwargs()
        self.assertEqual(kwargs["beam_size"], 5)
        self.assertIs(kwargs["condition_on_previous_text"], False)
        self.assertIs(kwargs["vad_filter"], True)
        self.assertIs(kwargs["word_timestamps"], True)
        self.assertEqual(kwargs["hallucination_silence_threshold"], 2.0)


if __name__ == "__main__":
    unittest.main()
