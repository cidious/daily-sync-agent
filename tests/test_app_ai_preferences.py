from __future__ import annotations

import unittest

from daily_sync_agent.app import _ai_preferences_control_state


class AppAIPreferencesStateTests(unittest.TestCase):
    def test_master_toggle_off_disables_all_ai_controls(self) -> None:
        st = _ai_preferences_control_state(
            transcribe_checked=False,
            summarize_checked=True,
            diarize_checked=True,
            hf_token="hf_xxx",
        )
        self.assertFalse(st["ai_enabled"])
        self.assertFalse(st["summarize_enabled"])
        self.assertFalse(st["diarize_enabled"])
        self.assertFalse(st["hf_token_enabled"])
        self.assertFalse(st["identify_enabled"])

    def test_identification_requires_diarization_and_token(self) -> None:
        # Missing token disables identification
        st_no_token = _ai_preferences_control_state(
            transcribe_checked=True,
            summarize_checked=True,
            diarize_checked=True,
            hf_token="   ",
        )
        self.assertFalse(st_no_token["identify_enabled"])

        # With token + diarization on, identification can be enabled
        st_with_token = _ai_preferences_control_state(
            transcribe_checked=True,
            summarize_checked=False,
            diarize_checked=True,
            hf_token="hf_xxx",
        )
        self.assertTrue(st_with_token["identify_enabled"])

    def test_summary_controls_follow_transcribe_and_summarize_checkboxes(self) -> None:
        st = _ai_preferences_control_state(
            transcribe_checked=True,
            summarize_checked=False,
            diarize_checked=False,
            hf_token="",
        )
        self.assertTrue(st["ai_enabled"])
        self.assertFalse(st["summarize_enabled"])


if __name__ == "__main__":
    unittest.main()

