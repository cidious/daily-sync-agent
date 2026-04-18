from __future__ import annotations

import unittest

from daily_sync_agent.ai.summarize import build_summary_user_content


class SummarizePromptStructureTests(unittest.TestCase):
    def test_daily_scrum_user_block_requires_owner_and_due_fields(self) -> None:
        out = build_summary_user_content(
            "Alice will finish API docs tomorrow. Bob will review.",
            summary_mode="daily_scrum",
            event_date_hint="2026-04-19 10:00",
        )
        self.assertIn("Recording start (local time): 2026-04-19 10:00", out)
        self.assertIn("Action items must include task + owner + due/time hint", out)

    def test_general_user_block_mentions_actions_and_owners_when_present(self) -> None:
        out = build_summary_user_content(
            "Ivan will fix the bug today.",
            summary_mode="general",
        )
        self.assertIn("Write the summary only in the same language", out)
        self.assertIn("mention key decisions, action items, and owners", out)


if __name__ == "__main__":
    unittest.main()

