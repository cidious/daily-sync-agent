from __future__ import annotations

import unittest
from unittest.mock import patch

from daily_sync_agent.ai.summarize import summarize_text


class _FakeResponse:
    def __init__(self, status_code: int, data: dict, text: str = "") -> None:
        self.status_code = status_code
        self._data = data
        self.text = text

    def json(self) -> dict:
        return self._data


class _FakeTagsClient:
    def __init__(self, timeout=None) -> None:
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def get(self, url: str) -> _FakeResponse:
        return _FakeResponse(200, {"models": [{"name": "qwen2.5:7b"}]})


class _FakeSummaryClient:
    def __init__(self, timeout=None) -> None:
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url: str, json: dict) -> _FakeResponse:
        return _FakeResponse(200, {"message": {"content": "short summary"}})


class SummarizeLoggingTests(unittest.TestCase):
    @patch("daily_sync_agent.ai.summarize.logger.debug")
    def test_summarize_text_logs_start_and_finish_timing(self, mock_debug) -> None:
        with patch(
            "daily_sync_agent.ai.summarize.httpx.Client",
            side_effect=[_FakeTagsClient(), _FakeSummaryClient()],
        ):
            out = summarize_text(
                "hello transcript",
                base_url="http://127.0.0.1:11434",
                model="",
                summary_mode="general",
                timeout_s=120.0,
                unload_model_after_task=False,
            )

        self.assertEqual(out, "short summary")
        messages = [str(call.args[0]) for call in mock_debug.call_args_list if call.args]
        self.assertTrue(any("Summary generation started" in msg for msg in messages))
        self.assertTrue(any("Summary generation finished" in msg for msg in messages))


if __name__ == "__main__":
    unittest.main()

