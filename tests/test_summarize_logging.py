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


class _FakeSummaryClientOomThenRetry:
    def __init__(self, timeout=None) -> None:
        self.timeout = timeout
        self.calls: list[tuple[str, dict]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url: str, json: dict) -> _FakeResponse:
        self.calls.append((url, json))
        if len(self.calls) == 1:
            return _FakeResponse(
                500,
                {},
                text='{"error":"llama runner process has terminated: cudaMalloc failed: out of memory"}',
            )
        return _FakeResponse(200, {"message": {"content": "retry summary"}})


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

    def test_summarize_text_retries_ollama_with_adaptive_options_after_cuda_oom(self) -> None:
        summary_client = _FakeSummaryClientOomThenRetry()
        with patch(
            "daily_sync_agent.ai.summarize.httpx.Client",
            side_effect=[_FakeTagsClient(), summary_client],
        ):
            with patch("daily_sync_agent.ai.summarize._adaptive_ollama_retry_plan_for_vram", return_value=[{"num_gpu": 0}]):
                with patch("daily_sync_agent.ai.summarize.unload_ollama_models") as mock_unload:
                    out = summarize_text(
                        "hello transcript",
                        base_url="http://127.0.0.1:11434",
                        model="",
                        summary_mode="general",
                        timeout_s=120.0,
                        unload_model_after_task=True,
                    )

        self.assertEqual(out, "retry summary")
        self.assertEqual(len(summary_client.calls), 2)
        first_payload = summary_client.calls[0][1]
        second_payload = summary_client.calls[1][1]
        self.assertNotIn("num_gpu", first_payload.get("options", {}))
        self.assertEqual(second_payload.get("options", {}).get("num_gpu"), 0)
        mock_unload.assert_called_once()


if __name__ == "__main__":
    unittest.main()

