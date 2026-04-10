from __future__ import annotations

import unittest

from daily_sync_agent.ai.summarize import _running_ollama_model_names, _with_ollama_keepalive
from daily_sync_agent.settings import AppConfig


class LowVramModeTests(unittest.TestCase):
    def test_config_default_is_disabled(self) -> None:
        self.assertIs(AppConfig().unload_models_after_task, False)

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


if __name__ == "__main__":
    unittest.main()
