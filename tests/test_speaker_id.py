from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from daily_sync_agent.ai.speaker_id import (
    load_name_map,
    load_profiles,
    rename_diarization_speakers,
    save_name_map,
    save_profiles,
    select_longest_clean_segments,
)


class SpeakerIdTests(unittest.TestCase):
    def test_select_longest_clean_segments_uses_non_overlap(self) -> None:
        diarization = [
            {"start": 0.0, "end": 3.5, "speaker": "Speaker_1"},
            {"start": 1.0, "end": 2.0, "speaker": "Speaker_2"},
            {"start": 4.0, "end": 7.2, "speaker": "Speaker_1"},
            {"start": 8.0, "end": 11.0, "speaker": "Speaker_2"},
        ]
        refs = select_longest_clean_segments(diarization)
        self.assertEqual(refs["Speaker_1"], (4.0, 7.2))
        self.assertEqual(refs["Speaker_2"], (8.0, 11.0))

    def test_profile_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.npz"
            save_profiles(path, {"speaker_01": np.array([0.1, 0.2], dtype=np.float32)})
            out = load_profiles(path)
            self.assertIn("speaker_01", out)
            self.assertEqual(out["speaker_01"].dtype, np.float32)

    def test_name_map_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speaker_names.txt"
            save_name_map(path, {"speaker_01": "Alice", "speaker_02": "Bob"})
            out = load_name_map(path)
            self.assertEqual(out["speaker_01"], "Alice")
            self.assertEqual(out["speaker_02"], "Bob")

    def test_rename_diarization_speakers(self) -> None:
        diarization = [
            {"start": 0.0, "end": 1.0, "speaker": "Speaker_1"},
            {"start": 1.0, "end": 2.0, "speaker": "Speaker_2"},
        ]
        renamed = rename_diarization_speakers(diarization, {"Speaker_1": "Alice"})
        self.assertEqual(renamed[0]["speaker"], "Alice")
        self.assertEqual(renamed[1]["speaker"], "Speaker_2")


if __name__ == "__main__":
    unittest.main()

