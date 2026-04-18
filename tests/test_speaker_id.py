from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from daily_sync_agent.ai.speaker_id import (
    identify_speakers_from_profiles,
    load_name_map,
    load_profiles,
    rename_diarization_speakers,
    save_name_map,
    save_profiles,
    select_reference_segments,
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

    def test_select_reference_segments_falls_back_to_best_available(self) -> None:
        diarization = [
            {"start": 0.0, "end": 2.8, "speaker": "Speaker_1"},
            {"start": 1.0, "end": 3.4, "speaker": "Speaker_2"},
            # No clean segment for Speaker_1 (overlap), but fallback should still keep a span.
            {"start": 5.0, "end": 8.5, "speaker": "Speaker_2"},
        ]
        refs = select_reference_segments(
            diarization,
            min_reference_segment_s=2.0,
            max_reference_segments=2,
        )
        self.assertIn("Speaker_1", refs)
        self.assertGreaterEqual(len(refs["Speaker_2"]), 1)

    @patch("daily_sync_agent.ai.speaker_id._speaker_embeddings_from_audio")
    @patch("daily_sync_agent.ai.speaker_id.load_profiles")
    def test_identify_speakers_enforces_one_to_one_profile_assignment(
        self,
        mock_load_profiles,
        mock_local_embeddings,
    ) -> None:
        # Both local speakers are closest to speaker_01; second should not collide onto same profile.
        mock_load_profiles.return_value = {
            "speaker_01": np.array([1.0, 0.0], dtype=np.float32),
        }
        mock_local_embeddings.return_value = {
            "Speaker_A": np.array([0.99, 0.01], dtype=np.float32),
            "Speaker_B": np.array([0.97, 0.03], dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            profiles_path = Path(tmp) / "profiles.npz"
            names_path = Path(tmp) / "speaker_names.txt"
            out = identify_speakers_from_profiles(
                Path("/tmp/fake.flac"),
                [{"start": 0.0, "end": 1.0, "speaker": "Speaker_A"}, {"start": 1.0, "end": 2.0, "speaker": "Speaker_B"}],
                hf_token="hf_token",
                device="cpu",
                profiles_path=profiles_path,
                names_path=names_path,
            )

        self.assertEqual(len(out), 2)
        self.assertNotEqual(out["Speaker_A"], out["Speaker_B"])


if __name__ == "__main__":
    unittest.main()

