"""Tests for speaker diarization with Pyannote."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as _np


# ---------------------------------------------------------------------------
# Minimal torch stub so tests run without the optional torch/cuda packages.
# The stub must be registered before any test class is instantiated so that
# @patch("torch.cuda.is_available") can resolve "torch.cuda" from sys.modules.
# ---------------------------------------------------------------------------
def _install_fake_torch() -> None:
    if "torch" in sys.modules:
        return

    class _FakeTensor:
        """Thin numpy wrapper exposing the torch.Tensor interface used in diarize.py."""

        def __init__(self, arr) -> None:
            self._arr = _np.asarray(arr)
            self.shape = self._arr.shape

        def unsqueeze(self, dim: int) -> "_FakeTensor":
            return _FakeTensor(_np.expand_dims(self._arr, dim))

    _fake_cuda = types.ModuleType("torch.cuda")
    _fake_cuda.is_available = lambda: False  # default; individual tests override via @patch

    _fake_torch = types.ModuleType("torch")
    _fake_torch.cuda = _fake_cuda
    _fake_torch.from_numpy = _FakeTensor
    _fake_torch.device = lambda d: d  # return device string as-is

    sys.modules["torch"] = _fake_torch
    sys.modules["torch.cuda"] = _fake_cuda


_install_fake_torch()

from daily_sync_agent.ai.diarize import merge_diarization_with_transcript


class TestDiarizationMerge(unittest.TestCase):
    """Test merging diarization segments with transcript text."""

    def test_merge_empty_diarization(self) -> None:
        """With no diarization speakers, return plain transcript."""
        segments = [
            {"text": "Hello world", "start": 0.0, "end": 2.0},
            {"text": "How are you", "start": 2.1, "end": 4.0},
        ]
        result = merge_diarization_with_transcript(segments, [])
        self.assertEqual(result, "Hello world\nHow are you")

    def test_merge_with_speaker_labels(self) -> None:
        """With speaker labels, prepend speaker ID to each segment."""
        segments = [
            {"text": "Good morning", "start": 0.0, "end": 1.5},
            {"text": "Hi there", "start": 1.6, "end": 2.5},
            {"text": "How are you", "start": 2.6, "end": 4.0},
        ]
        diarization = [
            {"start": 0.0, "end": 1.8, "speaker": "Speaker_1"},
            {"start": 1.9, "end": 4.1, "speaker": "Speaker_2"},
        ]
        result = merge_diarization_with_transcript(segments, diarization)
        lines = result.split("\n")
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("[Speaker_1]"))
        self.assertIn("Good morning", lines[0])
        self.assertIn("[Speaker_2]", lines[1])
        self.assertIn("Hi there", lines[1])

    def test_merge_unknown_speaker_when_no_overlap(self) -> None:
        """When segment doesn't overlap with any diarization, omit speaker label."""
        segments = [
            {"text": "Mystery text", "start": 10.0, "end": 11.0},
        ]
        diarization = [
            {"start": 0.0, "end": 2.0, "speaker": "Speaker_1"},
        ]
        result = merge_diarization_with_transcript(segments, diarization)
        self.assertEqual(result, "Mystery text")

    def test_merge_skips_empty_text_segments(self) -> None:
        """Empty segments are filtered out before merging."""
        segments = [
            {"text": "", "start": 0.0, "end": 0.5},
            {"text": "Hello", "start": 0.5, "end": 1.5},
            {"text": "   ", "start": 1.5, "end": 2.0},
        ]
        diarization = [
            {"start": 0.0, "end": 2.5, "speaker": "Speaker_1"},
        ]
        result = merge_diarization_with_transcript(segments, diarization)
        self.assertIn("Hello", result)
        self.assertEqual(len([line for line in result.split("\n") if line.strip()]), 1)

    def test_merge_prefers_speaker_with_max_overlap(self) -> None:
        segments = [
            {"text": "handoff", "start": 1.0, "end": 2.4},
        ]
        diarization = [
            {"start": 0.9, "end": 1.6, "speaker": "Speaker_1"},
            {"start": 1.5, "end": 2.5, "speaker": "Speaker_2"},
        ]
        # Overlap with Speaker_1 = 0.6s, with Speaker_2 = 0.9s.
        result = merge_diarization_with_transcript(segments, diarization)
        self.assertEqual(result, "[Speaker_2] handoff")

    def test_merge_cleans_invalid_and_merges_tiny_same_speaker_gaps(self) -> None:
        segments = [
            {"text": "hello", "start": 0.05, "end": 1.95},
        ]
        diarization = [
            {"start": 0.0, "end": 1.0, "speaker": "Speaker_1"},
            {"start": 1.08, "end": 2.0, "speaker": "Speaker_1"},
            {"start": 3.0, "end": 2.9, "speaker": "Speaker_2"},  # invalid, ignored
        ]
        result = merge_diarization_with_transcript(segments, diarization)
        self.assertEqual(result, "[Speaker_1] hello")


class TestDiarizationDeviceSelection(unittest.TestCase):
    @patch("torch.cuda.is_available", return_value=True)
    def test_resolve_device_auto_prefers_cuda_when_available(self, _mock_cuda: MagicMock) -> None:
        from daily_sync_agent.ai.diarize import _resolve_diarization_device

        self.assertEqual(_resolve_diarization_device("auto"), "cuda")

    @patch("torch.cuda.is_available", return_value=False)
    def test_resolve_device_auto_falls_back_to_cpu(self, _mock_cuda: MagicMock) -> None:
        from daily_sync_agent.ai.diarize import _resolve_diarization_device

        self.assertEqual(_resolve_diarization_device("auto"), "cpu")

    @patch("torch.cuda.is_available", return_value=False)
    def test_resolve_device_cuda_falls_back_to_cpu_when_unavailable(self, _mock_cuda: MagicMock) -> None:
        from daily_sync_agent.ai.diarize import _resolve_diarization_device

        self.assertEqual(_resolve_diarization_device("cuda"), "cpu")

    def test_place_pipeline_on_device_uses_to_when_supported(self) -> None:
        from daily_sync_agent.ai.diarize import _place_pipeline_on_device

        pipeline = MagicMock()
        _place_pipeline_on_device(pipeline, "cuda")
        pipeline.to.assert_called_once()


class TestDiarizeOutputCompatibility(unittest.TestCase):
    def test_annotation_from_diarize_output_supports_wrapper_attr(self) -> None:
        from daily_sync_agent.ai.diarize import _annotation_from_diarize_output

        annotation = MagicMock()
        annotation.itertracks = MagicMock()
        wrapped = SimpleNamespace(speaker_diarization=annotation)

        self.assertIs(_annotation_from_diarize_output(wrapped), annotation)

    @patch("daily_sync_agent.ai.diarize._load_audio_as_waveform")
    @patch("daily_sync_agent.ai.diarize._import_pyannote_pipeline")
    def test_diarize_speakers_accepts_wrapper_output(self, mock_import_pipeline: MagicMock, mock_load_audio: MagicMock) -> None:
        from daily_sync_agent.ai.diarize import diarize_speakers

        mock_load_audio.return_value = {"waveform": "x", "sample_rate": 16000}

        segment = SimpleNamespace(start=0.0, end=1.25)
        annotation = MagicMock()
        annotation.itertracks.return_value = [(segment, "track-0", "Speaker_1")]

        pipeline_instance = MagicMock()
        pipeline_instance.return_value = SimpleNamespace(speaker_diarization=annotation)
        mock_import_pipeline.return_value.from_pretrained.return_value = pipeline_instance

        out = diarize_speakers(Path("/fake/audio.flac"), hf_token="hf_123")

        self.assertEqual(out["num_speakers_detected"], 1)
        self.assertEqual(out["speakers"][0]["speaker"], "Speaker_1")


class TestPyannoteImportHandling(unittest.TestCase):
    @patch("daily_sync_agent.ai.diarize.importlib.import_module")
    def test_import_pyannote_pipeline_missing_package(self, mock_import: MagicMock) -> None:
        mock_import.side_effect = ImportError("missing")
        from daily_sync_agent.ai.diarize import _import_pyannote_pipeline

        with self.assertRaises(RuntimeError) as ctx:
            _import_pyannote_pipeline()
        self.assertIn("Pyannote not installed", str(ctx.exception))

    @patch("daily_sync_agent.ai.diarize.importlib.import_module")
    def test_import_pyannote_pipeline_runtime_dependency_failure(self, mock_import: MagicMock) -> None:
        mock_import.side_effect = OSError("libnppicc.so.13: cannot open shared object file")
        from daily_sync_agent.ai.diarize import _import_pyannote_pipeline

        with self.assertRaises(RuntimeError) as ctx:
            _import_pyannote_pipeline()
        self.assertIn("Pyannote failed to initialize", str(ctx.exception))


class TestLoadAudioWaveform(unittest.TestCase):
    """Test in-memory 16kHz mono waveform loading for diarization."""

    @patch("daily_sync_agent.ai.diarize.subprocess.run")
    def test_load_audio_returns_waveform_dict(self, mock_run: MagicMock) -> None:
        """Successful ffmpeg decode produces a {waveform, sample_rate} dict."""
        import numpy as np

        # Simulate 1 second of silence at 16kHz, 16-bit PCM.
        pcm_bytes = np.zeros(16000, dtype=np.int16).tobytes()
        mock_run.return_value = MagicMock(returncode=0, stdout=pcm_bytes, stderr=b"")

        from daily_sync_agent.ai.diarize import _load_audio_as_waveform

        result = _load_audio_as_waveform(Path("/fake/audio.flac"))
        self.assertIn("waveform", result)
        self.assertIn("sample_rate", result)
        self.assertEqual(result["sample_rate"], 16000)
        self.assertEqual(result["waveform"].shape[0], 1)      # mono channel dim
        self.assertEqual(result["waveform"].shape[1], 16000)   # 1 second

        # Verify ffmpeg was called with -ar 16000 -ac 1
        cmd = mock_run.call_args[0][0]
        self.assertIn("-ar", cmd)
        self.assertIn("16000", cmd)
        self.assertIn("-ac", cmd)
        self.assertIn("1", cmd)

    @patch("daily_sync_agent.ai.diarize.subprocess.run")
    def test_load_audio_ffmpeg_failure(self, mock_run: MagicMock) -> None:
        """ffmpeg failure raises RuntimeError."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=b"",
            stderr=b"ffmpeg error: no such file",
        )
        from daily_sync_agent.ai.diarize import _load_audio_as_waveform

        with self.assertRaises(RuntimeError) as ctx:
            _load_audio_as_waveform(Path("/fake/missing.flac"))
        self.assertIn("ffmpeg failed", str(ctx.exception))

    @patch("daily_sync_agent.ai.diarize.subprocess.run")
    def test_load_audio_empty_output(self, mock_run: MagicMock) -> None:
        """Empty ffmpeg output raises RuntimeError."""
        mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")

        from daily_sync_agent.ai.diarize import _load_audio_as_waveform

        with self.assertRaises(RuntimeError) as ctx:
            _load_audio_as_waveform(Path("/fake/empty.flac"))
        self.assertIn("no audio data", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

