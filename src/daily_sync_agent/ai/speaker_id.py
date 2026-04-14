"""Optional speaker identification with saved embeddings + editable name mapping."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from daily_sync_agent.ai.diarize import _load_audio_as_waveform

logger = logging.getLogger(__name__)

_MIN_REFERENCE_SEGMENT_S = 2.0
_SIMILARITY_THRESHOLD = 0.72


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 0.0 or bn <= 0.0:
        return -1.0
    return float(np.dot(a, b) / (an * bn))


def _is_clean_segment(seg: dict, diarization_speakers: list[dict]) -> bool:
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", 0.0))
    speaker = str(seg.get("speaker", ""))
    if end <= start:
        return False
    for other in diarization_speakers:
        if other is seg:
            continue
        if str(other.get("speaker", "")) == speaker:
            continue
        o_start = float(other.get("start", 0.0))
        o_end = float(other.get("end", 0.0))
        if min(end, o_end) - max(start, o_start) > 0.0:
            return False
    return True


def select_longest_clean_segments(diarization_speakers: list[dict]) -> dict[str, tuple[float, float]]:
    """Pick one longest non-overlapping segment per diarization speaker."""
    best: dict[str, tuple[float, float]] = {}
    for seg in diarization_speakers:
        if not _is_clean_segment(seg, diarization_speakers):
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end - start < _MIN_REFERENCE_SEGMENT_S:
            continue
        speaker = str(seg.get("speaker", ""))
        cur = best.get(speaker)
        if cur is None or (end - start) > (cur[1] - cur[0]):
            best[speaker] = (start, end)
    return best


def _next_profile_id(existing_ids: set[str]) -> str:
    n = 1
    while True:
        candidate = f"speaker_{n:02d}"
        if candidate not in existing_ids:
            return candidate
        n += 1


def load_profiles(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        return {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
    except Exception as e:
        logger.warning("Speaker ID: could not load profiles from %s: %s", path, e)
        return {}


def save_profiles(path: Path, profiles: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not profiles:
        return
    serializable = {k: np.asarray(v, dtype=np.float32) for k, v in profiles.items()}
    np.savez(path, **serializable)


def load_name_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        speaker_id, speaker_name = line.split(":", 1)
        k = speaker_id.strip()
        v = speaker_name.strip()
        if k:
            result[k] = v or k
    return result


def save_name_map(path: Path, names: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# speaker_id: display name",
        "# Edit right-hand values to rename speakers in transcripts.",
    ]
    for speaker_id in sorted(names):
        lines.append(f"{speaker_id}: {names[speaker_id]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_embedder(*, hf_token: str, device: str):
    import importlib

    pyannote_audio = importlib.import_module("pyannote.audio")
    model = pyannote_audio.Model.from_pretrained("pyannote/embedding", token=hf_token.strip())
    inference = pyannote_audio.Inference(model, window="whole")
    if hasattr(inference, "to"):
        import torch

        inference.to(torch.device(device))

    def embed(waveform, sample_rate: int) -> np.ndarray:
        emb = inference({"waveform": waveform, "sample_rate": sample_rate})
        return np.asarray(emb, dtype=np.float32).reshape(-1)

    return embed


def _speaker_embeddings_from_audio(
    audio_path: Path,
    diarization_speakers: list[dict],
    *,
    hf_token: str,
    device: str,
) -> dict[str, np.ndarray]:
    refs = select_longest_clean_segments(diarization_speakers)
    if not refs:
        return {}

    audio = _load_audio_as_waveform(audio_path)
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    embed = _build_embedder(hf_token=hf_token, device=device)

    result: dict[str, np.ndarray] = {}
    for local_speaker, (start, end) in refs.items():
        s = max(0, int(start * sample_rate))
        e = min(waveform.shape[1], int(end * sample_rate))
        if e - s < int(_MIN_REFERENCE_SEGMENT_S * sample_rate):
            continue
        clip = waveform[:, s:e]
        result[local_speaker] = embed(clip, sample_rate)
    return result


def _match_or_create_profile_id(
    emb: np.ndarray,
    profiles: dict[str, np.ndarray],
    *,
    threshold: float = _SIMILARITY_THRESHOLD,
) -> str:
    best_id = ""
    best_score = -1.0
    for profile_id, profile_emb in profiles.items():
        score = _cosine_similarity(emb, profile_emb)
        if score > best_score:
            best_score = score
            best_id = profile_id
    if best_id and best_score >= threshold:
        return best_id
    return _next_profile_id(set(profiles.keys()))


def identify_speakers_from_profiles(
    audio_path: Path,
    diarization_speakers: list[dict],
    *,
    hf_token: str,
    device: str,
    profiles_path: Path,
    names_path: Path,
) -> dict[str, str]:
    """Return mapping from local diarization speaker IDs to display names."""
    if not diarization_speakers:
        return {}

    profiles = load_profiles(profiles_path)
    local_embeddings = _speaker_embeddings_from_audio(
        audio_path,
        diarization_speakers,
        hf_token=hf_token,
        device=device,
    )
    if not local_embeddings:
        return {}

    local_to_profile: dict[str, str] = {}
    for local_speaker, emb in local_embeddings.items():
        profile_id = _match_or_create_profile_id(emb, profiles)
        if profile_id in profiles:
            profiles[profile_id] = ((profiles[profile_id] + emb) / 2.0).astype(np.float32)
        else:
            profiles[profile_id] = emb.astype(np.float32)
        local_to_profile[local_speaker] = profile_id

    save_profiles(profiles_path, profiles)

    names = load_name_map(names_path)
    changed = False
    for profile_id in sorted(local_to_profile.values()):
        if profile_id not in names:
            names[profile_id] = profile_id
            changed = True
    if changed:
        save_name_map(names_path, names)

    local_to_name: dict[str, str] = {}
    for local_speaker, profile_id in local_to_profile.items():
        local_to_name[local_speaker] = names.get(profile_id, profile_id)

    logger.debug(
        "Speaker ID matched speakers=%d profiles=%d names_file=%s profiles_file=%s",
        len(local_to_name),
        len(profiles),
        names_path,
        profiles_path,
    )
    return local_to_name


def rename_diarization_speakers(diarization_speakers: list[dict], local_to_name: dict[str, str]) -> list[dict]:
    if not local_to_name:
        return diarization_speakers
    renamed: list[dict] = []
    for seg in diarization_speakers:
        item = dict(seg)
        speaker = str(seg.get("speaker", ""))
        item["speaker"] = local_to_name.get(speaker, speaker)
        renamed.append(item)
    return renamed

