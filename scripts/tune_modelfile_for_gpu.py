#!/usr/bin/env python3
"""Tune an Ollama Modelfile for the current GPU.

This script probes the highest context length (num_ctx) that keeps the model
fully GPU-resident on the current machine. It also removes PARAMETER num_gpu,
which is often mistaken for VRAM size and can force partial CPU offload.

Workflow:
1. Read Modelfile.
2. Build temporary probe models with decreasing num_ctx values.
3. For each probe model, trigger one generation and inspect /api/ps.
4. Select the highest num_ctx with size_vram ~= size.
5. Update the target Modelfile (or print a dry-run preview).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass
class ProbeResult:
    context: int
    size: int
    size_vram: int
    fully_gpu: bool

    @property
    def gpu_ratio(self) -> float:
        if self.size <= 0:
            return 0.0
        return float(self.size_vram) / float(self.size)


def _http_json(host: str, method: str, endpoint: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = Request(url=f"{host.rstrip('/')}{endpoint}", data=body, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    if not raw:
        return {}
    return json.loads(raw)


def _run_ollama_cmd(args: list[str]) -> None:
    completed = subprocess.run(args, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        detail = stderr or stdout or "unknown error"
        raise RuntimeError(f"Command failed ({' '.join(args)}): {detail}")


def _extract_from_model(lines: list[str]) -> str:
    for line in lines:
        m = re.match(r"\s*FROM\s+(.+?)\s*$", line)
        if m:
            return m.group(1).strip()
    raise ValueError("Modelfile is missing a FROM line")


def _extract_num_ctx(lines: list[str], default_ctx: int) -> int:
    for line in lines:
        m = re.match(r"\s*PARAMETER\s+num_ctx\s+(\d+)\s*$", line)
        if m:
            return int(m.group(1))
    return default_ctx


def _looks_like_qwen3(from_model: str) -> bool:
    return "qwen3" in from_model.lower()


def _render_tuned_lines(lines: list[str], *, num_ctx: int) -> list[str]:
    out: list[str] = []
    had_ctx = False

    for line in lines:
        if re.match(r"\s*PARAMETER\s+num_gpu\b", line):
            # num_gpu is layer-count oriented; removing avoids accidental CPU offload caps.
            continue
        if re.match(r"\s*PARAMETER\s+num_ctx\b", line):
            out.append(f"PARAMETER num_ctx {num_ctx}")
            had_ctx = True
            continue
        out.append(line.rstrip("\n"))

    if not had_ctx:
        out.append(f"PARAMETER num_ctx {num_ctx}")

    return out


def _build_context_candidates(start_ctx: int, min_ctx: int, step: int) -> list[int]:
    values: list[int] = []
    ctx = start_ctx
    while ctx >= min_ctx:
        values.append(ctx)
        ctx -= step
    if min_ctx not in values:
        values.append(min_ctx)
    # Keep descending unique order.
    seen: set[int] = set()
    unique: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _create_probe_model(temp_name: str, modelfile_text: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".modelfile", delete=False) as fh:
        fh.write(modelfile_text)
        temp_path = fh.name
    try:
        _run_ollama_cmd(["ollama", "create", temp_name, "-f", temp_path])
    finally:
        Path(temp_path).unlink(missing_ok=True)


def _generate_once(host: str, model_name: str, prompt: str, timeout: float, probe_think_false: bool) -> None:
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "2m",
    }
    if probe_think_false:
        payload["options"] = {"think": False}
    _http_json(
        host,
        "POST",
        "/api/generate",
        payload=payload,
        timeout=timeout,
    )


def _find_loaded_model(host: str, model_name: str) -> dict[str, Any] | None:
    ps = _http_json(host, "GET", "/api/ps")
    models = ps.get("models") or []
    exact = model_name if ":" in model_name else f"{model_name}:latest"
    for model in models:
        name = str(model.get("name", ""))
        if name == exact or name == model_name:
            return model
    return None


def _list_loaded_model_names(host: str) -> list[str]:
    ps = _http_json(host, "GET", "/api/ps")
    models = ps.get("models") or []
    names: list[str] = []
    for model in models:
        name = str(model.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _stop_loaded_models(host: str) -> None:
    for name in _list_loaded_model_names(host):
        subprocess.run(["ollama", "stop", name], capture_output=True, text=True)
    # Give CUDA runtime a brief window to release memory between probes.
    time.sleep(1.0)


def _probe_context(
    *,
    host: str,
    base_lines: list[str],
    context: int,
    probe_think_false: bool,
    prompt: str,
    timeout: float,
    probe_prefix: str,
    stop_loaded_models: bool,
) -> ProbeResult:
    if stop_loaded_models:
        _stop_loaded_models(host)

    probe_name = f"{probe_prefix}-{context}-{int(time.time())}"
    tuned = _render_tuned_lines(base_lines, num_ctx=context)
    _create_probe_model(probe_name, "\n".join(tuned) + "\n")

    try:
        _generate_once(host, probe_name, prompt, timeout, probe_think_false)
        loaded = _find_loaded_model(host, probe_name)
        if not loaded:
            raise RuntimeError(f"Probe model {probe_name} was not visible in /api/ps")
        size = int(loaded.get("size") or 0)
        size_vram = int(loaded.get("size_vram") or 0)
        fully_gpu = size > 0 and size_vram >= size
        return ProbeResult(context=context, size=size, size_vram=size_vram, fully_gpu=fully_gpu)
    finally:
        # Best effort cleanup so probes do not clutter local model list.
        subprocess.run(["ollama", "stop", probe_name], capture_output=True, text=True)
        subprocess.run(["ollama", "rm", probe_name], capture_output=True, text=True)


def _write_modelfile(path: Path, lines: list[str], make_backup: bool) -> Path | None:
    backup_path: Path | None = None
    if make_backup:
        backup_path = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune an Ollama Modelfile for current GPU memory.")
    parser.add_argument(
        "--modelfile",
        default="/home/cds/dev/daily-sync-agent/Modelfile.daily-sync-summarizer",
        help="Path to Modelfile to tune.",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://127.0.0.1:11434",
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--min-ctx",
        type=int,
        default=4096,
        help="Minimum context to test.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2048,
        help="Context decrement step while probing.",
    )
    parser.add_argument(
        "--default-start-ctx",
        type=int,
        default=32768,
        help="Start context if Modelfile has no num_ctx parameter.",
    )
    parser.add_argument(
        "--probe-prefix",
        default="dsa-probe",
        help="Temporary probe model name prefix.",
    )
    parser.add_argument(
        "--prompt",
        default="Say exactly: OK",
        help="Short prompt used to force model load during probing.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout (seconds) for probe generation.",
    )
    parser.add_argument(
        "--set-think-false",
        action="store_true",
        help="Force PARAMETER think false regardless of FROM model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write Modelfile; print what would be changed.",
    )
    parser.add_argument(
        "--keep-loaded-models",
        action="store_true",
        help="Do not stop loaded Ollama models before probes.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup before writing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modelfile_path = Path(args.modelfile).expanduser().resolve()

    if not modelfile_path.exists():
        raise SystemExit(f"Modelfile not found: {modelfile_path}")

    try:
        _http_json(args.ollama_host, "GET", "/api/tags", timeout=10.0)
    except URLError as exc:
        raise SystemExit(f"Cannot connect to Ollama at {args.ollama_host}: {exc}") from exc

    original_lines = modelfile_path.read_text(encoding="utf-8").splitlines()
    from_model = _extract_from_model(original_lines)
    start_ctx = _extract_num_ctx(original_lines, args.default_start_ctx)

    if args.min_ctx <= 0 or args.step <= 0:
        raise SystemExit("--min-ctx and --step must be positive integers")

    candidates = _build_context_candidates(start_ctx, args.min_ctx, args.step)
    probe_think_false = bool(args.set_think_false or _looks_like_qwen3(from_model))

    print(f"Modelfile: {modelfile_path}")
    print(f"FROM: {from_model}")
    print(f"Probe contexts: {candidates}")
    print(f"Probe think=false: {probe_think_false}")
    print("---")

    best_full_gpu: ProbeResult | None = None
    best_ratio: ProbeResult | None = None

    for ctx in candidates:
        result = _probe_context(
            host=args.ollama_host,
            base_lines=original_lines,
            context=ctx,
            probe_think_false=probe_think_false,
            prompt=args.prompt,
            timeout=args.timeout,
            probe_prefix=args.probe_prefix,
            stop_loaded_models=not args.keep_loaded_models,
        )
        print(
            f"ctx={result.context} size={result.size/1024**3:.2f}GiB "
            f"vram={result.size_vram/1024**3:.2f}GiB gpu_ratio={result.gpu_ratio:.2%} "
            f"full_gpu={result.fully_gpu}"
        )

        if best_ratio is None or result.gpu_ratio > best_ratio.gpu_ratio:
            best_ratio = result

        if result.fully_gpu:
            best_full_gpu = result
            break

    if best_full_gpu:
        chosen = best_full_gpu
        reason = "highest tested context with full GPU residency"
    elif best_ratio:
        chosen = best_ratio
        reason = "no fully-GPU context found; selected best GPU ratio"
    else:
        raise SystemExit("No probe results collected")

    tuned_lines = _render_tuned_lines(
        original_lines,
        num_ctx=chosen.context,
    )

    print("---")
    print(f"Chosen num_ctx: {chosen.context} ({reason})")
    print("Will remove PARAMETER num_gpu if present.")
    if probe_think_false:
        print("Probe used options.think=false during benchmarking.")

    if args.dry_run:
        print("\nDry-run: updated Modelfile preview\n")
        print("\n".join(tuned_lines))
        return 0

    backup = _write_modelfile(modelfile_path, tuned_lines, make_backup=not args.no_backup)
    if backup:
        print(f"Backup saved: {backup}")
    print(f"Updated: {modelfile_path}")
    print("Rebuild the model after this change, for example:")
    print(f"  ollama create daily-sync-summary -f {modelfile_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

