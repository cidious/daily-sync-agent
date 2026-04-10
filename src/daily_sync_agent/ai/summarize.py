"""Short description via local LLM (Ollama, OpenAI-compatible, llama.cpp HTTP server)."""

from __future__ import annotations

import logging
import subprocess

import httpx

from daily_sync_agent.settings import AppConfig

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You summarize spoken content in exactly one short paragraph. "
    "CRITICAL language rule: write your entire answer ONLY in the same language as the transcript. "
    "Do not translate into English or any other language—match the transcript language only. "
    "Be neutral and concise. If the transcript is empty, say so in that same language."
)

_SYSTEM_DAILY_SCRUM = (
    "You analyze a transcript of a daily scrum / stand-up sync: discussion of work, problems, issues, and questions. "
    "CRITICAL language rule: write your ENTIRE response ONLY in the same language as the transcript—do not translate to English. "
    "Structure the answer with clear section headings; translate those heading titles into the transcript language "
    "(e.g. Russian headings if the transcript is Russian). Use bullet lists where asked.\n\n"
    "Required sections, in order:\n"
    "1) Title of discussion — one line: a short descriptive title for this meeting.\n"
    "2) Date of the event — use the recording timestamp given in the user message if present; "
    "otherwise infer from the transcript or state that the date is unknown.\n"
    "3) Topics — bullet list of the main discussion topics.\n"
    "4) Possible solutions — bullet list of solutions, decisions, or agreements mentioned.\n"
    "5) Tasks — bullet list of concrete action items, tasks, or follow-ups.\n"
    "6) Problems and blockers — bullet list of impediments, blockers, or unresolved problems; if none, say so clearly.\n\n"
    "Be factual; only include items supported by the transcript. If a section has nothing relevant, state that briefly."
)


def _letter_chars(s: str) -> list[str]:
    return [c for c in s if c.isalpha()]


def _cyrillic_ratio(text: str) -> float:
    letters = _letter_chars(text)
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04ff")
    return cyr / len(letters)


def build_summary_user_content(
    transcript: str,
    *,
    summary_mode: str = "general",
    event_date_hint: str = "",
) -> str:
    """Transcript plus language rules; scrum mode adds recording time and structured-output hints."""
    t = transcript.strip()
    if summary_mode == "daily_scrum":
        parts: list[str] = []
        if (event_date_hint or "").strip():
            parts.append(f"Recording start (local time): {event_date_hint.strip()}")
        parts.append(f"Transcript:\n\n{t}")
        block = "\n\n".join(parts) if parts else f"Transcript:\n\n{t}"
        if _cyrillic_ratio(transcript) >= 0.25:
            block += (
                "\n\n[Обязательно] Ответ целиком на русском. Соблюдайте разделы и списки из системной инструкции; "
                "заголовки разделов на русском."
            )
        else:
            block += (
                "\n\n[Required] Follow the section structure from the system instructions. "
                "Write every section in the same language as the transcript."
            )
        return block

    block = f"Transcript:\n\n{t}"
    if _cyrillic_ratio(transcript) >= 0.25:
        block += (
            "\n\n[Обязательно] Транскрипт на русском. Напишите краткое резюме целиком на русском языке. "
            "Не переводите на английский. Только русский."
        )
    else:
        block += (
            "\n\n[Required] Write the summary only in the same language as the transcript above. "
            "Do not switch to English unless the transcript itself is English."
        )
    return block


def _assistant_text_from_message(msg: dict) -> str:
    """Qwen3 and similar models may put text in ``thinking`` / ``reasoning`` when ``content`` is empty."""
    if not isinstance(msg, dict):
        return ""
    for key in ("content", "thinking", "reasoning"):
        part = msg.get(key)
        if isinstance(part, str) and part.strip():
            return part.strip()
    return ""


def _parse_ollama_chat(data: dict) -> str:
    msg = data.get("message") or {}
    content = _assistant_text_from_message(msg) if isinstance(msg, dict) else ""
    if not content and isinstance(data.get("response"), str):
        content = data["response"].strip()
    return content


def _parse_openai_chat(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return _assistant_text_from_message(msg) if isinstance(msg, dict) else ""


def _parse_openai_completion(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("text") or "").strip()


def _parse_llama_cpp_completion(data: dict) -> str:
    return (data.get("content") or data.get("response") or "").strip()


def _http_error_detail(resp: httpx.Response) -> str:
    try:
        t = resp.text.strip()
        return t[:1500] if t else "(empty body)"
    except Exception:
        return "(could not read body)"


def _ollama_model_names(tags_json: dict) -> list[str]:
    return [m["name"] for m in tags_json.get("models", []) if m.get("name")]


def _running_ollama_model_names(ps_json: dict) -> list[str]:
    out: list[str] = []
    for m in ps_json.get("models", []) if isinstance(ps_json, dict) else []:
        name = m.get("name") if isinstance(m, dict) else None
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def _with_ollama_keepalive(payload: dict, *, unload_model_after_task: bool) -> dict:
    if not unload_model_after_task:
        return payload
    p = dict(payload)
    # Ollama keep_alive=0 unloads the model runner after the request.
    p["keep_alive"] = 0
    return p


def resolve_ollama_model(tags_json: dict, configured: str) -> str:
    """Pick a model name that exists locally. Empty ``configured`` → first in list."""
    names = _ollama_model_names(tags_json)
    if not names:
        raise RuntimeError(
            "Ollama has no models. Run `ollama pull <model>` or install a GGUF, then retry."
        )
    want = configured.strip()
    if not want:
        chosen = names[0]
        logger.info("Ollama chat model not set; using first available: %r", chosen)
        return chosen
    if want not in names:
        raise RuntimeError(
            f"Ollama model {want!r} is not installed. Available: {', '.join(names)}. "
            "Set Preferences → Ollama chat model to one of these, or leave it empty to use the first."
        )
    return want


def list_ollama_models(base_url: str, *, timeout_s: float = 10.0) -> list[str]:
    """Names from ``GET /api/tags``; empty list if unreachable or not Ollama."""
    base = base_url.rstrip("/")
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(f"{base}/api/tags")
            if r.status_code != 200:
                return []
            return _ollama_model_names(r.json())
    except Exception as e:
        logger.debug("list_ollama_models failed: %s", e)
        return []


def summarize_text(
    text: str,
    *,
    base_url: str,
    model: str,
    summary_mode: str = "general",
    event_date_hint: str = "",
    timeout_s: float = 900.0,
    unload_model_after_task: bool = False,
) -> str:
    base = base_url.rstrip("/")
    mode = (summary_mode or "general").strip().lower()
    if mode not in ("general", "daily_scrum"):
        mode = "general"
    system = _SYSTEM_DAILY_SCRUM if mode == "daily_scrum" else _SYSTEM
    user_block = build_summary_user_content(
        text,
        summary_mode=mode,
        event_date_hint=event_date_hint,
    )
    full_prompt = f"{system}\n\n{user_block}"
    token_out = 1024 if mode == "daily_scrum" else 512

    read_s = max(120.0, float(timeout_s))
    # Local LLMs often need minutes for long outputs; use an explicit read timeout (not just connect).
    client_timeout = httpx.Timeout(connect=30.0, read=read_s, write=120.0, pool=30.0)
    tags_timeout = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=10.0)

    with httpx.Client(timeout=tags_timeout) as tags_client:
        tags = tags_client.get(f"{base}/api/tags")

    resolved_model = model
    looks_like_ollama = tags.status_code == 200
    if looks_like_ollama:
        logger.debug("GET /api/tags OK — server looks like Ollama")
        try:
            resolved_model = resolve_ollama_model(tags.json(), model)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Could not read Ollama model list: {e}") from e
    elif tags.status_code == 404:
        logger.warning(
            "GET %s/api/tags returned 404 — nothing on this URL looks like Ollama; "
            "summarizer will try several HTTP APIs anyway",
            base,
        )

    with httpx.Client(timeout=client_timeout) as client:
        ollama_chat = (
            "Ollama /api/chat",
            f"{base}/api/chat",
            _with_ollama_keepalive({
                "model": resolved_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_block},
                ],
                "stream": False,
                "options": {"num_predict": token_out, "temperature": 0.3},
            }, unload_model_after_task=unload_model_after_task),
            _parse_ollama_chat,
        )
        ollama_generate = (
            "Ollama /api/generate",
            f"{base}/api/generate",
            _with_ollama_keepalive({
                "model": resolved_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": token_out},
            }, unload_model_after_task=unload_model_after_task),
            lambda d: (d.get("response") or "").strip() if isinstance(d, dict) else "",
        )
        openai_chat = (
            "OpenAI /v1/chat/completions",
            f"{base}/v1/chat/completions",
            {
                "model": resolved_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_block},
                ],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": token_out,
            },
            _parse_openai_chat,
        )
        openai_completion = (
            "OpenAI /v1/completions",
            f"{base}/v1/completions",
            {
                "model": resolved_model,
                "prompt": full_prompt,
                "max_tokens": token_out,
                "temperature": 0.3,
            },
            _parse_openai_completion,
        )
        llama_cpp = (
            "llama.cpp /completion",
            f"{base}/completion",
            {
                "prompt": full_prompt,
                "n_predict": token_out,
                "temperature": 0.3,
                "stream": False,
            },
            _parse_llama_cpp_completion,
        )
        # When the server is Ollama, other routes hit the same process — avoid 5× identical long timeouts.
        if looks_like_ollama:
            attempts: list[tuple[str, str, dict, object]] = [ollama_chat, ollama_generate]
        else:
            attempts = [ollama_chat, openai_chat, ollama_generate, openai_completion, llama_cpp]

        failures: list[str] = []

        for name, url, payload, parser in attempts:
            try:
                r = client.post(url, json=payload)
            except httpx.RequestError as e:
                failures.append(f"{name} ({url}): request error: {e}")
                logger.info("%s failed: %s", name, e)
                continue

            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception as e:
                    failures.append(f"{name}: invalid JSON: {e} body={_http_error_detail(r)}")
                    continue
                if not isinstance(data, dict):
                    failures.append(f"{name}: expected JSON object, got {type(data)}")
                    continue
                out = parser(data)
                if isinstance(out, str) and out.strip():
                    return out.strip()
                logger.warning(
                    "%s returned HTTP 200 but no extractable assistant text (keys=%s). Trying next endpoint.",
                    name,
                    list(data.keys())[:12],
                )
                failures.append(
                    f"{name}: 200 OK but empty content (model may use a different JSON shape; see debug log)"
                )
                continue

            if r.status_code == 404:
                detail = _http_error_detail(r)
                if "not found" in detail.lower() and "model" in detail.lower():
                    raise RuntimeError(
                        f"Ollama rejected the model (404): {detail}. "
                        "Set Preferences → Ollama chat model to a name from `ollama list`, or leave it empty."
                    )
                failures.append(f"{name} ({url}): 404 {detail}")
                logger.info("%s returned 404; trying next endpoint", name)
                continue

            raise RuntimeError(
                f"{name} ({url}): HTTP {r.status_code} {_http_error_detail(r)}"
            )

        if failures:
            msg = (
                f"Could not get a non-empty summary from {base}. Tried: "
                + "; ".join(failures[:8])
            )
            if any("timed out" in f.lower() for f in failures):
                msg += (
                    f" Request timed out — local models can be slow for long summaries; increase "
                    f"`ollama_request_timeout_s` in {AppConfig.config_path()} "
                    f"(read timeout is {read_s:.0f}s per attempt), or use a smaller/faster model."
                )
            else:
                msg += (
                    ". For Russian audio, pick a multilingual chat model in Preferences "
                    "(e.g. a Qwen or Hermes variant you already have), or leave the model empty to use the first in `ollama list`."
                )
            raise RuntimeError(msg)
        raise RuntimeError(
            f"No LLM endpoint responded on {base}. Set Preferences → Ollama URL and model."
        )


def unload_ollama_models(
    base_url: str,
    *,
    preferred_model: str = "",
    stop_cli: bool = False,
    timeout_s: float = 8.0,
) -> None:
    """Best-effort unload of Ollama model runners from memory."""
    base = (base_url or "").rstrip("/")
    if not base:
        return
    logger.debug(
        "Low-VRAM cleanup: unload_ollama_models base=%s stop_cli=%s preferred_model_set=%s",
        base,
        stop_cli,
        bool(preferred_model.strip()),
    )

    names: set[str] = set()
    if preferred_model.strip():
        names.add(preferred_model.strip())

    timeout = httpx.Timeout(connect=5.0, read=timeout_s, write=timeout_s, pool=5.0)
    try:
        with httpx.Client(timeout=timeout) as client:
            try:
                ps = client.get(f"{base}/api/ps")
                if ps.status_code == 200:
                    names.update(_running_ollama_model_names(ps.json()))
            except Exception as e:
                logger.debug("Could not query running Ollama models: %s", e)

            for name in sorted(names):
                try:
                    r = client.post(
                        f"{base}/api/generate",
                        json={"model": name, "prompt": "", "stream": False, "keep_alive": 0},
                    )
                    if r.status_code not in (200, 404):
                        logger.info("Ollama unload request for %s returned HTTP %s", name, r.status_code)
                except Exception as e:
                    logger.debug("Ollama unload request failed for %s: %s", name, e)
    except Exception as e:
        logger.debug("Ollama unload client init failed: %s", e)

    if not stop_cli:
        logger.debug("Low-VRAM cleanup: HTTP unload phase finished (no CLI stop requested)")
        return
    for name in sorted(names):
        try:
            subprocess.run(
                ["ollama", "stop", name],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception as e:
            logger.debug("ollama stop failed for %s: %s", name, e)
    logger.debug("Low-VRAM cleanup: HTTP unload and CLI stop phases finished")

