"""Transport only: sends a request to an LLM and returns text.

Per CLAUDE.md, LLM calls originate in exactly two places: sensor.py and generate.py.
This file just carries them. Standard library only (no SDK needed).

Local by default: Ollama on the loopback address, with proxy settings ignored so the
request can never leave the machine. Cloud providers are opt-in.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

import config

CALL_COUNT = 0   # every sensor/generator call, including mock ones (tests check this)
_OLLAMA_UP: Optional[bool] = None


class LLMError(RuntimeError):
    pass


def count_call() -> None:
    global CALL_COUNT
    CALL_COUNT += 1


def _local_opener():
    """An opener that never uses HTTP(S)_PROXY: local requests stay local."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _ollama_url(path: str) -> str:
    host = urllib.parse.urlparse(config.OLLAMA_URL).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise LLMError("Ollama URL must be a loopback address (127.0.0.1)")
    return config.OLLAMA_URL.rstrip("/") + path


def ollama_running() -> bool:
    global _OLLAMA_UP
    if _OLLAMA_UP is None:
        try:
            with _local_opener().open(_ollama_url("/api/tags"), timeout=1.5):
                _OLLAMA_UP = True
        except (urllib.error.URLError, OSError, ValueError, LLMError):
            _OLLAMA_UP = False
    return _OLLAMA_UP


def resolve_provider(name: str) -> str:
    if name == "auto":   # never resolves to a cloud provider
        return "ollama" if ollama_running() else "mock"
    return name


def model_for(role: str, provider: str) -> str:
    override = config.SENSOR_MODEL if role == "sensor" else config.GENERATOR_MODEL
    return override or config.DEFAULT_MODELS.get(provider, {}).get(role, "")


def _post(url: str, payload: dict, headers: Dict[str, str], local: bool = False) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"content-type": "application/json", **headers})
    opener = _local_opener() if local else urllib.request.build_opener()
    try:
        with opener.open(req, timeout=config.LLM_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        if local and exc.code == 404:
            raise LLMError(f"Local model is missing. Run: ollama pull {payload.get('model')}") \
                from exc
        raise LLMError(f"HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        if local:
            raise LLMError("Cannot reach local Ollama. Open the Ollama app or run: "
                           "ollama serve") from exc
        raise LLMError(str(exc)) from exc


def complete(provider: str, role: str, system: str, messages: List[dict],
             max_tokens: int, json_mode: bool = False) -> str:
    provider = resolve_provider(provider)
    model = model_for(role, provider)
    if provider == "mock":
        raise LLMError("mock provider has no model")
    if provider == "ollama":
        payload = {"model": model, "stream": False, "keep_alive": config.OLLAMA_KEEP_ALIVE,
                   "messages": [{"role": "system", "content": system}] + messages,
                   "options": {"temperature": config.LLM_TEMPERATURE,
                               "num_predict": max_tokens}}
        if json_mode:
            payload["format"] = "json"
        out = _post(_ollama_url("/api/chat"), payload, {}, local=True)
        text = (out.get("message") or {}).get("content", "")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Local Ollama produced an empty response")
        return text.strip()
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set (add it to .env)")
        out = _post("https://api.anthropic.com/v1/messages",
                    {"model": model, "max_tokens": max_tokens, "system": system,
                     "messages": messages},
                    {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in out.get("content", [])
                       if b.get("type") == "text")
    if provider == "openai_compatible":
        headers = {}
        if os.environ.get("OPENAI_API_KEY"):
            headers["authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
        out = _post(config.OPENAI_BASE_URL.rstrip("/") + "/chat/completions",
                    {"model": model, "max_tokens": max_tokens,
                     "messages": [{"role": "system", "content": system}] + messages},
                    headers)
        try:
            return out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(out)[:200]}") from exc
    raise LLMError(f"unknown provider: {provider}")
