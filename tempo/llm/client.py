"""A small client for AMD Token Factory (OpenAI-compatible chat API running on AMD GPUs).

Plain Python standard library only (urllib). No extra packages, so nothing for Windows
Application Control to block, and it works with any OpenAI-compatible server later
(your own vLLM on the AMD Developer Cloud, for example): change two settings, nothing else.

Settings (environment variables, or a `.env` file in the project folder that is NOT committed):
    RADEON_API_KEY    your team key (starts with rc-). Never commit it, never print it.
    RADEON_BASE_URL   default https://developer.amd.com.cn/radeon/api/v1
    TEMPO_MODEL       default DeepSeek-V4-Flash  (call `models()` to see what is available today)

Token Factory limits (from its manual): about 20 requests/minute per account, 8 at once, a
daily spend cap, no SLA, and "do not proxy or resell the API". So this client throttles itself,
retries 429/502/503 with backoff, and the app must only use the model for its fixed jobs.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://developer.amd.com.cn/radeon/api/v1"
DEFAULT_MODEL = "DeepSeek-V4-Flash"


class LLMError(RuntimeError):
    """Anything that went wrong talking to the model. Messages never contain the API key."""


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Read KEY=VALUE lines into the environment. Existing environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_json_text(text: str) -> dict:
    """Parse a model reply as JSON, tolerating ```json fences and chatter around the object."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("the model did not return JSON")
        try:
            obj = json.loads(t[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"the model returned broken JSON ({e.msg})") from None
    if not isinstance(obj, dict):
        raise LLMError("the model returned JSON that is not an object")
    return obj


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 timeout: float = 120.0, max_retries: int = 5, rpm: float | None = 20.0,
                 reasoning_effort: str | None = None, trust_env_proxy: bool = True, sleep=time.sleep):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._api_key = api_key or ""
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.reasoning_effort = reasoning_effort
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_ok = 0.0
        self.calls = 0
        self._opener = (urllib.request.build_opener() if trust_env_proxy
                        else urllib.request.build_opener(urllib.request.ProxyHandler({})))

    @classmethod
    def from_env(cls, **kw) -> "LLMClient":
        load_dotenv()
        key = os.environ.get("RADEON_API_KEY", "").strip()
        if not key:
            raise LLMError("RADEON_API_KEY is not set. Put it in a .env file in the project folder "
                           "(RADEON_API_KEY=rc-...) or in your environment.")
        return cls(base_url=os.environ.get("RADEON_BASE_URL"), api_key=key, model=os.environ.get("TEMPO_MODEL"),
                   reasoning_effort=os.environ.get("TEMPO_REASONING") or None, **kw)

    def __repr__(self) -> str:
        return f"LLMClient(base_url={self.base_url!r}, model={self.model!r}, api_key=<hidden>)"

    # ---- low level ----
    def _throttle(self) -> None:
        if not self.min_interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_ok - now
            self._next_ok = max(now, self._next_ok) + self.min_interval
        if wait > 0:
            self._sleep(wait)

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = self.base_url + path
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                   "Accept": "application/json", "User-Agent": "tempo-ai/0.1"}
        last = "unknown error"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.calls += 1
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            try:
                with self._opener.open(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = self._error_detail(e)
                if e.code in (429, 502, 503) and attempt < self.max_retries:
                    self._sleep(self._retry_delay(e, attempt))
                    last = f"HTTP {e.code}: {detail}"
                    continue
                hint = {401: " (check RADEON_API_KEY)", 403: " (account may still be in review)"}.get(e.code, "")
                raise LLMError(f"HTTP {e.code}: {detail}{hint}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = f"network problem: {getattr(e, 'reason', e)}"
                if attempt < self.max_retries:
                    self._sleep(self._retry_delay(None, attempt))
                    continue
            except json.JSONDecodeError:
                raise LLMError("the server returned something that is not JSON") from None
        raise LLMError(f"gave up after {self.max_retries + 1} attempts ({last})")

    @staticmethod
    def _error_detail(e: urllib.error.HTTPError) -> str:
        try:
            raw = e.read().decode("utf-8", "replace")
            obj = json.loads(raw)
            err = obj.get("error") or obj.get("detail", {}).get("error") or obj.get("detail") or obj
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return (msg or raw)[:300]
        except Exception:  # noqa: BLE001
            return (e.reason or "error")[:300] if isinstance(e.reason, str) else "error"

    @staticmethod
    def _retry_delay(e: urllib.error.HTTPError | None, attempt: int) -> float:
        if e is not None:
            ra = e.headers.get("Retry-After") if e.headers else None
            if ra:
                try:
                    return float(ra) + random.uniform(0, 0.5)
                except ValueError:
                    pass
        return min(60.0, 2.0 ** attempt) * random.uniform(0.7, 1.3)

    # ---- API ----
    def models(self) -> list[str]:
        """Model ids available right now (the catalog changes, so ask instead of hard-coding)."""
        obj = self._request("GET", "/models")
        items = obj.get("data", obj.get("models", [])) if isinstance(obj, dict) else obj
        ids = [m["id"] if isinstance(m, dict) and "id" in m else m for m in (items if isinstance(items, list) else [])]
        return [i for i in ids if isinstance(i, str)]

    def chat(self, messages: list[dict], max_tokens: int = 1800, temperature: float = 0.2,
             json_mode: bool = False) -> str:
        body: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                      "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        obj = self._request("POST", "/chat/completions", body)
        try:
            choice = obj["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("unexpected reply shape from the server") from None
        if not content.strip():
            if choice.get("finish_reason") == "length":
                raise LLMError("empty reply: thinking used up max_tokens. Raise max_tokens or set "
                               "TEMPO_REASONING=low")
            raise LLMError("the model returned an empty reply")
        return content

    def chat_json(self, messages: list[dict], max_tokens: int = 1800, temperature: float = 0.2) -> dict:
        """Ask for a JSON object and parse it. One automatic retry if the reply is not valid JSON."""
        try:
            return parse_json_text(self.chat(messages, max_tokens, temperature, json_mode=True))
        except LLMError as first:
            if "JSON" not in str(first):
                raise
            nudge = messages + [{"role": "user", "content": "Your last reply was not valid JSON. "
                                                           "Reply again with ONLY the JSON object."}]
            return parse_json_text(self.chat(nudge, max_tokens, temperature, json_mode=True))
