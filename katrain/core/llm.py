"""OpenAI-compatible chat client.

Speaks the de-facto standard ``POST {base_url}/chat/completions`` protocol that
OpenAI, Azure OpenAI, DeepSeek, Ollama, LM Studio, vLLM and most other providers
expose, so one client covers all of them. Uses ``urllib3`` directly (already a
KaTrain dependency) instead of pulling in a vendor SDK.
"""

import json
import threading

import urllib3

from katrain.core.constants import OUTPUT_DEBUG, OUTPUT_ERROR

DEFAULT_TIMEOUT = 60.0
RETRIES = 1


class LLMError(Exception):
    """A failed chat completion request."""


class LLMClient:
    """Minimal, thread-safe OpenAI-compatible chat client."""

    def __init__(self, base_url, api_key, model, timeout=DEFAULT_TIMEOUT):
        base_url = (base_url or "").strip().rstrip("/")
        if not base_url:
            raise LLMError("Base URL is empty")
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            raise LLMError("Base URL must start with http:// or https://")
        self.url = base_url + "/chat/completions"
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout = timeout
        self.http = urllib3.PoolManager(num_pools=1, retries=RETRIES, cert_reqs="CERT_REQUIRED")

    def _headers(self):
        # Full header set per request: per-request headers fully replace pool
        # headers in urllib3, so Content-Type must be included here too.
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def chat(self, messages, temperature=None, max_tokens=None):
        """Send a chat request, return the assistant message text.

        ``messages`` is the standard ``[{"role": ..., "content": ...}, ...]`` list.
        Raises :class:`LLMError` on connection or API errors.
        """
        body = {"model": self.model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        try:
            response = self.http.request(
                "POST",
                self.url,
                body=json.dumps(body).encode("utf-8"),
                headers=self._headers(),
                timeout=self.timeout,
            )
        except urllib3.exceptions.HTTPError as exc:
            raise LLMError(f"Connection failed: {exc}") from exc

        try:
            data = json.loads(response.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError(f"Invalid response from server (HTTP {response.status}): {response.data[:200]!r}") from exc
        if response.status != 200:
            message = data.get("error", {}).get("message") or data.get("message") or f"HTTP {response.status}"
            raise LLMError(f"{message}")
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected response format: {json.dumps(data)[:200]}") from exc


class LLMConfig:
    """One named LLM configuration (the unit the settings popup edits)."""

    DEFAULTS = {"config_name": "", "base_url": "", "api_key": "", "model_name": ""}
    KEYS = list(DEFAULTS)

    def __init__(self, **kwargs):
        for key in self.DEFAULTS:
            value = kwargs.get(key, "") or ""
            setattr(self, key, str(value).strip().rstrip("/") if key == "base_url" else str(value).strip())

    @classmethod
    def from_config(cls, config_dict):
        return cls(**{k: config_dict.get(k, "") or "" for k in cls.KEYS})

    def to_config(self):
        return {k: getattr(self, k) for k in self.KEYS}

    def __eq__(self, other):
        return isinstance(other, LLMConfig) and self.to_config() == other.to_config()

    def empty(self):
        return not any(getattr(self, k) for k in self.KEYS)

    def client(self, timeout=DEFAULT_TIMEOUT):
        return LLMClient(self.base_url, self.api_key, self.model_name, timeout=timeout)


def load_llm_configs(katrain):
    """Read the ``llm`` section of the KaTrain config, as :class:`LLMConfig` objects."""
    raw = katrain.config("llm/configs") or []
    return [LLMConfig.from_config(c) for c in raw if isinstance(c, dict)]


def active_llm_config(katrain):
    """The currently selected :class:`LLMConfig`, or ``None`` if there is none."""
    configs = load_llm_configs(katrain)
    if not configs:
        return None
    ix = katrain.config("llm/active_config", 0) or 0
    return configs[max(0, min(ix, len(configs) - 1))]


def test_llm_connection(katrain, config, on_success=None, on_failure=None):
    """Send a one-message test request in a background thread."""

    def run():
        try:
            reply = config.client(timeout=30).chat(
                [{"role": "user", "content": "Reply with the single word: pong"}], max_tokens=8
            )
            katrain.log(f"LLM connection test succeeded: {reply!r}", OUTPUT_DEBUG)
            if on_success:
                on_success(reply)
        except LLMError as exc:
            katrain.log(f"LLM connection test failed: {exc}", OUTPUT_ERROR)
            if on_failure:
                on_failure(str(exc))

    threading.Thread(target=run, daemon=True).start()
