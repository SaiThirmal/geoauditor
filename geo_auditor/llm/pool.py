"""A failover pool over every free LLM provider in providers.py.

The point is to use *all* the free quota at once instead of being capped by
any single provider's free tier:

- Requests are spread round-robin across every configured provider, so each
  provider's daily/hourly allowance is drawn down in parallel.
- When a provider rate-limits (HTTP 429), runs out of credits (402), rejects
  the key (401/403) or errors (5xx/timeout), it's put on a cooldown (honoring
  Retry-After when sent) and the request moves on to the next provider.
- `ask_many` sends one prompt to several *different* providers - that's what
  the live spot check uses to compare answers across engines.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from .providers import Provider, enabled_providers

TIMEOUT = 45
DEFAULT_MAX_TOKENS = 700
DEFAULT_COOLDOWN = 60.0  # seconds, when a 429 doesn't say how long
LONG_COOLDOWN = 3600.0  # bad key / out of credits: don't retry this run


@dataclass
class LLMResponse:
    provider_id: str
    provider_name: str
    model: str
    text: Optional[str]
    error: Optional[str] = None
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


def _retry_after(resp: requests.Response) -> float:
    try:
        return max(1.0, float(resp.headers.get("Retry-After", "")))
    except ValueError:
        return DEFAULT_COOLDOWN


def call_provider(provider: Provider, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                  temperature: float = 0.2) -> tuple[LLMResponse, float]:
    """One OpenAI-compatible chat completion. Returns (response, cooldown_s);
    cooldown_s > 0 means the provider should be skipped for that long."""
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    payload = {"model": provider.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}

    def fail(msg: str, cooldown: float) -> tuple[LLMResponse, float]:
        return LLMResponse(provider.id, provider.name, provider.model, None, error=msg,
                           latency_s=round(time.monotonic() - start, 2)), cooldown

    start = time.monotonic()
    try:
        resp = requests.post(provider.endpoint, json=payload, headers=headers, timeout=TIMEOUT)
    except requests.exceptions.RequestException as e:
        return fail(f"request failed: {e}", DEFAULT_COOLDOWN)

    if resp.status_code == 429:
        return fail("rate limited (429)", _retry_after(resp))
    if resp.status_code in (401, 402, 403):
        return fail(f"HTTP {resp.status_code}: {resp.text[:200]}", LONG_COOLDOWN)
    if resp.status_code >= 400:
        return fail(f"HTTP {resp.status_code}: {resp.text[:200]}", DEFAULT_COOLDOWN if resp.status_code >= 500 else 0)

    try:
        data = resp.json()
        text = (data["choices"][0]["message"].get("content") or "").strip()
    except (ValueError, KeyError, IndexError, TypeError):
        return fail(f"unexpected response: {resp.text[:200]}", 0)
    if not text:
        return fail("empty completion", 0)

    return LLMResponse(provider.id, provider.name, provider.model, text,
                       latency_s=round(time.monotonic() - start, 2), usage=data.get("usage") or {}), 0.0


class FreeLLMPool:
    def __init__(self, providers: Optional[list[Provider]] = None):
        self.providers = providers if providers is not None else enabled_providers()
        self._cooldown_until: dict[str, float] = {}
        self._next = 0
        self._lock = threading.Lock()

    def available(self) -> list[Provider]:
        now = time.monotonic()
        return [p for p in self.providers if self._cooldown_until.get(p.id, 0) <= now]

    def _rotation(self) -> list[Provider]:
        """Available providers, starting from the round-robin cursor."""
        with self._lock:
            avail = self.available()
            if not avail:
                return []
            start = self._next % len(avail)
            self._next += 1
            return avail[start:] + avail[:start]

    def _call(self, provider: Provider, messages: list[dict], **kw) -> LLMResponse:
        resp, cooldown = call_provider(provider, messages, **kw)
        if cooldown:
            with self._lock:
                self._cooldown_until[provider.id] = time.monotonic() + cooldown
        return resp

    def complete(self, prompt: str, system: Optional[str] = None, **kw) -> LLMResponse:
        """Answer from whichever provider is next in rotation, failing over
        through every other provider before giving up."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        errors = []
        for provider in self._rotation():
            resp = self._call(provider, messages, **kw)
            if resp.ok:
                return resp
            errors.append(f"{provider.id}: {resp.error}")
        return LLMResponse("none", "none", "", None,
                           error="all free providers failed or are cooling down" + (f" ({'; '.join(errors)})" if errors else ""))

    def ask_many(self, prompt: str, n: int = 3, system: Optional[str] = None, **kw) -> list[LLMResponse]:
        """Get up to `n` successful answers, each from a different provider."""
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        answers: list[LLMResponse] = []
        for provider in self._rotation():
            if len(answers) >= n:
                break
            resp = self._call(provider, messages, **kw)
            if resp.ok:
                answers.append(resp)
        return answers
