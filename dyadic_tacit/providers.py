"""Frozen black-box LLM provider boundary used by the method.

The learner never receives provider internals or gradients.  It only receives
the host's observable response records.  GPT, Claude, and Gemini adapters all
implement the same single-request interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping, Protocol, Sequence

import httpx


Message = Mapping[str, str]


@dataclass(frozen=True)
class ProviderReply:
    text: str
    provider: str
    model: str
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class FrozenLLM(Protocol):
    provider: str
    model: str

    def complete(self, messages: Sequence[Message], *, role: str = "") -> ProviderReply:
        """Generate one frozen-agent response for the host prompt."""


def _validate_messages(messages: Sequence[Message]) -> list[dict[str, str]]:
    if not messages:
        raise ValueError("EMPTY_LLM_PROMPT")
    normalized = []
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("INVALID_LLM_MESSAGE_ROLE")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("INVALID_LLM_MESSAGE_CONTENT")
        normalized.append({"role": str(message["role"]), "content": content})
    return normalized


class _HttpProvider:
    provider = ""

    def __init__(self, model: str, api_key: str, *, endpoint: str,
                 timeout: float = 60.0, client: httpx.Client | None = None):
        if not model or not api_key:
            raise ValueError("MISSING_PROVIDER_MODEL_OR_KEY")
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.client = client or httpx.Client(timeout=timeout, trust_env=False)

    def _post(self, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any]) -> dict:
        response = self.client.post(url, headers=dict(headers), json=dict(payload))
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("PROVIDER_RESPONSE_NOT_OBJECT")
        return data


class GPTProvider(_HttpProvider):
    """OpenAI Chat Completions adapter for GPT models."""

    provider = "gpt"

    def __init__(self, model: str = "gpt-5.5", api_key: str | None = None, *,
                 endpoint: str = "https://api.openai.com/v1/chat/completions",
                 temperature: float = 0.0, max_tokens: int = 512,
                 timeout: float = 60.0, client: httpx.Client | None = None):
        super().__init__(model, api_key or os.environ.get("OPENAI_API_KEY", ""),
                         endpoint=endpoint, timeout=timeout, client=client)
        self.temperature, self.max_tokens = temperature, max_tokens

    def complete(self, messages: Sequence[Message], *, role: str = "") -> ProviderReply:
        data = self._post(self.endpoint,
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type": "application/json"},
                          payload={"model": self.model, "messages": _validate_messages(messages),
                                   "temperature": self.temperature,
                                   "max_completion_tokens": self.max_tokens})
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("INVALID_GPT_RESPONSE") from exc
        usage = data.get("usage") or {}
        return ProviderReply(str(text), self.provider, self.model, data.get("id"),
                             usage.get("prompt_tokens"), usage.get("completion_tokens"))


class ClaudeProvider(_HttpProvider):
    """Anthropic Messages API adapter for Claude models."""

    provider = "claude"

    def __init__(self, model: str = "claude-sonnet-4-8", api_key: str | None = None, *,
                 endpoint: str = "https://api.anthropic.com/v1/messages",
                 temperature: float = 0.0, max_tokens: int = 512,
                 timeout: float = 60.0, client: httpx.Client | None = None):
        super().__init__(model, api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
                         endpoint=endpoint, timeout=timeout, client=client)
        self.temperature, self.max_tokens = temperature, max_tokens

    def complete(self, messages: Sequence[Message], *, role: str = "") -> ProviderReply:
        normalized = _validate_messages(messages)
        system = "\n\n".join(m["content"] for m in normalized if m["role"] == "system")
        turns = [{"role": "user" if m["role"] == "system" else m["role"],
                  "content": m["content"]} for m in normalized if m["role"] != "system"]
        if not turns or turns[0]["role"] != "user":
            raise ValueError("CLAUDE_PROMPT_MUST_START_WITH_USER")
        payload: dict[str, Any] = {"model": self.model, "messages": turns,
                                   "max_tokens": self.max_tokens, "temperature": self.temperature}
        if system:
            payload["system"] = system
        data = self._post(self.endpoint,
                          headers={"x-api-key": self.api_key,
                                   "anthropic-version": "2023-06-01",
                                   "Content-Type": "application/json"}, payload=payload)
        try:
            text = "".join(part["text"] for part in data["content"] if part.get("type") == "text")
        except (KeyError, TypeError) as exc:
            raise ValueError("INVALID_CLAUDE_RESPONSE") from exc
        usage = data.get("usage") or {}
        return ProviderReply(text, self.provider, self.model, data.get("id"),
                             usage.get("input_tokens"), usage.get("output_tokens"))


class GeminiProvider(_HttpProvider):
    """Google Generative Language API adapter for Gemini models."""

    provider = "gemini"

    def __init__(self, model: str = "gemini-3.6-flash", api_key: str | None = None, *,
                 endpoint: str = "https://generativelanguage.googleapis.com/v1beta",
                 temperature: float = 0.0, max_tokens: int = 512,
                 timeout: float = 60.0, client: httpx.Client | None = None):
        super().__init__(model, api_key or os.environ.get("GEMINI_API_KEY", ""),
                         endpoint=endpoint, timeout=timeout, client=client)
        self.temperature, self.max_tokens = temperature, max_tokens

    def complete(self, messages: Sequence[Message], *, role: str = "") -> ProviderReply:
        normalized = _validate_messages(messages)
        system_parts = [m["content"] for m in normalized if m["role"] == "system"]
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in normalized if m["role"] != "system"]
        if not contents:
            raise ValueError("EMPTY_GEMINI_CONTENT")
        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": self.temperature,
                                  "maxOutputTokens": self.max_tokens},
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        url = f"{self.endpoint}/models/{self.model}:generateContent?key={self.api_key}"
        data = self._post(url, headers={"Content-Type": "application/json"}, payload=payload)
        try:
            text = "".join(part["text"] for part in data["candidates"][0]["content"]["parts"]
                           if "text" in part)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("INVALID_GEMINI_RESPONSE") from exc
        usage = data.get("usageMetadata") or {}
        return ProviderReply(text, self.provider, self.model, None,
                             usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))


def provider_from_environment(name: str | None = None) -> FrozenLLM:
    """Construct a provider from ``DTU_LLM_PROVIDER`` or an explicit name."""
    provider = (name or os.environ.get("DTU_LLM_PROVIDER", "gpt")).lower()
    if provider in {"gpt", "openai"}:
        return GPTProvider(model=os.environ.get("DTU_LLM_MODEL", "gpt-5.5"))
    if provider in {"claude", "anthropic"}:
        return ClaudeProvider(model=os.environ.get("DTU_LLM_MODEL", "claude-sonnet-4-8"))
    if provider == "gemini":
        return GeminiProvider(model=os.environ.get("DTU_LLM_MODEL", "gemini-3.6-flash"))
    raise ValueError(f"UNKNOWN_LLM_PROVIDER:{provider}")
