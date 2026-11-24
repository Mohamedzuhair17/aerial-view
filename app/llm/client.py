"""Thin provider-agnostic LLM wrapper. Providers are imported lazily."""
import logging

from app.config import settings

log = logging.getLogger("llm")

DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini", "groq": "llama-3.3-70b-versatile"}
KEYS = {"gemini": "gemini_api_key", "openai": "openai_api_key", "groq": "groq_api_key"}


class LLM:
    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.model = settings.llm_model or DEFAULT_MODELS.get(self.provider, "")
        self.key = getattr(settings, KEYS.get(self.provider, ""), "") if self.provider in KEYS else ""
        self.available = self.provider in KEYS and bool(self.key)
        self._client = None
        if self.provider != "none" and not self.available:
            log.warning("LLM_PROVIDER=%s but no API key set -> rule-based fallback", self.provider)

    def _get(self):
        if self._client is None:
            if self.provider == "gemini":
                from google import genai
                from google.genai import types
                self._client = genai.Client(
                    api_key=self.key,
                    http_options=types.HttpOptions(timeout=20_000),
                )
            else:
                from openai import OpenAI
                base = "https://api.groq.com/openai/v1" if self.provider == "groq" else None
                self._client = OpenAI(api_key=self.key, base_url=base, timeout=20.0, max_retries=1)
        return self._client

    def complete(self, system: str, user: str) -> str:
        if not self.available:
            raise RuntimeError("LLM not configured")
        client = self._get()
        if self.provider == "gemini":
            from google.genai import types
            r = client.models.generate_content(
                model=self.model, contents=user,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0),
            )
            return r.text or ""
        r = client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return r.choices[0].message.content or ""
