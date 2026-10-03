from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import dotenv_values

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load repository paths, compact settings, and live model settings."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    env_file = dotenv_values(root / ".env")

    def setting(name: str, default: str = "") -> str:
        return os.environ.get(name, env_file.get(name) or default)

    def positive_int(name: str, default: int) -> int:
        raw = setting(name, str(default))
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
        provider = normalize_provider(setting(f"{prefix}_PROVIDER", fallback.provider if fallback else "openai"))
        model_name = setting(f"{prefix}_MODEL", fallback.model_name if fallback else "gpt-4o-mini")
        key_names = {
            "openai": "OPENAI_API_KEY",
            "custom": "CUSTOM_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        base_names = {"custom": "CUSTOM_BASE_URL", "ollama": "OLLAMA_BASE_URL"}
        api_key = setting(f"{prefix}_API_KEY") or setting(key_names[provider]) if provider in key_names else None
        base_url = setting(f"{prefix}_BASE_URL") or setting(base_names[provider]) if provider in base_names else None
        return ProviderConfig(
            provider=provider,
            model_name=model_name,
            temperature=float(setting(f"{prefix}_TEMPERATURE", str(fallback.temperature if fallback else 0))),
            api_key=api_key or None,
            base_url=base_url or None,
        )

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    model = provider_config("LLM")
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=positive_int("COMPACT_THRESHOLD_TOKENS", 1200),
        compact_keep_messages=positive_int("COMPACT_KEEP_MESSAGES", 4),
        model=model,
        judge_model=provider_config("JUDGE", model),
    )
