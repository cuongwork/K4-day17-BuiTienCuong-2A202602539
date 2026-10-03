from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    DEFAULT_PROFILE_CONFIDENCE_THRESHOLD,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent with per-thread context, persistent user facts, and compaction."""

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
        profile_confidence_threshold: float = DEFAULT_PROFILE_CONFIDENCE_THRESHOLD,
    ) -> None:
        if not 0 <= profile_confidence_threshold <= 1:
            raise ValueError("profile_confidence_threshold must be between 0 and 1")
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_confidence_threshold = profile_confidence_threshold
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self._thread_keys: dict[tuple[str, str], str] = {}

        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Use the deterministic path unless a live model is configured."""
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        self._remember_and_append(user_id, thread_id, message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        context = self.thread_context(user_id, thread_id)
        system = self.profile_store.read_text(user_id)
        if context["summary"]:
            system += "\n\nEarlier conversation summary:\n" + str(context["summary"])
        prompt = [{"role": "system", "content": system or "Be helpful and concise."}, *context["messages"]]
        result = self.langchain_agent.invoke(prompt)
        response = result.content if isinstance(result.content, str) else str(result.content)
        return self._record_response(user_id, thread_id, response, prompt_tokens)

    def _thread_key(self, user_id: str, thread_id: str) -> str:
        key = (user_id, thread_id)
        return self._thread_keys.setdefault(key, json.dumps(key, ensure_ascii=False))

    def _matching_keys(self, thread_id: str, user_id: str | None) -> list[str]:
        return [key for (owner, current), key in self._thread_keys.items()
                if current == thread_id and (user_id is None or owner == user_id)]

    def thread_context(self, user_id: str, thread_id: str) -> dict[str, object]:
        return self.compact_memory.context(self._thread_key(user_id, thread_id))

    def token_usage(self, thread_id: str, user_id: str | None = None) -> int:
        return sum(self.thread_tokens.get(key, 0) for key in self._matching_keys(thread_id, user_id))

    def prompt_token_usage(self, thread_id: str, user_id: str | None = None) -> int:
        return sum(self.thread_prompt_tokens.get(key, 0) for key in self._matching_keys(thread_id, user_id))

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str, user_id: str | None = None) -> int:
        return sum(self.compact_memory.compaction_count(key) for key in self._matching_keys(thread_id, user_id))

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Run the same memory flow with a reproducible local response."""
        self._remember_and_append(user_id, thread_id, message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        response = self._offline_response(user_id, thread_id, message)
        return self._record_response(user_id, thread_id, response, prompt_tokens)

    def _remember_and_append(self, user_id: str, thread_id: str, message: str) -> None:
        for key, value in extract_profile_updates(message, self.profile_confidence_threshold).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(self._thread_key(user_id, thread_id), "user", message)

    def _record_response(self, user_id: str, thread_id: str, response: str, prompt_tokens: int) -> dict[str, Any]:
        key = self._thread_key(user_id, thread_id)
        tokens = estimate_tokens(response)
        self.compact_memory.append(key, "assistant", response)
        self.thread_tokens[key] = self.thread_tokens.get(key, 0) + tokens
        self.thread_prompt_tokens[key] = self.thread_prompt_tokens.get(key, 0) + prompt_tokens
        return {"response": response, "tokens": tokens, "prompt_tokens": prompt_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Count the profile, compact summary, and retained messages."""
        context = self.thread_context(user_id, thread_id)
        return (estimate_tokens(self.profile_store.read_text(user_id))
                + estimate_tokens(str(context["summary"]))
                + sum(estimate_tokens(item["content"]) for item in context["messages"]))

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer explicit recall questions from the user's stored facts."""
        facts = self.profile_store.facts(user_id)
        question = message.lower()
        cues = {
            "name": ("tên", "là ai"),
            "profession": ("nghề", "công việc", "làm gì"),
            "location": ("ở đâu", "nơi ở", "đang ở", "còn ở"),
            "response_style": ("style", "trả lời", "phong cách", "trình bày"),
            "favorite_drink": ("đồ uống", "uống gì"),
            "favorite_food": ("món ăn", "ăn gì"),
            "pet": ("nuôi con gì", "thú cưng"),
            "interests": ("quan tâm", "sở thích", "thích gì"),
        }
        recalled = [f"{key}: {facts[key]}" for key, phrases in cues.items()
                    if key in facts and any(phrase in question for phrase in phrases)]
        if recalled:
            return "; ".join(recalled)

        if "vừa nhắc" in question or "lúc nãy" in question:
            context = self.thread_context(user_id, thread_id)
            prior = [item["content"] for item in context["messages"]
                     if item["role"] == "user" and item["content"] != message]
            if prior:
                return f"Trong thread này, bạn vừa nói: {prior[-1][:160]}"
        return "Mình đã ghi nhận; hiện chưa có thông tin phù hợp để nhắc lại."

    def _maybe_build_langchain_agent(self):
        """Build an optional live model; memory stays in this agent's stores."""
        if self.force_offline:
            return None
        model = self.config.model
        if model.provider == "ollama":
            return build_chat_model(model)
        if model.provider == "custom" and model.base_url and model.api_key:
            return build_chat_model(model)
        if model.api_key:
            return build_chat_model(model)
        return None
