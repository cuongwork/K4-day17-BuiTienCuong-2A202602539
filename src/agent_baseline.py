from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Student TODO: implement Agent A.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[tuple[str, str], SessionState] = {}

        self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply using only the history attached to this thread."""
        if self.langchain_agent is None:
            return self._reply_offline(user_id, thread_id, message)
        session = self.sessions.setdefault((user_id, thread_id), SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        result = self.langchain_agent.invoke(session.messages)
        response = result.content if isinstance(result.content, str) else str(result.content)
        return self._record_response(session, response, prompt_tokens)

    def token_usage(self, thread_id: str, user_id: str | None = None) -> int:
        return sum(session.token_usage for (owner, current), session in self.sessions.items()
                   if current == thread_id and (user_id is None or owner == user_id))

    def prompt_token_usage(self, thread_id: str, user_id: str | None = None) -> int:
        return sum(session.prompt_tokens_processed for (owner, current), session in self.sessions.items()
                   if current == thread_id and (user_id is None or owner == user_id))

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Generate a reproducible answer from this thread's messages only."""
        session = self.sessions.setdefault((user_id, thread_id), SessionState())
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))

        question = message.lower()
        requested = []
        for key, cues in {
            "name": ("tên", "ai"),
            "location": ("ở đâu", "nơi ở", "địa điểm"),
            "profession": ("nghề", "làm gì", "công việc"),
            "response_style": ("style", "trả lời", "phong cách"),
            "favorite_drink": ("đồ uống", "uống gì"),
            "favorite_food": ("món ăn", "ăn gì"),
            "pet": ("con gì", "thú cưng"),
            "interests": ("quan tâm", "sở thích"),
        }.items():
            if any(cue in question for cue in cues) and key in facts:
                requested.append(f"{key}: {facts[key]}")
        response = "; ".join(requested) if requested else "Mình đã nhận thông tin trong cuộc trò chuyện này."
        return self._record_response(session, response, prompt_tokens)

    @staticmethod
    def _record_response(session: SessionState, response: str, prompt_tokens: int) -> dict[str, Any]:
        tokens = estimate_tokens(response)
        session.messages.append({"role": "assistant", "content": response})
        session.token_usage += tokens
        session.prompt_tokens_processed += prompt_tokens
        return {"response": response, "tokens": tokens, "prompt_tokens": prompt_tokens}

    def _maybe_build_langchain_agent(self):
        """Build a live chat model only when a connection is configured."""
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
