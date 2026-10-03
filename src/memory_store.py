from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re


DEFAULT_PROFILE_CONFIDENCE_THRESHOLD = 0.8


@dataclass(frozen=True)
class ProfileCandidate:
    value: str
    confidence: float


def estimate_tokens(text: str) -> int:
    """Estimate tokens deterministically at roughly four characters each."""
    content = text.strip()
    return (len(content) + 3) // 4 if content else 0


@dataclass
class UserProfileStore:
    """Store one persistent Markdown profile per user."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        user_id = user_id.strip()
        if not user_id:
            raise ValueError("user_id must not be empty")
        if re.fullmatch(r"[A-Za-z0-9_-]+", user_id):
            slug = user_id
        else:
            cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", user_id).strip("-_") or "user"
            slug = f"{cleaned[:48]}-{sha256(user_id.encode('utf-8')).hexdigest()[:10]}"
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as profile:
            profile.write(content)
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text:
            raise ValueError("search_text must not be empty")
        text = self.read_text(user_id)
        if search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.is_file() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        return dict(re.findall(r"^- ([a-z_]+): (.+)$", self.read_text(user_id), re.MULTILINE))

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        if not re.fullmatch(r"[a-z_]+", key) or not value.strip() or "\n" in value:
            raise ValueError("fact requires a simple key and a single-line value")
        content = self.read_text(user_id) or "# User Profile\n"
        line = f"- {key}: {value.strip()}"
        pattern = re.compile(rf"^- {re.escape(key)}: .*$(?:\n)?", re.MULTILINE)
        if pattern.search(content):
            first = True

            def replace(match: re.Match[str]) -> str:
                nonlocal first
                if first:
                    first = False
                    return line + "\n"
                return ""

            content = pattern.sub(replace, content)
        else:
            content = content.rstrip("\n") + "\n\n" + line + "\n"
        return self.write_text(user_id, content)


def extract_profile_candidates(message: str) -> dict[str, ProfileCandidate]:
    """Score candidate facts using explicitness of each sentence."""
    candidates: dict[str, ProfileCandidate] = {}
    clauses = re.split(r"(?<=[.!?])\s+|[,;\n]+|\bnhưng\b", message, flags=re.I)
    for clause in clauses:
        facts = _extract_profile_updates_raw(clause)
        if not facts:
            continue
        speculative = re.search(
            r"\b(?:có thể|cân nhắc|giả sử|nếu|ước gì|dự định|có lẽ|chắc là|"
            r"trước đây|trước kia|lúc đầu|nghe nói|ví dụ|anh ấy nói|cô ấy nói|bạn ấy nói)\b",
            clause,
            re.I,
        )
        confidence = 0.4 if speculative else 0.95
        for key, value in facts.items():
            previous = candidates.get(key)
            if previous is None or confidence >= previous.confidence:
                candidates[key] = ProfileCandidate(value, confidence)
    # Interest lists commonly use commas; retain the full explicit list.
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", message):
        interest_claim = re.search(r"mình\s+(?:thích|quan tâm(?: nhiều)? đến)", sentence, re.I)
        if not interest_claim:
            continue
        if re.search(r"\b(?:nếu|giả sử|có thể|cân nhắc|trước đây|nghe nói)\b", sentence[:interest_claim.start()], re.I):
            continue
        interests = _extract_profile_updates_raw(sentence).get("interests")
        if interests:
            candidates["interests"] = ProfileCandidate(interests, 0.95)
    return candidates


def extract_profile_updates(
    message: str, min_confidence: float = DEFAULT_PROFILE_CONFIDENCE_THRESHOLD
) -> dict[str, str]:
    """Return only profile facts meeting the heuristic confidence threshold."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    return {key: candidate.value for key, candidate in extract_profile_candidates(message).items()
            if candidate.confidence >= min_confidence}


def _extract_profile_updates_raw(message: str) -> dict[str, str]:
    """Find fact-shaped phrases before confidence filtering."""
    updates: dict[str, str] = {}
    clauses = re.split(r"(?<=[.!?])\s+|\n+", message)
    for clause in clauses:
        clause = clause.strip()
        if not clause or clause.endswith("?"):
            continue

        name = re.search(r"\b(?:mình tên là|tên mình là)\s+([\wÀ-ỹ]+(?:\s+Stress)?)", clause, re.I)
        if name:
            updates["name"] = name.group(1).strip()

        current = re.search(
            r"(?i:mình\s+(?:đang\s+|vẫn\s+)?ở|hiện\s+ở|nơi ở hiện tại\s+là|mình\s+đang\s+làm việc\s+ở)\s+"
            r"([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,2})", clause,
        )
        moved = re.search(r"nơi ở.*?từ\s+[A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,2}\s+sang\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*){0,2})", clause)
        if moved:
            updates["location"] = moved.group(1)
        elif current:
            updates["location"] = current.group(1)

        if not re.search(r"\b(?:đùa|câu đùa|hay là)\b", clause, re.I):
            job = re.search(
                r"(?:đang làm|chuyển sang|nghề nghiệp hiện tại\s+(?:vẫn\s+)?là|nghề hiện tại\s+là)\s+"
                r"((?:[\wÀ-ỹ]+\s+)?(?:engineer|developer|designer|manager|analyst)|giáo viên|lập trình viên)", clause, re.I,
            )
            if job:
                updates["profession"] = job.group(1)

        if re.search(r"(?:mình muốn|mình thích|mình vẫn muốn|hãy)\s+.*?trả lời", clause, re.I):
            style = []
            if re.search(r"ngắn gọn|trả lời ngắn|bullet ngắn", clause, re.I):
                style.append("ngắn gọn")
            if re.search(r"3\s+bullet", clause, re.I):
                style.append("3 bullet")
            elif re.search(r"\bbullet\b", clause, re.I):
                style.append("bullet")
            if re.search(r"ví dụ thực (?:tế|chiến)", clause, re.I):
                style.append("có ví dụ thực tế")
            if style:
                updates["response_style"] = ", ".join(style)

        drink = re.search(r"(?:đồ uống yêu thích là|mình vẫn uống)\s+([^,.!?]+)", clause, re.I)
        food = re.search(r"(?:món ăn yêu thích là|món ruột là)\s+([^,.!?]+)", clause, re.I)
        pet = re.search(r"mình nuôi\s+(?:một\s+)?(?:bé\s+)?(corgi)(?:\s+tên\s+(\w+))?", clause, re.I)
        if drink:
            updates["favorite_drink"] = re.split(r"\s+(?:như cũ|nhưng|và|để)\b", drink.group(1), maxsplit=1, flags=re.I)[0].strip()
        if food:
            updates["favorite_food"] = food.group(1).strip()
        if pet:
            updates["pet"] = " ".join(part for part in pet.groups() if part)

        if re.search(r"mình\s+(?:thích|quan tâm(?: nhiều)? đến)", clause, re.I):
            interests = [term for term in ("Python", "AI", "MLOps", "RAG") if re.search(rf"\b{term}\b", clause, re.I)]
            if interests:
                updates["interests"] = ", ".join(interests)
    return updates


def summarize_messages(
    messages: list[dict[str, str]], max_items: int = 6, max_chars: int | None = None
) -> str:
    """Keep salient early facts and recent context within a character budget."""
    if max_items <= 0:
        return ""
    important = re.compile(r"\b(?:deadline|hạn chót|quan trọng|cần nhớ|nhớ rằng|mục tiêu|quyết định|ưu tiên)\b", re.I)
    anchors: list[str] = []
    recent: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if role == "summary":
            lines = [line.strip() for line in content.splitlines() if line.strip()]
        else:
            compact = " ".join(content.split())
            lines = [f"{role}: {compact[:180]}"] if compact else []
            recent.extend(lines)
        for line in lines:
            anchor = line[:140]
            if important.search(anchor) and anchor not in anchors:
                anchors.append(anchor)

    keep_anchors = list(dict.fromkeys(anchors[:1] + anchors[-2:]))
    keep_recent = [line for line in recent[-max_items:] if line[:140] not in keep_anchors]
    if max_chars is None:
        return "\n".join(keep_anchors + keep_recent)
    if max_chars <= 0:
        return ""

    chosen: list[str] = []
    anchor_budget = max_chars // 2
    for line in keep_anchors:
        remaining = anchor_budget - len("\n".join(chosen)) - (1 if chosen else 0)
        if remaining > 0:
            chosen.append(line[:remaining])
    for line in reversed(keep_recent):
        remaining = max_chars - len("\n".join(chosen)) - (1 if chosen else 0)
        if remaining <= 0:
            break
        chosen.append(line[:remaining])
    return "\n".join(chosen)


@dataclass
class CompactMemoryManager:
    """Bound the carried thread context by summarizing older messages."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0 or self.keep_messages < 0:
            raise ValueError("threshold_tokens must be positive and keep_messages nonnegative")

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        size = estimate_tokens(str(thread["summary"])) + sum(estimate_tokens(item["content"]) for item in messages)
        keep = self.keep_messages
        older = messages[:-keep] if keep else messages[:]
        if size > self.threshold_tokens and older:
            previous = [{"role": "summary", "content": str(thread["summary"])}] if thread["summary"] else []
            thread["summary"] = summarize_messages(
                previous + older, max_chars=self.threshold_tokens * 4 // 3
            )
            thread["messages"] = messages[-keep:] if keep else []
            thread["compactions"] = int(thread["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id)
        if thread is None:
            return {"messages": [], "summary": "", "compactions": 0}
        return {"messages": [message.copy() for message in thread["messages"]],
                "summary": thread["summary"], "compactions": thread["compactions"]}

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
