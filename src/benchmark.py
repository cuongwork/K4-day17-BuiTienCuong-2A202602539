from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Load one of the shared Vietnamese benchmark datasets."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Score a recall answer as none, partial, or complete."""
    if not expected:
        return 0.0
    matches = sum(fact.casefold() in answer.casefold() for fact in expected)
    return 1.0 if matches == len(expected) else 0.5 if matches else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Score directness, concision, and informative structure separately from recall."""
    if not answer.strip():
        return 0.0
    direct = 0.0 if re.search(r"đã nhận thông tin|chưa có thông tin|không nhớ|không biết", answer, re.I) else 1.0
    concise = 1.0 if 0 < len(answer.strip()) <= 250 else 0.0
    structured = 1.0 if re.search(r"(?:^|;\s*)[a-z_]+:\s*\S", answer) else 0.0
    return round(0.5 * direct + 0.25 * concise + 0.25 * structured, 2)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Run the same turns and fresh-thread recall questions for one agent."""
    user_ids = {conversation["user_id"] for conversation in conversations}
    memory_size = getattr(agent, "memory_file_size", None)
    before_bytes = sum(memory_size(user_id) for user_id in user_ids) if memory_size else 0
    agent_tokens = 0
    prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    thread_ids: list[str] = []

    def record(result: dict[str, Any]) -> None:
        nonlocal agent_tokens, prompt_tokens
        agent_tokens += result["tokens"]
        prompt_tokens += result["prompt_tokens"]

    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = f"benchmark:{agent_name}:{conversation['id']}:turns"
        thread_ids.append(thread_id)
        for turn in conversation["turns"]:
            record(agent.reply(user_id, thread_id, turn))
        for index, item in enumerate(conversation.get("recall_questions", [])):
            recall_thread = f"benchmark:{agent_name}:{conversation['id']}:recall:{index}"
            thread_ids.append(recall_thread)
            result = agent.reply(user_id, recall_thread, item["question"])
            record(result)
            recall_scores.append(recall_points(result["response"], item["expected_contains"]))
            quality_scores.append(heuristic_quality(result["response"], item["expected_contains"]))

    after_bytes = sum(memory_size(user_id) for user_id in user_ids) if memory_size else 0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=after_bytes - before_bytes,
        compactions=sum(agent.compaction_count(thread_id) for thread_id in thread_ids),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render the required comparison columns as a Markdown table."""
    header = ("| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | "
              "Response quality | Memory growth (bytes) | Compactions |")
    divider = "|---|---:|---:|---:|---:|---:|---:|"
    entries = [
        f"| {row.agent_name} | {row.agent_tokens_only} | {row.prompt_tokens_processed} | "
        f"{row.recall_score:.0%} | {row.response_quality:.2f} | "
        f"{row.memory_growth_bytes} | {row.compactions} |"
        for row in rows
    ]
    return "\n".join([header, divider, *entries])


def main() -> None:
    """Print Standard and Long-Context Stress comparisons in offline mode."""

    config = load_config(Path(__file__).resolve().parent.parent)
    suites = (
        ("Standard Benchmark", config.data_dir / "conversations.json"),
        ("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json"),
    )
    with TemporaryDirectory(prefix="memory-lab-benchmark-") as temporary:
        for suite_name, dataset_path in suites:
            conversations = load_conversations(dataset_path)
            suite_dir = Path(temporary) / dataset_path.stem
            baseline_config = replace(config, state_dir=suite_dir / "baseline")
            advanced_config = replace(config, state_dir=suite_dir / "advanced")
            rows = [
                run_agent_benchmark("Baseline", BaselineAgent(baseline_config, force_offline=True), conversations, baseline_config),
                run_agent_benchmark("Advanced", AdvancedAgent(advanced_config, force_offline=True), conversations, advanced_config),
            ]
            print(f"\n## {suite_name}\n")
            print(format_rows(rows))


if __name__ == "__main__":
    main()
