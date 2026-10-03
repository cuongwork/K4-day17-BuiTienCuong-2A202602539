from pathlib import Path

from agent_baseline import BaselineAgent
from config import load_config


def test_baseline_remembers_only_within_thread(tmp_path: Path) -> None:
    agent = BaselineAgent(load_config(tmp_path), force_offline=True)
    agent.reply("user", "first", "Mình tên là Lan.")
    remembered = agent.reply("user", "first", "Mình tên gì?")
    forgotten = agent.reply("user", "second", "Mình tên gì?")

    assert "Lan" in remembered["response"]
    assert "Lan" not in forgotten["response"]
    assert not (tmp_path / "state" / "profiles").exists()


def test_baseline_counts_generated_and_processed_tokens_per_thread(tmp_path: Path) -> None:
    agent = BaselineAgent(load_config(tmp_path), force_offline=True)
    first = agent.reply("user", "a", "Xin chào")
    second = agent.reply("user", "a", "Mình thích Python")

    assert first["response"]
    assert agent.token_usage("a") == first["tokens"] + second["tokens"]
    assert second["prompt_tokens"] > first["prompt_tokens"]
    assert agent.prompt_token_usage("a") == first["prompt_tokens"] + second["prompt_tokens"]
    assert agent.token_usage("other") == 0
    assert agent.compaction_count("a") == 0
