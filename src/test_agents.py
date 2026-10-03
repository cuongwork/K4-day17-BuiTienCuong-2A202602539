from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Give each test isolated state and a threshold that triggers quickly."""
    return replace(load_config(tmp_path), compact_threshold_tokens=160, compact_keep_messages=2)


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(make_config(tmp_path).state_dir / "profiles")
    assert store.read_text("lan") == ""
    assert store.file_size("lan") == 0

    path = store.write_text("lan", "# User Profile\n\n- location: Huế\n")
    assert path == tmp_path / "state" / "profiles" / "lan" / "User.md"
    assert path.is_file()
    assert store.edit_text("lan", "Huế", "Đà Nẵng") is True
    assert store.read_text("lan") == "# User Profile\n\n- location: Đà Nẵng\n"
    assert store.file_size("lan") == len(store.read_text("lan").encode("utf-8"))
    assert store.edit_text("lan", "Huế", "Hà Nội") is False


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    turns = [f"Lượt {index}: " + "nội dung tạm " * 14 for index in range(10)]
    for turn in turns:
        agent.reply("lan", "long", turn)

    context = agent.thread_context("lan", "long")
    assert agent.compaction_count("long") >= 2
    assert context["summary"]
    assert len(context["messages"]) <= 4
    assert all(turns[0] not in item["content"] for item in context["messages"])
    assert any(turns[-1] in item["content"] for item in context["messages"])
    assert agent.compaction_count("unused") == 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for agent in (baseline, advanced):
        agent.reply("lan", "first", "Mình tên là Lan. Mình ở Huế.")

    question = "Mình tên gì và ở đâu?"
    assert "Lan" in baseline.reply("lan", "first", question)["response"]
    assert "Lan" not in baseline.reply("lan", "second", question)["response"]

    reopened = AdvancedAgent(config, force_offline=True)
    answer = reopened.reply("lan", "second", question)["response"]
    assert "Lan" in answer and "Huế" in answer
    assert "Lan" not in reopened.reply("other", "other-thread", question)["response"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for index in range(20):
        turn = f"Lượt {index}: " + "đây là đoạn hội thoại dài " * 10
        baseline.reply("lan", "long", turn)
        advanced.reply("lan", "long", turn)

    assert advanced.compaction_count("long") > 0
    assert baseline.compaction_count("long") == 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    assert advanced.prompt_token_usage("long") > 0


def test_correction_replaces_old_fact_without_saving_jokes(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("lan", "first", "Mình ở Huế và đang làm backend engineer.")
    agent.reply("lan", "first", "Mình đang ở Đà Nẵng. Giờ chuyển sang MLOps engineer.")
    agent.reply("lan", "first", "Mình đùa là chuyển sang product manager. Hà Nội chỉ là nơi đi họp.")
    agent.reply("lan", "first", "Mình ở đâu?")

    profile = agent.profile_store.read_text("lan")
    assert "Đà Nẵng" in profile and "MLOps engineer" in profile
    assert "Huế" not in profile and "backend engineer" not in profile
    assert "product manager" not in profile and "Hà Nội" not in profile


def test_low_confidence_claim_does_not_replace_confirmed_profile(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("lan", "first", "Mình ở Huế và đang làm MLOps engineer.")
    stable_size = agent.memory_file_size("lan")
    agent.reply("lan", "first", "Nếu mình ở Hà Nội thì đi lại sẽ tiện hơn.")
    agent.reply("lan", "first", "Mình có thể chuyển sang product manager.")

    profile = agent.profile_store.facts("lan")
    assert profile["location"] == "Huế"
    assert profile["profession"] == "MLOps engineer"
    assert "Hà Nội" not in agent.profile_store.read_text("lan")
    assert "product manager" not in agent.profile_store.read_text("lan")
    assert agent.memory_file_size("lan") == stable_size
