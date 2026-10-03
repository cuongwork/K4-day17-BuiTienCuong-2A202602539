from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def test_same_thread_id_does_not_share_baseline_history_between_users(tmp_path: Path) -> None:
    agent = BaselineAgent(load_config(tmp_path), force_offline=True)
    agent.reply("alice", "shared", "Mình tên là Lan.")
    assert "Lan" in agent.reply("alice", "shared", "Mình tên gì?")["response"]
    assert "Lan" not in agent.reply("bob", "shared", "Mình tên gì?")["response"]


def test_same_thread_id_does_not_share_advanced_temporary_context_between_users(tmp_path: Path) -> None:
    agent = AdvancedAgent(load_config(tmp_path), force_offline=True)
    agent.reply("alice", "shared", "Hôm nay mình vừa họp với An về dự án.")
    assert "An" in agent.reply("alice", "shared", "Mình vừa nhắc chuyện gì?")["response"]
    assert "An" not in agent.reply("bob", "shared", "Mình vừa nhắc chuyện gì?")["response"]
