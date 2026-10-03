from dataclasses import replace
import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


def test_advanced_persists_corrected_facts_across_instances(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    agent.reply("lan", "one", "Mình tên là Lan. Mình đang ở Huế và đang làm backend engineer.")
    agent.reply("lan", "one", "Mình đang ở Đà Nẵng chứ không còn ở Huế. Giờ chuyển sang MLOps engineer.")
    agent.reply("lan", "one", "Mình đùa rằng chuyển sang product manager. Hà Nội chỉ là nơi đi họp.")

    reopened = AdvancedAgent(config, force_offline=True)
    answer = reopened.reply("lan", "two", "Mình tên gì, hiện ở đâu và làm nghề gì?")["response"]
    assert "Lan" in answer
    assert "Đà Nẵng" in answer
    assert "MLOps engineer" in answer
    assert "backend engineer" not in answer
    assert "product manager" not in answer
    assert "Hà Nội" not in answer
    assert reopened.memory_file_size("lan") > 0
    assert reopened.memory_file_size("other") == 0


def test_advanced_keeps_temporary_context_in_thread_only(tmp_path: Path) -> None:
    agent = AdvancedAgent(load_config(tmp_path), force_offline=True)
    agent.reply("lan", "one", "Hôm nay mình vừa họp với An về dự án.")
    same = agent.reply("lan", "one", "Mình vừa nhắc chuyện gì?")["response"]
    other = agent.reply("lan", "two", "Mình vừa nhắc chuyện gì?")["response"]
    assert "An" in same
    assert "An" not in other
    assert agent.profile_store.facts("lan") == {}


def test_advanced_compacts_and_reduces_prompt_load(tmp_path: Path) -> None:
    config = replace(load_config(tmp_path), compact_threshold_tokens=140, compact_keep_messages=2)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for index in range(20):
        message = f"Lượt {index}: " + "nội dung tạm thời " * 12
        advanced.reply("lan", "long", message)
        baseline.reply("lan", "long", message)
    assert advanced.compaction_count("long") > 0
    assert len(advanced.thread_context("lan", "long")["messages"]) <= 4
    assert all("Lượt 0:" not in item["content"] for item in advanced.thread_context("lan", "long")["messages"])
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
    assert advanced.token_usage("long") > 0


def test_advanced_recalls_stress_dataset_in_new_thread(tmp_path: Path) -> None:
    data = json.loads((Path(__file__).resolve().parent.parent / "data" / "advanced_long_context.json").read_text(encoding="utf-8"))[0]
    agent = AdvancedAgent(load_config(tmp_path), force_offline=True)
    for turn in data["turns"]:
        agent.reply(data["user_id"], "stress", turn)
    for index, recall in enumerate(data["recall_questions"]):
        answer = agent.reply(data["user_id"], f"fresh-{index}", recall["question"])["response"]
        for expected in recall["expected_contains"]:
            assert expected in answer
    assert agent.compaction_count("stress") > 0


def test_advanced_recalls_all_standard_dataset_questions(tmp_path: Path) -> None:
    data = json.loads((Path(__file__).resolve().parent.parent / "data" / "conversations.json").read_text(encoding="utf-8"))
    agent = AdvancedAgent(load_config(tmp_path), force_offline=True)
    for conversation in data:
        for turn in conversation["turns"]:
            agent.reply(conversation["user_id"], conversation["id"], turn)
        for index, recall in enumerate(conversation["recall_questions"]):
            answer = agent.reply(conversation["user_id"], f"{conversation['id']}-recall-{index}", recall["question"])["response"]
            for expected in recall["expected_contains"]:
                assert expected.lower() in answer.lower(), (conversation["id"], expected, answer)
