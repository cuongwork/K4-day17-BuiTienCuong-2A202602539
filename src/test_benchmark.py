from dataclasses import replace
import json
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from benchmark import BenchmarkRow, format_rows, heuristic_quality, load_conversations, main, recall_points, run_agent_benchmark
from config import load_config


def test_load_and_score_recall(tmp_path: Path) -> None:
    dataset = [{"id": "one", "user_id": "u", "turns": ["Mình tên là Lan."],
                "recall_questions": [{"question": "Tên mình?", "expected_contains": ["Lan", "Huế"]}]}]
    path = tmp_path / "small.json"
    path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    assert load_conversations(path) == dataset
    assert recall_points("Không nhớ", ["Lan", "Huế"]) == 0
    assert recall_points("LAN", ["Lan", "Huế"]) == 0.5
    assert recall_points("Lan ở Huế", ["Lan", "Huế"]) == 1
    assert heuristic_quality("Lan ở Huế", ["Lan", "Huế"]) > heuristic_quality("Không nhớ", ["Lan", "Huế"])


def test_response_quality_does_not_reuse_recall_ground_truth() -> None:
    assert heuristic_quality("name: Lan", ["Lan"]) == heuristic_quality("name: Mai", ["Lan"])
    assert heuristic_quality("name: Lan", ["Lan"]) > heuristic_quality("Mình đã nhận thông tin trong cuộc trò chuyện này.", ["Lan"])


def test_benchmark_uses_fresh_threads_and_measures_memory(tmp_path: Path) -> None:
    config = replace(load_config(tmp_path), compact_threshold_tokens=80, compact_keep_messages=2)
    conversations = [{"id": "one", "user_id": "lan", "turns": ["Mình tên là Lan.", "Mình ở Huế."],
                      "recall_questions": [{"question": "Mình tên gì và ở đâu?", "expected_contains": ["Lan", "Huế"]}]}]
    baseline = run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), conversations, config)
    advanced = run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), conversations, config)
    assert baseline.recall_score == 0
    assert advanced.recall_score == 1
    assert baseline.memory_growth_bytes == 0
    assert advanced.memory_growth_bytes > 0
    assert baseline.agent_tokens_only > 0
    assert advanced.prompt_tokens_processed > 0


def test_format_has_six_required_metrics() -> None:
    table = format_rows([BenchmarkRow("Baseline", 1, 2, 0, 0.25, 0, 0),
                         BenchmarkRow("Advanced", 3, 4, 1, 1, 128, 2)])
    for heading in ("Agent tokens only", "Prompt tokens processed", "Cross-session recall",
                    "Response quality", "Memory growth (bytes)", "Compactions"):
        assert heading in table
    assert "Baseline" in table and "Advanced" in table


def test_main_prints_both_benchmark_suites(capsys) -> None:
    main()
    output = capsys.readouterr().out
    assert "Standard Benchmark" in output
    assert "Long-Context Stress Benchmark" in output
    assert output.count("Prompt tokens processed") == 2
    assert output.count("| Baseline |") == 2
    assert output.count("| Advanced |") == 2
    main()
    assert capsys.readouterr().out == output
