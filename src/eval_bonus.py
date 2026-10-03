"""Compare profile confidence filtering on an adversarial offline dataset."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from agent_advanced import AdvancedAgent
from config import load_config


def run_ablation() -> dict[str, dict[str, float | int]]:
    root = Path(__file__).resolve().parent.parent
    cases = json.loads((root / "data" / "bonus_ablation.json").read_text(encoding="utf-8"))
    base_config = load_config(root)
    rows: dict[str, dict[str, float | int]] = {}
    with TemporaryDirectory(prefix="memory-bonus-ablation-") as temporary:
        for label, threshold in (("protected", 0.8), ("unfiltered", 0.0)):
            config = replace(base_config, state_dir=Path(temporary) / label)
            agent = AdvancedAgent(config, force_offline=True, profile_confidence_threshold=threshold)
            correct = false_facts = expected_total = prompt_tokens = agent_tokens = 0
            for case in cases:
                user_id = case["user_id"]
                for turn in case["turns"]:
                    result = agent.reply(user_id, case["id"], turn)
                    prompt_tokens += result["prompt_tokens"]
                    agent_tokens += result["tokens"]
                expected = case["expected_facts"]
                actual = agent.profile_store.facts(user_id)
                expected_total += len(expected)
                correct += sum(actual.get(key) == value for key, value in expected.items())
                false_facts += sum(key not in expected or expected[key] != value for key, value in actual.items())
            memory_bytes = sum(agent.memory_file_size(case["user_id"]) for case in cases)
            rows[label] = {
                "threshold": threshold,
                "precision": round(correct / (correct + false_facts), 3) if correct + false_facts else 0.0,
                "recall": round(correct / expected_total, 3) if expected_total else 0.0,
                "false_facts": false_facts,
                "memory_bytes": memory_bytes,
                "prompt_tokens_processed": prompt_tokens,
                "agent_tokens_only": agent_tokens,
            }
    return rows


def main() -> None:
    rows = run_ablation()
    if "--json" in sys.argv[1:]:
        print(json.dumps(rows, ensure_ascii=True))
        return
    print("| Mode | Threshold | Fact precision | Fact recall | False facts | Memory bytes | Prompt tokens processed |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, row in rows.items():
        print(f"| {label} | {row['threshold']:.1f} | {row['precision']:.1%} | {row['recall']:.1%} | "
              f"{row['false_facts']} | {row['memory_bytes']} | {row['prompt_tokens_processed']} |")


if __name__ == "__main__":
    main()
