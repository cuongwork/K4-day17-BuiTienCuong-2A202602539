import json
from pathlib import Path
import subprocess
import sys


def test_bonus_ablation_reports_fewer_false_facts_with_threshold() -> None:
    script = Path(__file__).with_name("eval_bonus.py")
    result = subprocess.run([sys.executable, str(script), "--json"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    protected, unfiltered = rows["protected"], rows["unfiltered"]
    assert protected["precision"] > unfiltered["precision"]
    assert protected["recall"] >= unfiltered["recall"]
    assert protected["false_facts"] < unfiltered["false_facts"]
    assert protected["false_facts"] == 0
