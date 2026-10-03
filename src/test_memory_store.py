from pathlib import Path

import pytest

from memory_store import (
    DEFAULT_PROFILE_CONFIDENCE_THRESHOLD,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_candidates,
    extract_profile_updates,
    summarize_messages,
)


def test_estimate_tokens_handles_empty_and_nonempty_text() -> None:
    assert estimate_tokens("  \n ") == 0
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("a") == 1


def test_profile_store_reads_writes_edits_and_upserts(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("dungct") == ""
    assert store.file_size("dungct") == 0
    path = store.write_text("dungct", "# User\n\n- location: Huế\n")
    assert path == tmp_path / "profiles" / "dungct" / "User.md"
    assert store.edit_text("dungct", "Huế", "Đà Nẵng") is True
    assert store.edit_text("dungct", "missing", "x") is False
    store.upsert_fact("dungct", "profession", "backend engineer")
    store.upsert_fact("dungct", "profession", "MLOps engineer")
    assert store.facts("dungct")["profession"] == "MLOps engineer"
    assert "backend engineer" not in store.read_text("dungct")
    assert store.file_size("dungct") == path.stat().st_size


def test_upsert_preserves_manual_markdown_and_replaces_only_its_fact(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.write_text("lan", "# Hồ sơ của Lan\n\nGhi chú thủ công cần giữ.\n\n- location: Huế\n")
    store.upsert_fact("lan", "location", "Đà Nẵng")
    store.upsert_fact("lan", "profession", "MLOps engineer")
    content = store.read_text("lan")
    assert "# Hồ sơ của Lan" in content
    assert "Ghi chú thủ công cần giữ." in content
    assert content.count("- location:") == 1
    assert "- location: Đà Nẵng" in content
    assert "- profession: MLOps engineer" in content


def test_profile_path_cannot_escape_root(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    path = store.path_for("../../someone")
    assert path.parent.parent == tmp_path / "profiles"
    with pytest.raises(ValueError):
        store.path_for("  ")


def test_extract_profile_facts_and_ignores_noise() -> None:
    assert extract_profile_updates("Mình tên là DũngCT.")["name"] == "DũngCT"
    assert extract_profile_updates("Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")["location"] == "Đà Nẵng"
    assert extract_profile_updates("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")["profession"] == "MLOps engineer"
    assert extract_profile_updates("Mình đang ở Huế chứ không còn ở Đà Nẵng.")["location"] == "Huế"
    assert extract_profile_updates("Nơi ở hiện tại là Đà Nẵng.")["location"] == "Đà Nẵng"
    assert extract_profile_updates("Mình muốn bạn trả lời ngắn gọn thành 3 bullet.")["response_style"] == "ngắn gọn, 3 bullet"
    assert extract_profile_updates("Đồ uống yêu thích là cà phê sữa đá.")["favorite_drink"] == "cà phê sữa đá"
    assert extract_profile_updates("Món ăn yêu thích là mì Quảng.")["favorite_food"] == "mì Quảng"
    assert extract_profile_updates("Bạn có biết DũngCT không?") == {}
    assert extract_profile_updates("Mình tên là ai?") == {}
    assert extract_profile_updates("Mình ở Hải Phòng.")["location"] == "Hải Phòng"
    assert extract_profile_updates("Mình đang làm data analyst.")["profession"] == "data analyst"
    assert extract_profile_updates("Món ăn yêu thích là phở bò.")["favorite_food"] == "phở bò"
    assert "location" not in extract_profile_updates("Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày.")
    assert "profession" not in extract_profile_updates("Mình đùa rằng chuyển sang product manager.")


def test_speculative_facts_are_not_treated_as_profile_updates() -> None:
    assert extract_profile_updates("Mình có thể chuyển sang product manager.") == {}
    assert extract_profile_updates("Nếu mình ở Hà Nội thì sẽ đi làm gần nhà.") == {}
    assert extract_profile_updates("Mình đang cân nhắc chuyển sang data analyst.") == {}
    assert extract_profile_updates("Mình đang làm MLOps engineer.")["profession"] == "MLOps engineer"


def test_confirmed_fact_survives_adjacent_speculation_and_reported_speech() -> None:
    assert extract_profile_updates("Mình ở Huế, nhưng có thể tháng sau chuyển đi.")["location"] == "Huế"
    assert extract_profile_updates("Trước đây mình ở Huế, hiện mình ở Đà Nẵng.")["location"] == "Đà Nẵng"
    assert extract_profile_updates("Anh ấy nói mình ở Hà Nội.") == {}


def test_confidence_threshold_is_explicit_and_configurable() -> None:
    speculative = "Mình có thể chuyển sang product manager."
    confirmed = "Mình đang làm MLOps engineer."
    assert extract_profile_candidates(speculative)["profession"].confidence < DEFAULT_PROFILE_CONFIDENCE_THRESHOLD
    assert extract_profile_candidates(confirmed)["profession"].confidence >= DEFAULT_PROFILE_CONFIDENCE_THRESHOLD
    assert extract_profile_updates(speculative, min_confidence=0.3)["profession"] == "product manager"
    with pytest.raises(ValueError):
        extract_profile_updates(confirmed, min_confidence=1.1)


def test_summary_and_compaction_keep_recent_messages() -> None:
    messages = [{"role": "user", "content": f"message {i} " + "x" * 60} for i in range(8)]
    summary = summarize_messages(messages, max_items=3)
    assert "message 7" in summary
    assert "message 0" not in summary

    memory = CompactMemoryManager(threshold_tokens=55, keep_messages=2)
    for message in messages:
        memory.append("a", **message)
    context = memory.context("a")
    assert memory.compaction_count("a") > 0
    assert context["compactions"] == memory.compaction_count("a")
    assert context["messages"] == messages[-2:]
    assert context["summary"]
    assert memory.compaction_count("b") == 0
    assert memory.context("b")["messages"] == []


def test_repeated_compaction_keeps_early_important_fact_with_bounded_summary() -> None:
    memory = CompactMemoryManager(threshold_tokens=120, keep_messages=2)
    memory.append("long", "user", "Lưu ý quan trọng: deadline thứ Sáu cho bản demo.")
    for index in range(30):
        memory.append("long", "user", f"nhiễu lượt {index} " + "x" * 90)
    context = memory.context("long")
    assert memory.compaction_count("long") >= 2
    assert "deadline thứ Sáu" in context["summary"]
    assert len(context["summary"]) <= 160
