from __future__ import annotations

import hashlib
import pytest

from tools import prepare_example_data, prepare_full_llm


def test_verified_corpus_helper_reuses_a_cached_checksum(tmp_path, monkeypatch):
    text = "".join(chr(code) for code in range(32, 97))
    raw = text.encode("utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "tinyshakespeare.txt").write_bytes(raw)
    monkeypatch.setattr(prepare_example_data, "TEXT_SHA256", hashlib.sha256(raw).hexdigest())

    actual_text, vocabulary, digest = prepare_example_data.verified_tiny_shakespeare(cache)

    assert actual_text == text
    assert vocabulary == sorted(text) and len(vocabulary) == 65
    assert digest == hashlib.sha256(raw).hexdigest()


def test_full_corpus_split_offsets_window_counts_and_omitted_tails():
    text = "x" * 1_115_394

    splits = prepare_full_llm.split_corpus(text)

    assert {
        name: (item["start"], item["end"], item["windows"], item["omittedTailCharacters"])
        for name, item in splits.items()
    } == {
        "train": (0, 892_315, 6_971, 26),
        "validation": (892_315, 1_003_854, 871, 50),
        "test": (1_003_854, 1_115_394, 871, 51),
    }
    for item in splits.values():
        segment = item["text"]
        assert all(len(segment[start : start + 129]) == 129 for start in item["starts"])
        assert item["starts"] == list(range(0, len(segment) - 128, 128))
        assert item["starts"][-1] + 129 + item["omittedTailCharacters"] == len(segment)


def test_window_selection_never_crosses_split_boundaries_or_keeps_incomplete_windows():
    text = "A" * 1_200 + "B" * 150 + "C" * 150

    splits = prepare_full_llm.split_corpus(text)

    assert splits["train"]["text"] == text[:1_200]
    assert splits["validation"]["text"] == text[1_200:1_350]
    assert splits["test"]["text"] == text[1_350:]
    # This deliberately exercises the actual 80/10/10 boundaries as well as a short final tail.
    for item in splits.values():
        segment = item["text"]
        assert all(segment[start : start + 129] == segment[start] * 129 for start in item["starts"])
        assert len(segment) - (item["starts"][-1] + 129) == item["omittedTailCharacters"]


def test_full_payload_records_all_offsets_and_uses_full_sorted_vocabulary():
    alphabet = [chr(code) for code in range(32, 97)]
    text = "".join(alphabet[index % len(alphabet)] for index in range(1_115_394))

    payload = prepare_full_llm.full_payload(text, "fixture-sha256", sorted(set(text)))

    assert payload["provenance"]["retainedCharacters"] == len(text)
    assert payload["provenance"]["vocabularySize"] == 65
    assert payload["provenance"]["splits"]["train"]["start"] == 0
    assert payload["provenance"]["splits"]["test"]["end"] == len(text)
    assert len(payload["starts"]["train"]) == 6_971
    assert len(payload["starts"]["validation"]) == 871
    assert len(payload["starts"]["test"]) == 871


def test_preparation_refuses_to_overwrite_an_existing_project(tmp_path, monkeypatch):
    destination = tmp_path / "existing-project"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("untouched")
    normalized_alias = tmp_path / "missing-parent" / ".." / "existing-project"
    assert not normalized_alias.exists()
    monkeypatch.setattr(
        prepare_full_llm,
        "verified_tiny_shakespeare",
        lambda _cache: pytest.fail("existing destination must be rejected before reading sources"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_full_llm.prepare_project(normalized_alias, tmp_path / "cache")

    assert sentinel.read_text() == "untouched"
    assert not (tmp_path / "missing-parent").exists()
    assert prepare_full_llm.TRAINING_CONFIG == {
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.001,
        "seed": 0,
        "publish_every_steps": 100,
    }


def test_copy_project_skips_ignored_environment_symlinks(tmp_path):
    source = tmp_path / "source"
    (source / ".venv" / "lib").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "shared").write_text("ignored")
    (source / ".venv" / "lib" / "shared").symlink_to(outside / "shared")
    destination = tmp_path / "destination"

    prepare_full_llm._copy_project(source, destination)

    assert not (destination / ".venv").exists()


def test_copy_project_rejects_symlinks_in_included_resources(tmp_path):
    source = tmp_path / "source"
    resource = source / "datasets" / "llm.tokens-1.0.0"
    resource.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of the project")
    (resource / "data.json").symlink_to(outside)

    with pytest.raises(ValueError, match="source example contains a symlink"):
        prepare_full_llm._copy_project(source, tmp_path / "destination")
