import dataclasses
import hashlib
import json
from types import SimpleNamespace

import pytest

from src import inputs
from src.artifacts import ExperimentConfig


class ToyTokenizer:
    special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>"}
    bos_token_id = 1
    eos_token_id = 2

    def get_vocab(self):
        return {str(index): index for index in range(1000)}

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": True, "truncation": False, "return_attention_mask": False}
        return {"input_ids": [1] + [10 + int(value) for value in text.split()] + [2]}


def config(tmp_path):
    return ExperimentConfig(
        project_root=str(tmp_path), experiment={}, execution={},
        model={"source": "model", "revision": "a" * 40,
               "tokenizer": "tokenizer", "tokenizer_revision": "b" * 40},
        data={**{split: f"raw/{split}.jsonl" for split in inputs._SPLITS}, "max_sequence_length": 4},
        sweep={}, training={}, checkpointing={}, target_selection={},
        evaluation={"max_validation_samples": 2, "max_test_samples": 3}, quantization={}, predictor={},
    )


def write_documents(cfg):
    for split_index, split in enumerate(inputs._SPLITS):
        path = inputs._source_paths(cfg)[split]
        path.parent.mkdir(parents=True, exist_ok=True)
        documents = [{"id": f"{split}-{index}",
                      "text": f"{split_index * 20 + index * 4} {split_index * 20 + index * 4 + 1} "
                              f"{split_index * 20 + index * 4 + 2} {split_index * 20 + index * 4 + 3}"}
                     for index in range(2)]
        path.write_text("".join(json.dumps(doc) + "\n" for doc in documents), encoding="utf-8")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    write_documents(cfg)
    monkeypatch.setattr(inputs, "_load_tokenizer", lambda _: ToyTokenizer())
    monkeypatch.setattr(inputs, "_software_versions", lambda: {"transformers": "test", "tokenizers": "test"})
    root = tmp_path / "artifacts"
    return cfg, root, inputs.prepare_data(cfg, root)


def test_prepare_freezes_document_boundaries_masks_caps_and_eos(prepared):
    cfg, root, refs = prepared
    assert len(inputs.load_data(refs["forget_train"], root)) == 4
    assert len(inputs.load_data(refs["forget_validation"], root)) == 2
    assert len(inputs.load_data(refs["forget_test"], root)) == 3
    dataset = inputs.load_data(refs["forget_train"], root)
    assert dataset[0]["input_ids"] == [1, 10, 11, 12]
    assert dataset[1]["input_ids"] == [13, 2]
    assert dataset[1]["labels"] == [13, 2]
    assert dataset[1]["attention_mask"] == [1, 1]
    dataset[0]["labels"][0] = -100
    assert dataset[0]["labels"][0] == 1
    manifest = json.loads((root / refs["forget_train"].manifest).read_text())
    assert manifest["samples"][0]["document_id"] == manifest["samples"][1]["document_id"]
    assert manifest["samples"][2]["document_id"] != manifest["samples"][1]["document_id"]
    assert manifest["samples"][1]["start_token"] == 4
    assert manifest["preprocessing"]["legacy"] is True
    assert inputs.prepare_data(cfg, root) == refs


def test_document_disjointness_is_checked_before_tokenization(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    write_documents(cfg)
    paths = inputs._source_paths(cfg)
    overlap = {"id": "different-id-same-document", "text": "0 1 2 3"}
    paths["forget_validation"].write_text(json.dumps(overlap) + "\n", encoding="utf-8")
    def forbidden(_):
        pytest.fail("Tokenizer loaded before document leakage check")
    monkeypatch.setattr(inputs, "_load_tokenizer", forbidden)
    with pytest.raises(ValueError, match="overlapping source document"):
        inputs.prepare_data(cfg, tmp_path / "artifacts")


def test_duplicate_document_ids_and_modified_hashes_are_rejected(tmp_path):
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_text(json.dumps({"id": "same", "text": "first content"}) + "\n")
    second.write_text(json.dumps({"id": "same", "text": "different content"}) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        inputs._read_documents({"train": first, "test": second})
    second.write_text(json.dumps({"id": "unique", "text": "different content", "sha256": "wrong"}) + "\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        inputs._read_documents({"train": first, "test": second})


def test_loading_rejects_changed_samples_or_reference_identity(prepared):
    _, root, refs = prepared
    ref = refs["forget_test"]
    with pytest.raises(ValueError, match="identity mismatch"):
        inputs.load_data(dataclasses.replace(ref, tokenizer_revision="c" * 40), root)
    path = root / ref.location
    payload = json.loads(path.read_text())
    payload["samples"][0]["input_ids"][1] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="content identity mismatch"):
        inputs.load_data(ref, root)


def test_preprocessing_identity_changes_with_sample_cap(prepared):
    cfg, root, refs = prepared
    changed = dataclasses.replace(cfg, evaluation={**cfg.evaluation, "max_test_samples": 2})
    new_refs = inputs.prepare_data(changed, root)
    assert new_refs["forget_test"].dataset_id != refs["forget_test"].dataset_id
    assert new_refs["forget_train"] == refs["forget_train"]


def test_loading_rejects_changed_sample_document_membership(prepared):
    _, root, refs = prepared
    ref = refs["forget_validation"]
    path = root / ref.manifest
    manifest = json.loads(path.read_text())
    manifest["samples"][0]["document_id"] = "wrong-source-document"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="content identity mismatch"):
        inputs.load_data(ref, root)


def test_materialization_is_pinned_deterministic_and_disjoint(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg = dataclasses.replace(cfg, data={**cfg.data, "source": "muse-bench/MUSE-News", "revision": "c" * 40,
                                       "validation_documents": 2, "test_documents": 3})
    calls = []
    def load_dataset(repo, subset, *, split, revision):
        calls.append((repo, subset, split, revision))
        prefix = "forget" if split == "forget" else "retain"
        return [{"text": f"{prefix} whole article {index}"} for index in range(10)]
    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    paths = inputs.materialize_muse_splits(cfg)
    documents = inputs._read_documents(paths)
    assert [len(documents[f"forget_{part}"]) for part in ("train", "validation", "test")] == [5, 2, 3]
    assert calls == [("muse-bench/MUSE-News", "raw", split, "c" * 40) for split in ("forget", "retain1")]
    original = {split: path.read_bytes() for split, path in paths.items()}
    assert inputs.materialize_muse_splits(cfg) == paths
    assert all(path.read_bytes() == original[split] for split, path in paths.items())
    assert all(doc["revision"] == "c" * 40 and doc["source_index"] >= 0
               for docs in documents.values() for doc in docs)


def test_upstream_blank_and_duplicate_rows_are_excluded_with_pinned_audit(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg = dataclasses.replace(cfg, data={**cfg.data, "source": "muse-bench/MUSE-News", "revision": "c" * 40,
                                       "validation_documents": 2, "test_documents": 3})
    def source_rows(repo, subset, *, split, revision):
        offset = 0 if split == "forget" else 100
        rows = [{"text": f"{offset + index * 4} {offset + index * 4 + 1} {offset + index * 4 + 2} {offset + index * 4 + 3}"}
                for index in range(10)]
        rows.insert(2, {"text": ""})
        rows.insert(4, {"text": " \n\t "})
        rows.append(dict(rows[0]))
        return rows
    monkeypatch.setattr("datasets.load_dataset", source_rows)
    monkeypatch.setattr(inputs, "_load_tokenizer", lambda _: ToyTokenizer())
    monkeypatch.setattr(inputs, "_software_versions", lambda: {"transformers": "test", "tokenizers": "test"})
    root = tmp_path / "artifacts"
    refs = inputs.prepare_data(cfg, root)
    audit_path = inputs._source_audit_path(cfg)
    audit = json.loads(audit_path.read_text())
    assert audit["source"] == "muse-bench/MUSE-News" and audit["revision"] == "c" * 40
    for split in ("forget", "retain1"):
        details = audit["splits"][split]
        assert details["source_row_count"] == 13
        assert details["valid_unique_document_count"] == 10
        assert details["excluded_empty_count"] == 2 and details["excluded_duplicate_count"] == 1
        assert [row["id"] for row in details["excluded_empty_rows"]] == [f"{split}:2", f"{split}:4"]
        assert details["excluded_duplicate_rows"][0]["duplicate_of"].startswith(f"{split}:0:")
    assert (root / "data" / "source-audit.json").read_bytes() == audit_path.read_bytes()
    assert len(inputs.load_data(refs["forget_train"], root)) == 10


def test_upstream_non_string_rows_are_still_errors(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg = dataclasses.replace(cfg, data={**cfg.data, "revision": "c" * 40})
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: [{"text": None}])
    with pytest.raises(ValueError, match="non-string document"):
        inputs.materialize_muse_splits(cfg)


def test_external_prepared_blank_documents_remain_invalid(tmp_path):
    path = tmp_path / "external.jsonl"
    path.write_text(json.dumps({"id": "external-blank", "text": " \n "}) + "\n")
    with pytest.raises(ValueError, match="Invalid document"):
        inputs._read_documents({"forget_train": path})


def test_tokenizer_uses_verified_slow_legacy_identity(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    tokenizer_file = tmp_path / "tokenizer.model"
    tokenizer_file.write_bytes(b"same-sentencepiece-model")
    cfg = dataclasses.replace(cfg, model={**cfg.model, "tokenizer_model_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()})
    calls = []
    def tokenizer_loader(*args, **kwargs):
        calls.append(kwargs)
        return ToyTokenizer()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", tokenizer_loader)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained",
                        lambda *args, **kwargs: SimpleNamespace(vocab_size=1000, bos_token_id=1, eos_token_id=2))
    monkeypatch.setattr("transformers.utils.hub.cached_file", lambda *args, **kwargs: str(tokenizer_file))
    assert isinstance(inputs._load_tokenizer(cfg), ToyTokenizer)
    assert calls[0]["use_fast"] is False and calls[0]["legacy"] is True
    tokenizer_file.write_bytes(b"changed-model")
    with pytest.raises(ValueError, match="expected original tokenizer"):
        inputs._load_tokenizer(cfg)


def test_partial_source_bundle_is_an_explicit_error(tmp_path):
    cfg = config(tmp_path)
    path = inputs._source_paths(cfg)["forget_train"]
    path.parent.mkdir(parents=True)
    path.write_text('{"id":"one","text":"one two"}\n')
    with pytest.raises(ValueError, match="Incomplete document split bundle"):
        inputs.prepare_data(cfg, tmp_path / "artifacts")
