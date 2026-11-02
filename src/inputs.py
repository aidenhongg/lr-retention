"""Freeze document membership before producing shared causal-LM samples."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any

from .artifacts import DatasetRef, ExperimentConfig, publish_record, resolve_artifact_path, stable_id


_SPLITS = tuple(f"{domain}_{partition}" for domain in ("forget", "retain")
                for partition in ("train", "validation", "test"))


def _pinned_revision(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{name} must be an immutable 40-character commit SHA")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _source_paths(config: ExperimentConfig) -> dict[str, Path]:
    paths = {}
    for split in _SPLITS:
        path = Path(config.data[split])
        path = path if path.is_absolute() else Path(config.project_root) / path
        if path.suffix != ".jsonl":
            raise ValueError("Document-preserving inputs must use .jsonl files with id/text fields")
        paths[split] = path
    if len({p.resolve() for p in paths.values()}) != len(paths):
        raise ValueError("Every data split requires a distinct source file")
    return paths


def _document_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _write_once(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != payload:
                raise FileExistsError(f"Conflicting fixed data artifact: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _software_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")}


def _source_audit_path(config: ExperimentConfig) -> Path:
    return Path(config.project_root) / "data" / "source-audit.json"


def materialize_muse_splits(config: ExperimentConfig) -> dict[str, Path]:
    """Download pinned whole documents and deterministically reserve held-out sets.

    Validation and test document counts differ from the later sample caps: no
    fragment of a held-out document can enter unlearning training.
    """
    from datasets import load_dataset

    paths = _source_paths(config)
    source = config.data.get("source", "muse-bench/MUSE-News")
    revision = _pinned_revision(config.data.get("revision"), "data.revision")
    validation_count = _positive_int(config.data.get("validation_documents", 16), "validation_documents")
    test_count = _positive_int(config.data.get("test_documents", 32), "test_documents")
    all_documents: dict[str, list[dict[str, Any]]] = {}
    seen_hashes: dict[str, tuple[str, str]] = {}
    audit = {"schema_version": 1, "source": source, "revision": revision, "splits": {}}
    for domain, source_split in (("forget", "forget"), ("retain", "retain1")):
        source_data = load_dataset(source, "raw", split=source_split, revision=revision)
        documents = []
        excluded_empty, excluded_duplicates = [], []
        row_count = 0
        for index, row in enumerate(source_data):
            row_count += 1
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Invalid non-string document {source_split}:{index}")
            if not text.strip():
                excluded_empty.append({"id": f"{source_split}:{index}", "source_index": index})
                continue
            digest = _document_hash(text)
            document_id = f"{source_split}:{index}:{digest}"
            if digest in seen_hashes:
                previous_domain, previous_id = seen_hashes[digest]
                if previous_domain != domain:
                    raise ValueError("A source document occurs in both forget and retain corpora")
                excluded_duplicates.append({"id": document_id, "source_index": index,
                                            "sha256": digest, "duplicate_of": previous_id})
                continue
            seen_hashes[digest] = (domain, document_id)
            documents.append({"id": document_id, "text": text,
                              "sha256": digest, "source": source, "revision": revision,
                              "source_split": source_split, "source_index": index})
        documents.sort(key=lambda document: document["sha256"])
        if len(documents) <= validation_count + test_count:
            raise ValueError(f"Insufficient {domain} documents for disjoint training/validation/test")
        all_documents[f"{domain}_validation"] = documents[:validation_count]
        all_documents[f"{domain}_test"] = documents[validation_count:validation_count + test_count]
        all_documents[f"{domain}_train"] = documents[validation_count + test_count:]
        audit["splits"][source_split] = {
            "source_row_count": row_count, "valid_unique_document_count": len(documents),
            "excluded_empty_count": len(excluded_empty), "excluded_empty_rows": excluded_empty,
            "excluded_duplicate_count": len(excluded_duplicates), "excluded_duplicate_rows": excluded_duplicates,
        }
    _write_once(_source_audit_path(config), _json(audit))
    for split, documents in all_documents.items():
        _write_once(paths[split], "".join(_json(document) + "\n" for document in documents))
    return paths


def _read_documents(paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    documents_by_split = {}
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    for split, path in paths.items():
        documents = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                document = json.loads(line)
                if not isinstance(document, dict) or not isinstance(document.get("id"), str):
                    raise ValueError(f"Missing document ID at {path}:{line_number}")
                text = document.get("text")
                if not isinstance(text, str) or not text.strip() or not document["id"]:
                    raise ValueError(f"Invalid document at {path}:{line_number}")
                digest = _document_hash(text)
                if "sha256" in document and document["sha256"] != digest:
                    raise ValueError(f"Document content hash mismatch at {path}:{line_number}")
                if document["id"] in seen_ids or digest in seen_hashes:
                    raise ValueError(f"Duplicate or overlapping source document in {split}")
                seen_ids[document["id"]] = split
                seen_hashes[digest] = split
                documents.append({**document, "sha256": digest})
        if not documents:
            raise ValueError(f"Empty source split: {split}")
        documents_by_split[split] = documents
    return documents_by_split


def _load_tokenizer(config: ExperimentConfig):
    from transformers import AutoConfig, AutoTokenizer

    tokenizer_id = config.model["tokenizer"]
    tokenizer_revision = _pinned_revision(config.model.get("tokenizer_revision"), "model.tokenizer_revision")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision,
                                              use_fast=False, legacy=True, trust_remote_code=False)
    expected_hash = config.model.get("tokenizer_model_sha256")
    if expected_hash:
        from transformers.utils.hub import cached_file
        model_file = cached_file(tokenizer_id, "tokenizer.model", revision=tokenizer_revision)
        if hashlib.sha256(Path(model_file).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Tokenizer model file does not match the expected original tokenizer")
    model_revision = _pinned_revision(config.model.get("revision"), "model.revision")
    local_model = Path(config.model.get("path", ""))
    if not local_model.is_absolute():
        local_model = Path(config.project_root) / local_model
    model_source = str(local_model) if (local_model / "config.json").is_file() else config.model["source"]
    model_config = AutoConfig.from_pretrained(model_source, revision=model_revision, trust_remote_code=False)
    vocab = tokenizer.get_vocab()
    if not vocab or max(vocab.values()) >= model_config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not fit the original model")
    for name in ("bos_token_id", "eos_token_id"):
        expected = getattr(model_config, name, None)
        if expected is not None and expected != getattr(tokenizer, name, None):
            raise ValueError(f"Tokenizer/model {name} mismatch")
    return tokenizer


def prepare_data(config: ExperimentConfig, artifact_root: Path) -> dict[str, DatasetRef]:
    """Validate all whole documents, then freeze tokenization and sample manifests."""
    paths = _source_paths(config)
    present = [path.exists() for path in paths.values()]
    if not any(present):
        paths = materialize_muse_splits(config)
    elif not all(present):
        raise ValueError("Incomplete document split bundle; refusing to mix old and regenerated splits")
    documents_by_split = _read_documents(paths)
    if "revision" in config.data:
        revision = _pinned_revision(config.data["revision"], "data.revision")
        source = config.data.get("source", "muse-bench/MUSE-News")
        for split, documents in documents_by_split.items():
            source_split = "forget" if split.startswith("forget_") else "retain1"
            if any(doc.get("source") != source or doc.get("revision") != revision
                   or doc.get("source_split") != source_split for doc in documents):
                raise ValueError(f"Source provenance does not match pinned corpus for {split}")
    audit_path = _source_audit_path(config)
    if audit_path.is_file():
        audit_payload = audit_path.read_text(encoding="utf-8")
        audit = json.loads(audit_payload)
        if "revision" in config.data and (audit.get("revision") != config.data["revision"]
                or audit.get("source") != config.data.get("source", "muse-bench/MUSE-News")):
            raise ValueError("Source audit does not match pinned corpus")
        _write_once(resolve_artifact_path(artifact_root, "data/source-audit.json"), audit_payload)
    max_length = _positive_int(config.data["max_sequence_length"], "max_sequence_length")
    if max_length < 2:
        raise ValueError("Causal-LM samples need at least two tokens")
    tokenizer = _load_tokenizer(config)
    preprocessing = {
        "algorithm": "per_document_nonoverlapping_chunks_v1", "max_sequence_length": max_length,
        "add_special_tokens": True, "use_fast": False, "legacy": True, "drop_chunks_shorter_than": 2,
        "tokenizer_id": config.model["tokenizer"], "tokenizer_revision": config.model["tokenizer_revision"],
        "tokenizer_fingerprint": stable_id({"vocab": tokenizer.get_vocab(),
                                             "special_tokens": tokenizer.special_tokens_map}),
        "software_versions": _software_versions(),
    }
    refs = {}
    for split, documents in documents_by_split.items():
        partition = split.rsplit("_", 1)[1]
        cap = config.evaluation.get(f"max_{partition}_samples") if partition != "train" else None
        if cap is not None:
            _positive_int(cap, f"max_{partition}_samples")
        settings = {**preprocessing, "sample_cap": cap}
        preprocessing_id = stable_id(settings)
        samples, sample_manifest = [], []
        for document in documents:
            ids = tokenizer(document["text"], add_special_tokens=True, truncation=False,
                            return_attention_mask=False)["input_ids"]
            if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("Tokenizer returned invalid input IDs")
            for start in range(0, len(ids), max_length):
                chunk = ids[start:start + max_length]
                if len(chunk) < 2:
                    continue
                samples.append({"input_ids": chunk, "attention_mask": [1] * len(chunk), "labels": list(chunk)})
                sample_manifest.append({"document_id": document["id"], "document_sha256": document["sha256"],
                                        "start_token": start, "end_token": start + len(chunk)})
                if cap is not None and len(samples) >= cap:
                    break
            if cap is not None and len(samples) >= cap:
                break
        if not samples:
            raise ValueError(f"No valid causal-LM samples for {split}")
        document_manifest = [{key: value for key, value in doc.items() if key != "text"} for doc in documents]
        identity = {"split": split, "preprocessing_id": preprocessing_id,
                    "samples_sha256": stable_id(samples), "documents": document_manifest,
                    "sample_manifest_sha256": stable_id(sample_manifest)}
        dataset_id = stable_id(identity)
        location = f"data/{dataset_id}/samples.json"
        manifest_location = f"data/{dataset_id}/manifest.json"
        manifest = {"schema_version": 1, "dataset_id": dataset_id, **identity,
                    "preprocessing": settings, "samples": sample_manifest, "sample_count": len(samples)}
        _write_once(resolve_artifact_path(artifact_root, location), _json({"schema_version": 1, "samples": samples}))
        _write_once(resolve_artifact_path(artifact_root, manifest_location), _json(manifest))
        reference = DatasetRef(dataset_id, location, split, manifest_location, config.model["tokenizer"],
                               config.model["tokenizer_revision"], preprocessing_id, max_length)
        publish_record(reference, artifact_root)
        refs[split] = reference
    return refs


def load_data(reference: DatasetRef, artifact_root: Path):
    """Load a verified prepared dataset; no tokenization happens in a run."""
    from torch.utils.data import Dataset

    payload = json.loads(resolve_artifact_path(artifact_root, reference.location).read_text(encoding="utf-8"))
    manifest = json.loads(resolve_artifact_path(artifact_root, reference.manifest).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported prepared-data schema")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples or len(samples) != manifest.get("sample_count"):
        raise ValueError("Prepared dataset has no samples or an invalid sample count")
    settings = manifest["preprocessing"]
    if (manifest.get("dataset_id") != reference.dataset_id or manifest.get("split") != reference.split
            or manifest.get("preprocessing_id") != reference.preprocessing_id
            or stable_id(settings) != reference.preprocessing_id
            or settings.get("tokenizer_id") != reference.tokenizer_id
            or settings.get("tokenizer_revision") != reference.tokenizer_revision
            or settings.get("max_sequence_length") != reference.max_sequence_length):
        raise ValueError("Prepared data reference or preprocessing identity mismatch")
    identity = {"split": reference.split, "preprocessing_id": reference.preprocessing_id,
                "samples_sha256": stable_id(samples), "documents": manifest["documents"],
                "sample_manifest_sha256": stable_id(manifest["samples"])}
    if (identity["samples_sha256"] != manifest.get("samples_sha256")
            or identity["sample_manifest_sha256"] != manifest.get("sample_manifest_sha256")
            or len(manifest["samples"]) != len(samples)
            or stable_id(identity) != reference.dataset_id):
        raise ValueError("Prepared dataset content identity mismatch")
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != {"input_ids", "attention_mask", "labels"}:
            raise ValueError("Invalid causal-LM sample fields")
        ids, attention, labels = sample["input_ids"], sample["attention_mask"], sample["labels"]
        if (not isinstance(ids, list) or not 2 <= len(ids) <= reference.max_sequence_length
                or any(type(token) is not int or token < 0 for token in ids)
                or attention != [1] * len(ids) or labels != ids):
            raise ValueError("Invalid causal-LM tokens or masks")

    class PreparedDataset(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return {key: list(value) for key, value in samples[index].items()}

    return PreparedDataset()
