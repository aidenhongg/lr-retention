"""Copy selected/final weights and evidence to an attached RunPod network volume."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.artifacts import (CheckpointRef, RunResult, RunSpec, SelectionResult,
                           read_record, resolve_artifact_path)


HASH_BLOCK_BYTES = 8 * 1024 * 1024


class ArchiveError(RuntimeError):
    """Sanitized archive failure; no credential or provider response text."""


def _records(root: Path, kind: str, expected_type):
    return [(path.relative_to(root).as_posix(), read_record(path, expected_type))
            for path in sorted((root / "records" / kind).glob("*.json"))]


def _safe_file(root: Path, path: Path) -> None:
    if not path.resolve().is_relative_to(root):
        raise ArchiveError("Archive input escapes the artifact root")
    relative = path.relative_to(root)
    if any((root.joinpath(*relative.parts[:index])).is_symlink() for index in range(1, len(relative.parts) + 1)):
        raise ArchiveError("Archive inputs cannot contain symbolic links")


def _validate_checkpoint(root: Path, reference: CheckpointRef) -> Path:
    if reference.model.source not in ("artifact", "local"):
        raise ArchiveError("Checkpoint archive requires local artifact-relative weights")
    path = resolve_artifact_path(root, reference.model.location)
    _safe_file(root, root / reference.model.location)
    required = ("config.json", "tokenizer_config.json", "checkpoint_complete.json")
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise ArchiveError("A selected or final checkpoint bundle is incomplete")
    marker = json.loads((path / "checkpoint_complete.json").read_text(encoding="utf-8"))
    if marker.get("run_id") != reference.run_id or marker.get("step") != reference.step:
        raise ArchiveError("Checkpoint completion marker disagrees with its reference")
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        raise ArchiveError("Checkpoint bundle has no model weights")
    for index in path.glob("*.index.json"):
        weights = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not isinstance(weights, dict) or not weights:
            raise ArchiveError("Checkpoint shard index is invalid")
        for filename in set(weights.values()):
            shard = resolve_artifact_path(path, filename)
            if not shard.is_file():
                raise ArchiveError("Checkpoint shard listed in its index is missing")
    return path


def plan_archive(root: Path) -> dict:
    """List retained weights and evidence without hashing or changing any files."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ArchiveError("Artifact root is not a directory")
    specs = {record.run_id: record for _, record in _records(root, "run_spec", RunSpec)}
    outcomes = _records(root, "run_result", RunResult)
    selections = _records(root, "selection_result", SelectionResult)
    refs = [record for _, record in _records(root, "checkpoint_ref", CheckpointRef)]
    for _, outcome in outcomes:
        refs.extend(outcome.checkpoints)
    kept: dict[str, dict] = {}

    def keep(reference, role):
        location = reference.model.location
        if location in kept:
            if kept[location]["reference"] != dataclasses.asdict(reference):
                raise ArchiveError("Checkpoint location has conflicting references")
            if role not in kept[location]["roles"]:
                kept[location]["roles"].append(role)
            return
        _validate_checkpoint(root, reference)
        kept[location] = {"location": location, "run_id": reference.run_id, "step": reference.step,
                          "model_id": reference.model.model_id, "roles": [role],
                          "reference": dataclasses.asdict(reference)}

    for _, selection in selections:
        if selection.status != "complete":
            continue
        for entry in selection.selections:
            if entry.checkpoint is not None:
                keep(entry.checkpoint, "selected")
    recorded_runs = set()
    for _, outcome in outcomes:
        recorded_runs.add(outcome.run_id)
        if not outcome.checkpoints:
            if outcome.status == "complete":
                raise ArchiveError("A completed run has no saved final checkpoint")
            candidates = [reference for reference in refs if reference.run_id == outcome.run_id]
            if candidates:
                keep(max(candidates, key=lambda checkpoint: checkpoint.step), "last_saved_incomplete_run")
            continue
        final = max(outcome.checkpoints, key=lambda checkpoint: checkpoint.step)
        expected = specs.get(outcome.run_id)
        if outcome.status == "complete":
            if final.step != outcome.last_step or (expected is not None and final.step != max(expected.checkpoint_steps)):
                raise ArchiveError("A completed run is missing its configured final checkpoint")
            keep(final, "final")
        else:
            keep(final, "last_saved_incomplete_run")
    missing_run_ids = sorted((set(specs) | {reference.run_id for reference in refs}) - recorded_runs)
    for run_id in missing_run_ids:
        candidates = [reference for reference in refs if reference.run_id == run_id]
        if candidates:
            keep(max(candidates, key=lambda checkpoint: checkpoint.step), "last_saved_without_outcome")
    incomplete_run_ids = sorted(outcome.run_id for _, outcome in outcomes if outcome.status != "complete")
    experiment_status = "complete" if outcomes and not missing_run_ids and not incomplete_run_ids else "incomplete"
    all_checkpoint_dirs = {reference.model.location for reference in refs if reference.model.source in ("artifact", "local")}
    run_directories = {spec.output_dir for spec in specs.values()}
    checkpoint_bytes = 0
    files = []
    excluded_names = {".env", "credentials", "credentials.json", "secrets", "secrets.json"}
    for directory, names, filenames in os.walk(root, followlinks=False):
        directory = Path(directory)
        names[:] = [name for name in names if not name.startswith(".") and name != "archive"]
        for name in names:
            _safe_file(root, directory / name)
        for name in filenames:
            path = directory / name
            relative = path.relative_to(root).as_posix()
            if name.startswith(".") or name.endswith(".tmp") or name.lower() in excluded_names or path.suffix.lower() in (".pem", ".key"):
                continue
            _safe_file(root, path)
            in_kept = any(relative.startswith(location.rstrip("/") + "/") for location in kept)
            in_any_checkpoint = (relative.startswith("checkpoints/") or
                                 any(relative.startswith(location.rstrip("/") + "/")
                                     for location in all_checkpoint_dirs | run_directories))
            if in_any_checkpoint and not in_kept:
                continue
            if not in_kept and path.suffix in (".safetensors", ".bin", ".pt", ".pth", ".ckpt"):
                continue
            size = path.stat().st_size
            files.append({"path": relative, "bytes": size})
            if in_kept:
                checkpoint_bytes += size
    files.sort(key=lambda item: item["path"])
    total_bytes = sum(item["bytes"] for item in files)
    minimum_gb = max(10, math.ceil(total_bytes / 1_000_000_000 * 1.1) + 1)
    return {"schema_version": 1, "artifact_name": root.name,
            "experiment_status": experiment_status, "missing_run_ids": missing_run_ids,
            "incomplete_run_ids": incomplete_run_ids, "archive_scope": "weights_and_evidence" if kept else "evidence_only",
            "checkpoint_directories": sorted(kept.values(), key=lambda item: item["location"]),
            "selection_records": [{"path": path, "status": selection.status} for path, selection in selections],
            "run_outcomes": [{"path": path, "run_id": outcome.run_id, "status": outcome.status} for path, outcome in outcomes],
            "files": files, "file_count": len(files), "total_bytes": total_bytes,
            "checkpoint_bytes": checkpoint_bytes, "evidence_bytes": total_bytes - checkpoint_bytes,
            "minimum_volume_size_gb": minimum_gb,
            "estimated_minimum_monthly_usd": min(minimum_gb, 1000) * 0.07 + max(0, minimum_gb - 1000) * 0.05}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _publish_manifest(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ArchiveError("Existing immutable archive manifest conflicts") from None
    finally:
        temporary.unlink(missing_ok=True)


def _destination(root: Path, destination: Path) -> Path:
    absolute = Path(destination).absolute()
    if any(path.is_symlink() for path in (absolute, *absolute.parents)):
        raise ArchiveError("Archive destination cannot contain symbolic links")
    destination = absolute.resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ArchiveError("Archive source and destination must be separate directory trees")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _copy_verified(source: Path, target: Path, expected_size: int) -> str:
    before = source.stat()
    if before.st_size != expected_size:
        raise ArchiveError("An archive input changed after planning")
    target.parent.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in (target, *target.parents)):
        raise ArchiveError("Archive destination cannot contain symbolic links")
    if target.exists():
        digest = _sha256(source)
    else:
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        digest_builder = hashlib.sha256()
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                while block := reader.read(HASH_BLOCK_BYTES):
                    digest_builder.update(block)
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            digest = digest_builder.hexdigest()
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ArchiveError("An archive input changed during copying")
    if target.stat().st_size != expected_size or _sha256(target) != digest:
        raise ArchiveError("Destination file size or SHA256 verification failed")
    return digest


def stage_final_checkpoints(root: Path, destination: Path, *, include_selected: bool = False) -> dict:
    """Stage completed finals and optional frozen selections; publish no manifest.

    Existing same-size files skip byte reads here. Final archive_to_volume always
    performs full source/destination SHA256 verification before its final manifest.
    """
    root = Path(root).resolve()
    specs = {record.run_id: record for _, record in _records(root, "run_spec", RunSpec)}
    staged, files, completed_runs = {}, [], set()

    def add(reference, role):
        location = reference.model.location
        if location in staged:
            if staged[location]["model_id"] != reference.model.model_id:
                raise ArchiveError("Checkpoint location has conflicting references")
            if role not in staged[location]["roles"]:
                staged[location]["roles"].append(role)
            return
        directory = _validate_checkpoint(root, reference)
        staged[location] = {"run_id": reference.run_id, "step": reference.step,
                            "model_id": reference.model.model_id, "location": location, "roles": [role]}
        for parent, directories, filenames in os.walk(directory, followlinks=False):
            for name in directories:
                _safe_file(root, Path(parent) / name)
            for name in filenames:
                path = Path(parent) / name
                _safe_file(root, path)
                files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size})

    for _, outcome in _records(root, "run_result", RunResult):
        if outcome.status != "complete":
            continue
        if not outcome.checkpoints:
            raise ArchiveError("A completed run has no saved final checkpoint")
        final = max(outcome.checkpoints, key=lambda checkpoint: checkpoint.step)
        expected = specs.get(outcome.run_id)
        if final.step != outcome.last_step or (expected is not None and final.step != max(expected.checkpoint_steps)):
            raise ArchiveError("A completed run is missing its configured final checkpoint")
        add(final, "final")
        completed_runs.add(outcome.run_id)
    if include_selected:
        for _, selection in _records(root, "selection_result", SelectionResult):
            if selection.status != "complete":
                continue
            for entry in selection.selections:
                if entry.checkpoint is not None and entry.checkpoint.run_id in completed_runs:
                    add(entry.checkpoint, "selected")
    destination = _destination(root, destination)
    remaining = sum(item["bytes"] for item in files if not (destination / item["path"]).exists())
    if shutil.disk_usage(destination).free < remaining + 1024 * 1024:
        raise ArchiveError("Insufficient free network-volume space for staged checkpoints")
    for item in sorted(files, key=lambda item: item["path"]):
        _safe_file(destination, destination / item["path"])
        target = resolve_artifact_path(destination, item["path"])
        if target.exists():
            if not target.is_file() or target.stat().st_size != item["bytes"]:
                raise ArchiveError("Existing staged destination file has an unexpected type or size")
            print(json.dumps({"event": "staged_existing_file_verification_deferred", **item}), flush=True)
            continue
        _copy_verified(resolve_artifact_path(root, item["path"]), target, item["bytes"])
        print(json.dumps({"event": "staged_final_file", **item}), flush=True)
    return {"status": "staged", "checkpoints": list(staged.values()), "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files), "destination": str(destination)}


def archive_to_volume(root: Path, destination: Path) -> dict:
    """Copy atomically, rehash destination bytes, and publish the manifest last.

    The caller supplies the mounted network-volume directory. Original files and
    unselected weights are never deleted, including after verification succeeds.
    """
    root = Path(root).resolve()
    plan = plan_archive(root)
    destination = _destination(root, destination)
    remaining = sum(item["bytes"] for item in plan["files"] if not (destination / item["path"]).exists())
    if shutil.disk_usage(destination).free < remaining + 1024 * 1024:
        raise ArchiveError("Insufficient free network-volume space for the planned archive")
    try:
        copied = []
        for item in plan["files"]:
            source = resolve_artifact_path(root, item["path"])
            _safe_file(root, source)
            _safe_file(destination, destination / item["path"])
            target = resolve_artifact_path(destination, item["path"])
            digest = _copy_verified(source, target, item["bytes"])
            copied.append({**item, "sha256": digest, "status": "verified"})
            print(json.dumps({"event": "archived_file", "path": item["path"], "bytes": item["bytes"]}), flush=True)
        identity = {"checkpoint_directories": plan["checkpoint_directories"], "files": copied}
        archive_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest_path = destination / "archive-manifest.json"
        _safe_file(destination, manifest_path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("archive_id") != archive_id or manifest.get("status") != "verified":
                raise ArchiveError("Existing immutable archive manifest conflicts")
        else:
            manifest = {**plan, "files": copied, "archive_id": archive_id, "status": "verified",
                        "completed_at": time.time(), "verification": "sha256_source_and_destination"}
            payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            _publish_manifest(manifest_path, payload)
        payload = manifest_path.read_bytes()
        receipt = root / "archive" / f"{archive_id}.json"
        _safe_file(root, receipt)
        _publish_manifest(receipt, payload)
        return {**manifest, "destination_manifest_path": str(manifest_path), "local_manifest_path": str(receipt)}
    except ArchiveError:
        raise
    except Exception:
        raise ArchiveError("Archive copy failed; original files retained") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "copy", "stage-finals"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--include-selected", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = plan_archive(args.root)
        elif args.command == "stage-finals":
            if args.destination is None:
                raise ArchiveError("Staging requires a mounted network-volume destination")
            result = stage_final_checkpoints(args.root, args.destination, include_selected=args.include_selected)
        else:
            if args.destination is None:
                raise ArchiveError("Copy requires a mounted network-volume destination")
            result = archive_to_volume(args.root, args.destination)
            result = {key: result[key] for key in ("status", "experiment_status", "archive_scope", "missing_run_ids", "incomplete_run_ids",
                                                  "archive_id", "destination_manifest_path", "local_manifest_path", "total_bytes", "file_count")}
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False), flush=True)
        return 0
    except Exception as error:
        message = str(error) if isinstance(error, ArchiveError) else "Archive failed; diagnostic text withheld"
        print(json.dumps({"event": "archive_error", "error": message, "original_files_retained": True}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
